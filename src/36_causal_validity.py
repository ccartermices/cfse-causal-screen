# -*- coding: utf-8 -*-
"""Step 2-2: SAE 指标的「因果效度」主实验

问题: 常用 SAE 评估指标 (激活频率/幅度、重构贡献、上下文一致性/auto-interp 代理)
      能否预测一个特征是否**因果有用** —— 即对它做单特征 ablation/steering 时
      能否真实改变 ground-truth 任务上的行为?

流水线 (每个 (model, layer, task) 独立 resume):
  1. 任务构造: IOI (间接宾语识别) / Greater-Than(两数比较), 每模型做**单 token 校验**;
  2. 任务筛查: 无干预行为 B0 是否高于随机; 处于随机的组合标 at_chance 并跳过;
  3. 层对齐校准: 用 hook 抓 block 输出, 与 output_hidden_states 逐层比对,
     自动确定 hidden_states[L] <-> layers_mod[j] 的映射 (解决 27 的 off-by-one 隐患);
  4. 单特征干预: 编码任务激活 -> 按平均激活选 top-N 特征 ->
     ablation(v=0) / steering(v=alpha*s) -> C_ablate / C_steer / 翻转率;
     同时算标准指标 M1..M5;
  5. 基线: 随机特征 / 范数匹配随机方向 / 全特征 ablation / 整段残差 mean ablation;
  6. 相关性: spearman(M_i, C), 逐 (model,layer,task) + 汇总.

口径说明 (重要):
  - 干预公式: h_new = h + (decode(sparse2) - decode(sparse)) * std
      ablate:  sparse2[f]=0            -> delta = -sparse[:,f] * W_dec[:,f] * std
      steer :  sparse2[f]=alpha*s[:,f] -> delta = (alpha-1)*sparse[:,f]*W_dec[:,f]*std
  - 指标 (无干预, 都在任务 prompt 上算; M5 用项目激活语料):
      M1 act_freq_task   任务 token 中该特征被 topk 选中的比例
      M2 mean_act_task   任务 token 上该特征的平均激活值
      M3 max_act_task    任务 token 上该特征的最大激活值
      M4 recon_mse_gain  移除该特征带来的重构 MSE 增量 (越大=重构越关键)
      M5 ctx_coherence   该特征 top-10 激活文本的 TF-IDF 平均两两余弦 (auto-interp 代理)
      M6 corpus_freq     语料激活中该特征出现频率 (与任务无关的通用频率)
  注: Phase E/phase3 的 auto-interp 分数基于**跨层池化** SAE 的特征索引空间,
      与本次 per-layer SAE 不同, 无法复用, 故 M5 改为本地无 LLM 的上下文一致性代理。

资源: CPU 严格节流 (OMP/MKL=1, torch threads=1), 单进程串行, 独显, 不打扰同机其他实验。
输出: output/phaseStep2/causal_validity/{model}/layer{L}/{task}.json
      output/phaseStep2/causal_validity/summary.json
"""
import os

for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
           "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import re
import sys
import json
import random
import argparse
import numpy as np
import torch
from pathlib import Path
from datetime import datetime
from importlib import import_module

torch.set_num_threads(1)

from model_utils import load_config, resolve_model_cfg, apply_path_template, model_needs_fp8

_mod05 = import_module("05_train_text_sae")
LogSaver = _mod05.LogSaver

_mod27 = import_module("27_h6_semantic_injection")
find_decoder_layers = _mod27.find_decoder_layers
load_model_for_generation = _mod27.load_model_for_generation

_mod34 = import_module("34_step1_metric_validity")
spearman = _mod34.spearman
pearson = _mod34.pearson


class ResumeManager:
    def __init__(self, output_dir, script_name):
        self.output_dir = Path(output_dir)
        self.state_file = self.output_dir / f"resume_{script_name}.json"

    def load_state(self):
        if self.state_file.exists():
            with open(self.state_file, "r", encoding="utf-8") as f:
                return json.load(f)
        return {}

    def save_state(self, state):
        self.output_dir.mkdir(parents=True, exist_ok=True)
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(state, f, ensure_ascii=False, indent=2)

    def is_completed(self):
        return self.load_state().get("status") == "completed"

    def mark_completed(self, result_info=None):
        state = self.load_state()
        state["status"] = "completed"
        state["completed_at"] = datetime.now().isoformat()
        if result_info:
            state["result"] = result_info
        self.save_state(state)

    def reset(self):
        if self.state_file.exists():
            self.state_file.unlink()


# ----------------------------------------------------------------------------
# 任务构造
# ----------------------------------------------------------------------------
NAMES = ["Mary", "John", "Alice", "Bob", "Tom", "James", "Emma", "Sarah",
         "David", "Anna", "Paul", "Laura", "Peter", "Julia", "Mark", "Clara",
         "Henry", "Grace", "Adam", "Nina"]

IOI_TEMPLATES = [
    ("When {A} and {B} went to the store, {B} gave a drink to", False),  # giver=B -> IO=A
    ("When {A} and {B} went to the store, {A} gave a drink to", True),   # giver=A -> IO=B
    ("When {A} and {B} got to the store, {A} gave a bottle to", True),
    ("Then, {A} and {B} went to the store. {B} gave a drink to", False),
]

GT_TEMPLATES = [
    "Compare {x} and {y}: the larger number is",
    "Which number is larger, {x} or {y}? The larger number is",
]

# --- IOI 加难变体 (用于检验「IOI 的翻转率饱和能否靠换任务解除」) ---
# ioi_long: 名字子句与提问之间插入多个无关子句 -> 注意力稀释, 目标仍为 IO
IOI_LONG_TEMPLATES = [
    ("When {A} and {B} went to the busy store, {A} picked up some bread and "
     "looked at the shelves for a while, {B} walked around the aisles, and then "
     "{B} gave a drink to", False),
    ("After {A} and {B} walked all the way to the store and looked at many things "
     "on the shelves, {A} decided to buy something and gave a bottle to", True),
    ("{A} and {B} went to the store. {A} read the labels on all the boxes, "
     "{B} checked every price tag, and in the end {B} gave a drink to", False),
    ("{A} and {B} went to the store. {B} carried a heavy basket, {A} looked around "
     "the shop for a long time, and eventually {A} gave a bottle to", True),
]

# ioi_distract: 额外提及一个无关名字作为干扰实体, 目标仍为 IO
IOI_DISTRACT_TEMPLATES = [
    ("When {A} and {B} went to the store, {C} stayed at home, and then "
     "{B} gave a drink to", False),
    ("While {C} was reading at home, {A} and {B} went to the store and "
     "{A} gave a bottle to", True),
    ("{C} was waiting outside. When {A} and {B} went to the store, "
     "{B} gave a drink to", False),
    ("When {A} and {B} got to the store, {C} was still at work, and "
     "{A} gave a bottle to", True),
]


def _ids(tokenizer, text):
    return tokenizer(text, add_special_tokens=False)["input_ids"]


def space_token_id(tokenizer):
    """独立的空格 token id (这批 tokenizer 均存在且唯一)。"""
    i = _ids(tokenizer, " ")
    return i[0] if len(i) == 1 else None


def build_ioi(tokenizer, n_prompts, seed=42):
    """IOI: 答案 token = `" " + 名字` 的**最后一个** token, 其余部分作显式前缀。

    不同 tokenizer 对 " Mary" 切分不同 (Qwen 合成 1 个 token; Phi 切成 [空格, Mary]),
    统一取末位为目标、前缀显式喂入, 保证末位 logits 恰好预测名字 token 且不越出分布。
    """
    rng = random.Random(seed)
    norm = {nm: _ids(tokenizer, " " + nm) for nm in NAMES}
    valid = [nm for nm in NAMES if norm[nm]]
    if len(valid) < 4:
        return None, {"error": f"可用名字不足: {len(valid)}"}
    items = []
    for _ in range(n_prompts):
        A, B = rng.sample(valid, 2)
        tmpl, giver_is_first = rng.choice(IOI_TEMPLATES)
        prompt = tmpl.format(A=A, B=B)
        io, s = (B, A) if giver_is_first else (A, B)
        a_io, a_s = norm[io], norm[s]
        if a_io[-1] == a_s[-1]:
            continue
        items.append({"prompt": prompt, "prefix_ids": a_io[:-1],
                      "correct_id": a_io[-1], "distractor_id": a_s[-1],
                      "correct": io, "distractor": s})
    return items, {"valid_names": valid, "n": len(items),
                   "space_id": space_token_id(tokenizer)}


def build_gt(tokenizer, n_prompts, seed=42):
    """Greater-Than: 前缀 = 空格 token(s), 目标 = 其后的首个数字 token。

    各 tokenizer 对 " 47" 的切分差异很大 (Qwen: [空格,"4","7"]; tinyllama/phi:
    [▁,"4","7"]; minicpm: [空格, 整数字])。取所有 `" " + str(n)` 的**公共前缀**
    作为显式前缀, 其后首个 token 作为评分目标, 即可统一适配且要求两数十位不同。
    """
    rng = random.Random(seed)
    seqs = {}
    for n in range(10, 100):
        t = _ids(tokenizer, " " + str(n))
        if len(t) >= 2:
            seqs[n] = t
    if len(seqs) < 8:
        return None, {"error": f"数字 token 不足: {len(seqs)}"}
    all_seqs = list(seqs.values())
    plen = 0
    while (all(len(s) > plen + 1 for s in all_seqs)
           and len({s[plen] for s in all_seqs}) == 1):
        plen += 1
    prefix = all_seqs[0][:plen]
    pool = [(n, s[plen]) for n, s in seqs.items()]
    items = []
    tries = 0
    while len(items) < n_prompts and tries < n_prompts * 40:
        tries += 1
        (x, tx), (y, ty) = rng.sample(pool, 2)
        if tx == ty:
            continue
        tmpl = rng.choice(GT_TEMPLATES)
        prompt = tmpl.format(x=x, y=y)
        big, small = (x, y) if x > y else (y, x)
        bid, sid = (tx, ty) if x > y else (ty, tx)
        items.append({"prompt": prompt, "prefix_ids": list(prefix),
                      "correct_id": bid, "distractor_id": sid,
                      "correct": str(big), "distractor": str(small)})
    return items, {"number_pool": [n for n, _ in pool], "n": len(items),
                   "prefix_len": plen, "prefix_ids": list(prefix),
                   "distinct_target_kinds": len({t for _, t in pool})}


# ----------------------------------------------------------------------------
# 自然文本任务 (非合成): 真实维基百科文章的下一个词预测
# ----------------------------------------------------------------------------
NATURAL_CORPUS = "wikipedia_en"
NATURAL_MAX_PREFIX = 120          # 与 _encode_with_prefix 的 128 token 窗口对齐


def _strip_marker(tstr, marker):
    if marker and tstr.startswith(marker):
        return tstr[len(marker):]
    return tstr


def _wordlike(tstr, marker):
    """词首实词: 去掉空格标记后为纯 ASCII 字母且长度 >= 2。"""
    s = _strip_marker(tstr, marker)
    return s.isascii() and s.isalpha() and len(s) >= 2


def _detect_space_marker(tokenizer):
    """byte-level BPE 用 "Ġ", sentencepiece 用 "▁"; 都没有则返回 None。"""
    for tid in range(min(len(tokenizer), 20000)):
        t = tokenizer.convert_ids_to_tokens(tid)
        if isinstance(t, str):
            if t.startswith("Ġ"):
                return "Ġ"
            if t.startswith("▁"):
                return "▁"
    return None


def _natural_texts(corpus, max_articles=400):
    d = Path(__file__).resolve().parent / "data" / "text" / corpus
    if not d.exists():
        return []
    texts = []
    for p in sorted(d.glob("*.json")):
        try:
            with open(p, "r", encoding="utf-8") as f:
                obj = json.load(f)
        except Exception:
            continue
        txt = obj.get("text") if isinstance(obj, dict) else None
        if isinstance(txt, str) and len(txt) > 200:
            texts.append(txt)
        if len(texts) >= max_articles:
            break
    return texts


def _unigram_stats(tokenizer, texts, marker, n_docs=200):
    """返回 (counts, boundary_ids): 词频 + 「在语料中确为完整词」的 token id 集合。

    完整词判据: 该 token 后面紧跟词首 token (含空格标记) 或非字母数字 token。
    用于排除 "▁charg" 这类只在词内部出现的碎片 token 充当目标/foil。
    """
    counts, boundary = None, set()
    for t in texts[:n_docs]:
        ids = _ids(tokenizer, t)
        if not ids:
            continue
        arr = np.bincount(np.asarray(ids, dtype=np.int64))
        if counts is None:
            counts = arr
        else:
            if len(arr) > len(counts):
                counts = np.pad(counts, (0, len(arr) - len(counts)))
            counts[:len(arr)] += arr
        toks = tokenizer.convert_ids_to_tokens(ids)
        for i in range(len(ids) - 1):
            nxt = toks[i + 1]
            if not isinstance(nxt, str):
                continue
            if (marker and nxt.startswith(marker)) or not nxt.lstrip("Ġ▁").isalnum():
                boundary.add(int(ids[i]))
    return counts, boundary


def build_natural(tokenizer, n_prompts, seed=42, corpus=NATURAL_CORPUS,
                  max_prefix_tokens=NATURAL_MAX_PREFIX, n_count_docs=200,
                  min_word_count=3):
    """自然文本 next-token 任务 (非合成, held-out 语料).

    prompt 为空; prefix_ids = 文章真实词序列的前 k 个 token (k∈[32,120],
    锚定在词首实词上); correct_id = 该位置的真实下一词;
    distractor = 同语料里**词频匹配**的另一个词首实词 (unigram count ∈ [c/2, 2c])。
    B = logit(真实词) − logit(频率匹配 foil): 排除用高频虚词充当 foil 导致 B0 饱和。
    目标与 foil 都要求是语料中的**完整词**(非词内碎片), 且词频 >= min_word_count。
    前 k <= 120 以适配 `_encode_with_prefix` 的 128-token 窗口 (不做静默截断)。
    """
    rng = random.Random(seed)
    texts = _natural_texts(corpus)
    if len(texts) < 5:
        return None, {"error": f"自然语料不足: {corpus} -> {len(texts)} 篇"}
    marker = _detect_space_marker(tokenizer)
    counts, boundary = _unigram_stats(tokenizer, texts, marker, n_count_docs)
    if counts is None:
        return None, {"error": f"语料 tokenize 失败: {corpus}"}

    nonempty = np.nonzero(counts)[0].tolist()
    toks = tokenizer.convert_ids_to_tokens(nonempty)
    word_ids, word_freq = [], []
    for tid, tstr in zip(nonempty, toks):
        if (int(counts[tid]) >= min_word_count and int(tid) in boundary
                and isinstance(tstr, str) and _wordlike(tstr, marker)):
            word_ids.append(int(tid))
            word_freq.append(int(counts[tid]))
    if len(word_ids) < 50:
        return None, {"error": f"词首实词不足: {len(word_ids)} (marker={marker})"}
    word_ids = np.asarray(word_ids, dtype=np.int64)
    word_freq = np.asarray(word_freq, dtype=np.int64)

    items, used = [], set()
    tries = 0
    while len(items) < n_prompts and tries < n_prompts * 60:
        tries += 1
        ti = rng.randrange(len(texts))
        ids = _ids(tokenizer, texts[ti])[:max_prefix_tokens + 8]
        kmax = min(len(ids) - 8, max_prefix_tokens)
        if kmax <= 32:
            continue
        k = rng.randint(32, kmax)
        if (ti, k) in used:
            continue
        cid = int(ids[k])
        if cid >= len(counts) or counts[cid] < min_word_count or cid not in boundary:
            continue
        ct = tokenizer.convert_ids_to_tokens(cid)
        if not isinstance(ct, str) or not _wordlike(ct, marker):
            continue
        c = float(counts[cid])
        lo, hi = c / 2.0, c * 2.0
        cand = np.where((word_freq >= lo) & (word_freq <= hi) & (word_ids != cid))[0]
        relax = 0
        while len(cand) == 0 and relax < 4:
            relax += 1
            lo, hi = c / (2.0 ** (relax + 1)), c * (2.0 ** (relax + 1))
            cand = np.where((word_freq >= lo) & (word_freq <= hi) & (word_ids != cid))[0]
        if len(cand) == 0:
            continue
        did = int(word_ids[rng.choice(cand.tolist())])
        if did == cid:
            continue
        dt = tokenizer.convert_ids_to_tokens(did)
        used.add((ti, k))
        items.append({"prompt": "", "prefix_ids": [int(x) for x in ids[:k]],
                      "correct_id": cid, "distractor_id": did,
                      "correct": _strip_marker(ct, marker),
                      "distractor": _strip_marker(dt, marker),
                      "source_idx": ti, "pos": k})
    meta = {"corpus": corpus, "n": len(items), "n_corpus_texts": len(texts),
            "n_count_docs": min(len(texts), n_count_docs), "space_marker": marker,
            "word_vocab": int(len(word_ids)), "max_prefix_tokens": max_prefix_tokens,
            "foil": "frequency_matched", "held_out_note": "wikipedia_en 未参与该模型 SAE 训练"}
    if not items:
        meta["error"] = f"自然任务构造失败 (tries={tries})"
        return None, meta
    return items, meta


def _build_ioi_like(tokenizer, n_prompts, templates, n_extra_names=0, seed=42):
    """IOI 变体通用构造: 指定模板集与额外干扰名字数, 返回结构与 build_ioi 完全一致,
    以便 36/38/40 复用同一套行为度量与干预管线 (只换任务名, 不换口径)。
    """
    rng = random.Random(seed)
    norm = {nm: _ids(tokenizer, " " + nm) for nm in NAMES}
    valid = [nm for nm in NAMES if norm[nm]]
    need = 2 + n_extra_names
    if len(valid) < need:
        return None, {"error": f"可用名字不足: {len(valid)} < {need}"}
    items = []
    for _ in range(n_prompts):
        picks = rng.sample(valid, need)
        A, B = picks[0], picks[1]
        C = picks[2] if n_extra_names else None
        tmpl, giver_is_first = rng.choice(templates)
        prompt = tmpl.format(A=A, B=B, C=C) if C is not None else tmpl.format(A=A, B=B)
        io, s = (B, A) if giver_is_first else (A, B)
        a_io, a_s = norm[io], norm[s]
        if a_io[-1] == a_s[-1]:
            continue
        if C is not None and norm[C][-1] in (a_io[-1], a_s[-1]):
            continue
        items.append({"prompt": prompt, "prefix_ids": a_io[:-1],
                      "correct_id": a_io[-1], "distractor_id": a_s[-1],
                      "correct": io, "distractor": s})
    return items, {"valid_names": valid, "n": len(items),
                   "space_id": space_token_id(tokenizer)}


def build_ioi_anti(tokenizer, n_prompts, seed=42):
    """反事实 IOI: prompt 与 build_ioi **逐条相同**, 但目标换成 giver(主语)。

    同分布输入 + 目标取反, 用来把「任务难度」与「prompt 选样」两种解释分开:
    若反事实目标的边际大幅缩小, 说明饱和来自任务本身太容易, 而不是 prompt 不够难。
    """
    items, meta = build_ioi(tokenizer, n_prompts, seed=seed)
    if not items:
        return items, meta
    out = []
    for it in items:
        j = dict(it)
        j["correct_id"], j["distractor_id"] = it["distractor_id"], it["correct_id"]
        j["correct"], j["distractor"] = it["distractor"], it["correct"]
        out.append(j)
    meta = dict(meta or {})
    meta["variant"] = "anti"
    meta["n"] = len(out)
    return out, meta


def build_ioi_long(tokenizer, n_prompts, seed=42):
    """长上下文 IOI: 名字子句与提问之间插入无关子句 (注意力稀释), 目标仍为 IO。"""
    return _build_ioi_like(tokenizer, n_prompts, IOI_LONG_TEMPLATES, 0, seed)


def build_ioi_distract(tokenizer, n_prompts, seed=42):
    """三名字 IOI: 额外提及一个无关名字作为干扰实体, 目标仍为 IO。"""
    return _build_ioi_like(tokenizer, n_prompts, IOI_DISTRACT_TEMPLATES, 1, seed)


# ----------------------------------------------------------------------------
# 非-IOI 任务: 句法一致 / 事实检索 / 词汇反义
#
# 动机: 既有任务全部属于 IOI 家族 (ioi/anti/long/distract) 或数字比较 (gt),
# 「解析预测器贴测量天花板」这一结论若只在模板化检索上成立, 外部效度就不足。
# 三个新任务各对应一种不同机制:
#   sva      句法一致 —— 主谓一致 + attractor (干扰名词的数与主语相反)
#   capital  参数化知识 —— 国家 -> 首都, 靠记忆检索而非句内指代
#   antonym  词汇语义 —— 反义关系, 答案由词对表固定, 不依赖世界事实
# 三者都要求目标词能写成「公共前缀 + 单 token」(见 _target_pool), 否则末位 logits
# 无法直接比较; 68_task_probe.py 已在 10 个 tokenizer 上验证可行性。
# ----------------------------------------------------------------------------
SVA_NOUN_PL = ["keys", "dogs", "books", "cars", "students", "teachers",
               "birds", "friends", "tables", "chairs", "windows", "letters"]
SVA_NOUN_SG = ["key", "dog", "book", "car", "student", "teacher",
               "bird", "friend", "table", "chair", "window", "letter"]
SVA_ATTR_PL = ["doors", "boxes", "walls", "trees", "desks", "shelves"]
SVA_ATTR_SG = ["door", "box", "wall", "tree", "desk", "shelf"]
SVA_TEMPLATES = ["The {subj} near the {attr}",
                 "The {subj} behind the {attr}",
                 "The {subj} beside the {attr}",
                 "The {subj} inside the {attr}",
                 "The {subj} next to the {attr}"]

CAP_PAIRS = [("France", "Paris"), ("Japan", "Tokyo"), ("Germany", "Berlin"),
             ("Italy", "Rome"), ("Spain", "Madrid"), ("Russia", "Moscow"),
             ("Egypt", "Cairo"), ("Canada", "Ottawa"), ("China", "Beijing"),
             ("the United Kingdom", "London"), ("Greece", "Athens"),
             ("Portugal", "Lisbon"), ("Austria", "Vienna"),
             ("Poland", "Warsaw"), ("Sweden", "Stockholm"),
             ("Norway", "Oslo"), ("Denmark", "Copenhagen"),
             ("Ireland", "Dublin")]
CAP_REGION = {"France": "eu", "Germany": "eu", "Italy": "eu", "Spain": "eu",
              "Russia": "eu", "Greece": "eu", "Portugal": "eu",
              "Austria": "eu", "Poland": "eu", "Sweden": "eu",
              "Norway": "eu", "Denmark": "eu", "Ireland": "eu",
              "the United Kingdom": "eu", "Japan": "as", "China": "as",
              "Egypt": "af", "Canada": "na"}
CAP_TEMPLATES = ["The capital of {c} is",
                 "The capital city of {c} is"]

ANT_PAIRS = [("hot", "cold"), ("big", "small"), ("up", "down"),
             ("fast", "slow"), ("high", "low"), ("young", "old"),
             ("long", "short"), ("early", "late"), ("open", "closed"),
             ("hard", "soft"), ("rich", "poor"), ("full", "empty"),
             ("dark", "light"), ("near", "far"), ("wet", "dry"),
             ("true", "false"), ("strong", "weak"), ("cheap", "expensive"),
             ("clean", "dirty"), ("happy", "sad")]
ANT_TEMPLATES = ["The opposite of {w} is",
                 "The opposite of the word {w} is"]


def _target_pool(tokenizer, words):
    """把词表压成「共享前缀 + 各自唯一末 token」的最大可用子集。

    返回 (prefix_ids, {word: last_id}, 剔除明细)。不可用时 prefix 为 None。
    与 build_gt 的公共前缀法同一思路, 但允许词表部分失败 (不同 tokenizer 对
    " Cairo" / " walks" 的切分长度不同), 只保留长度一致且末 token 不冲突的词。
    """
    seqs = {}
    for w in words:
        t = _ids(tokenizer, " " + w)
        if t:
            seqs[w] = t
    if len(seqs) < 2:
        return None, {}, {"error": f"可 tokenize 的词不足: {len(seqs)}"}
    # 按序列长度分组, 取最大组 (通常是单 token 或 [空格, 词] 两类)
    cnt = {}
    for s in seqs.values():
        cnt[len(s)] = cnt.get(len(s), 0) + 1
    L = max(cnt, key=lambda k: (cnt[k], -k))
    dropped = [w for w, s in seqs.items() if len(s) != L]
    seqs = {w: s for w, s in seqs.items() if len(s) == L}
    if len(seqs) < 2:
        return None, {}, {"error": "同长度词不足", "dropped": dropped}
    # 前缀: 所有序列共有的前 L-1 个 token 必须完全一致
    plen = 0
    while plen < L - 1 and len({s[plen] for s in seqs.values()}) == 1:
        plen += 1
    if plen != L - 1:
        return None, {}, {"error": "前缀不统一", "dropped": dropped}
    last = {w: s[plen] for w, s in seqs.items()}
    keep, seen = {}, {}
    for w, tid in last.items():
        if tid in seen:
            dropped.append(w)
            continue
        seen[tid] = w
        keep[w] = tid
    if len(keep) < 2:
        return None, {}, {"error": "末 token 全部冲突", "dropped": dropped}
    pre = list(next(iter(seqs.values()))[:plen])
    return pre, keep, {"prefix_len": plen, "kept": len(keep),
                       "dropped": dropped}


def build_sva(tokenizer, n_prompts, seed=42):
    """主谓一致 (SVA): 复数主语 + 单数 attractor, 或单数主语 + 复数 attractor。

    prompt 形如 "The keys near the door" (关系从句省略关系代词), 目标是紧随的
    be 动词。attractor 的数与主语相反, 因此「按最近名词选动词」会答错 ——
    这正是 SVA 范式要测的干扰。
    correct/distractor 在两类各占一半, 避免正确答案恒为同一个词
    (否则全局 unigram 偏好会主导 B, 而非句法依赖)。
    """
    pre, last, info = _target_pool(tokenizer, ["is", "are"])
    if pre is None:
        return None, {"error": f"is/are 不可用: {info}"}
    rng = random.Random(seed)
    items, tries = [], 0
    while len(items) < n_prompts and tries < n_prompts * 40:
        tries += 1
        plural_subj = (len(items) % 2 == 0)
        if plural_subj:
            subj, attr = rng.choice(SVA_NOUN_PL), rng.choice(SVA_ATTR_SG)
            correct, distract = "are", "is"
        else:
            subj, attr = rng.choice(SVA_NOUN_SG), rng.choice(SVA_ATTR_PL)
            correct, distract = "is", "are"
        prompt = rng.choice(SVA_TEMPLATES).format(subj=subj, attr=attr)
        items.append({"prompt": prompt, "prefix_ids": list(pre),
                      "correct_id": last[correct],
                      "distractor_id": last[distract],
                      "correct": correct, "distractor": distract,
                      "subj_plural": plural_subj})
    meta = {"n": len(items), "prefix_len": len(pre),
            "frac_plural_subj": sum(i["subj_plural"] for i in items) / max(len(items), 1),
            "n_templates": len(SVA_TEMPLATES)}
    return items, meta


def build_capital(tokenizer, n_prompts, seed=42):
    """事实检索: "The capital of {country} is" -> 首都。

    distractor 优先取**同区域**的另一首都 (欧洲->欧洲), 否则任意;
    这样干扰项在词频/语域上与正确项可比, B 不会因「首都 vs 随机词」而虚高。
    """
    pre, last, info = _target_pool(tokenizer, [c for _, c in CAP_PAIRS])
    if pre is None:
        return None, {"error": f"首都词表不可用: {info}"}
    pool = [(co, ci) for co, ci in CAP_PAIRS if ci in last]
    if len(pool) < 6:
        return None, {"error": f"可用国家不足: {len(pool)}"}
    by_region = {}
    for co, ci in pool:
        by_region.setdefault(CAP_REGION.get(co, "?"), []).append((co, ci))
    rng = random.Random(seed)
    items = []
    for _ in range(n_prompts):
        co, ci = rng.choice(pool)
        region = by_region.get(CAP_REGION.get(co, "?"), [])
        cands = [x for x in region if x[1] != ci] or [x for x in pool if x[1] != ci]
        _, di = rng.choice(cands)
        prompt = rng.choice(CAP_TEMPLATES).format(c=co)
        items.append({"prompt": prompt, "prefix_ids": list(pre),
                      "correct_id": last[ci], "distractor_id": last[di],
                      "correct": ci, "distractor": di,
                      "same_region": CAP_REGION.get(co) == CAP_REGION.get(
                          [x for x in pool if x[1] == di][0][0], "?")})
    meta = {"n": len(items), "prefix_len": len(pre),
            "n_countries": len(pool),
            "frac_same_region": sum(i["same_region"] for i in items) / max(len(items), 1),
            "dropped": info.get("dropped", [])}
    return items, meta


def build_antonym(tokenizer, n_prompts, seed=42):
    """词汇反义: "The opposite of {w} is" -> 该词的反义词。

    distractor 从**其他词对**里取 (避免取到同义/近义而污染 B), 且排除任何
    与正确答案相同的词 (不同词对可能共享词形)。
    """
    words = sorted({w for p in ANT_PAIRS for w in p})
    pre, last, info = _target_pool(tokenizer, words)
    if pre is None:
        return None, {"error": f"反义词表不可用: {info}"}
    pairs = [(a, b) for a, b in ANT_PAIRS if a in last and b in last]
    if len(pairs) < 6:
        return None, {"error": f"可用词对不足: {len(pairs)}"}
    rng = random.Random(seed)
    items = []
    for _ in range(n_prompts):
        w, ans = rng.choice(pairs)
        if rng.random() < 0.5:                    # 双向: 也可问 cold 的反义
            w, ans = ans, w
        others = [x for x in last if x not in (w, ans)]
        di = rng.choice(others)
        prompt = rng.choice(ANT_TEMPLATES).format(w=w)
        items.append({"prompt": prompt, "prefix_ids": list(pre),
                      "correct_id": last[ans], "distractor_id": last[di],
                      "correct": ans, "distractor": di,
                      "cue": w})
    meta = {"n": len(items), "prefix_len": len(pre),
            "n_pairs": len(pairs), "n_words": len(last),
            "dropped": info.get("dropped", [])}
    return items, meta


TASK_BUILDERS = {"ioi": build_ioi, "gt": build_gt,
                 "ioi_anti": build_ioi_anti,
                 "ioi_long": build_ioi_long,
                 "ioi_distract": build_ioi_distract,
                 "natural": build_natural,
                 "sva": build_sva,
                 "capital": build_capital,
                 "antonym": build_antonym}


# ----------------------------------------------------------------------------
# 层对齐校准
# ----------------------------------------------------------------------------
def calibrate_layer_map(model, tokenizer, layers_mod, device, probe_text):
    n = len(layers_mod)
    caps = {}
    handles = []
    for j in range(n):
        def mk(jj):
            def h(module, args, output):
                hs = output[0] if isinstance(output, (tuple, list)) else output
                caps[jj] = hs.detach().float()
            return h
        handles.append(layers_mod[j].register_forward_hook(mk(j)))
    try:
        enc = tokenizer(probe_text, return_tensors="pt", truncation=True, max_length=64).to(device)
        with torch.no_grad():
            out = model(**enc, output_hidden_states=True, use_cache=False)
    finally:
        for hd in handles:
            hd.remove()
    hs = out.hidden_states
    mapping = {}
    diffs = {}
    for k in range(1, len(hs)):
        target = hs[k].detach().float()
        best_j, best_d = None, float("inf")
        for j in range(n):
            if j not in caps or caps[j].shape != target.shape:
                continue
            d = float((caps[j] - target).abs().max().item())
            if d < best_d:
                best_j, best_d = j, d
        if best_j is not None:
            mapping[str(k)] = {"block_idx": best_j, "max_abs_diff": best_d}
            diffs[k] = best_d
    return mapping


# ----------------------------------------------------------------------------
# 单特征 hook
# ----------------------------------------------------------------------------
def make_feature_hook(sae, mean_t, std_t, feat_idx, mode, alpha):
    """单特征 ablation / steering hook (在层 block 输出处修改残差流)."""
    W_f = sae.decoder.weight[:, feat_idx].detach().float()  # (input_dim,)

    def hook(module, args, output):
        hs = output[0] if isinstance(output, (tuple, list)) else output
        orig_dtype = hs.dtype
        h = hs.float()
        with torch.no_grad():
            normed = (h - mean_t) / std_t
            sparse, _ = sae.encode(normed)
            s_f = sparse[..., feat_idx:feat_idx + 1]          # (B, seq, 1)
        if mode == "ablate":
            coef = -s_f
        else:                                                  # steer
            coef = (alpha - 1.0) * s_f
        delta = coef * W_f.view(1, 1, -1) * std_t.view(1, 1, -1)
        h_new = (h + delta).to(orig_dtype)
        if isinstance(output, (tuple, list)):
            return (h_new,) + tuple(output[1:])
        return h_new

    return hook


def make_mean_ablate_hook(mean_t):
    """整段残差 mean ablation: block 输出替换为数据均值向量."""
    def hook(module, args, output):
        hs = output[0] if isinstance(output, (tuple, list)) else output
        h_new = mean_t.view(1, 1, -1).expand_as(hs).to(hs.dtype)
        if isinstance(output, (tuple, list)):
            return (h_new,) + tuple(output[1:])
        return h_new
    return hook


def make_all_feature_ablate_hook(sae, mean_t, std_t):
    """全特征 ablation: 去掉整个 SAE 重建成分 (保留 h - recon)."""
    def hook(module, args, output):
        hs = output[0] if isinstance(output, (tuple, list)) else output
        orig_dtype = hs.dtype
        h = hs.float()
        with torch.no_grad():
            normed = (h - mean_t) / std_t
            sparse, _ = sae.encode(normed)
            recon = sae.decode(sparse)
        h_new = (h - recon * std_t.view(1, 1, -1)).to(orig_dtype)
        if isinstance(output, (tuple, list)):
            return (h_new,) + tuple(output[1:])
        return h_new
    return hook


def make_norm_matched_random_hook(sae, mean_t, std_t, rand_dir, pair_feat):
    """范数匹配随机方向: 用随机特征的激活强度作用于随机单位方向."""
    W_pair = sae.decoder.weight[:, pair_feat].detach().float()
    d = rand_dir / (rand_dir.norm() + 1e-8)

    def hook(module, args, output):
        hs = output[0] if isinstance(output, (tuple, list)) else output
        orig_dtype = hs.dtype
        h = hs.float()
        with torch.no_grad():
            normed = (h - mean_t) / std_t
            sparse, _ = sae.encode(normed)
            s = sparse[..., pair_feat:pair_feat + 1]
            # 范数匹配: 用 pair 特征的 decoder 方向长度标定随机方向
            scale = (s * W_pair.norm()).abs()
        delta = -scale * d.view(1, 1, -1) * std_t.view(1, 1, -1)
        h_new = (h + delta).to(orig_dtype)
        if isinstance(output, (tuple, list)):
            return (h_new,) + tuple(output[1:])
        return h_new

    return hook


# ----------------------------------------------------------------------------
# 行为度量
# ----------------------------------------------------------------------------
def _encode_with_prefix(tokenizer, items, device, max_length=128):
    """构造 input_ids = tokenizer(prompt) + item["prefix_ids"], 左 padding.

    这样末位 logits 恰好落在 prompt 末尾(对 IOI 是名字前的空格位置、
    对 GT 是数字前的位置), 与 prefix_ids 一起保证评分目标是唯一的下一 token。
    """
    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0
    seqs = []
    for it in items:
        s = tokenizer(it["prompt"], add_special_tokens=True)["input_ids"] + list(it["prefix_ids"])
        seqs.append(s[-max_length:])
    L = max(len(s) for s in seqs)
    ids, attn = [], []
    for s in seqs:
        k = L - len(s)
        ids.append([pad_id] * k + list(s))
        attn.append([0] * k + [1] * len(s))
    return (torch.tensor(ids, dtype=torch.long, device=device),
            torch.tensor(attn, dtype=torch.long, device=device))


def batch_behavior(model, tokenizer, items, layer_mod, hook_fn, device,
                   batch_size=16):
    """返回 (per-prompt logit diff 列表, 平均 B)."""
    outs = []
    for s in range(0, len(items), batch_size):
        chunk = items[s:s + batch_size]
        input_ids, attn = _encode_with_prefix(tokenizer, chunk, device)
        handle = layer_mod.register_forward_hook(hook_fn) if (hook_fn and layer_mod) else None
        try:
            with torch.no_grad():
                logits = model(input_ids=input_ids,
                               attention_mask=attn).logits[:, -1, :].float()
        finally:
            if handle:
                handle.remove()
        cid = torch.tensor([it["correct_id"] for it in chunk], device=logits.device)
        did = torch.tensor([it["distractor_id"] for it in chunk], device=logits.device)
        diff = (logits.gather(1, cid[:, None]) - logits.gather(1, did[:, None])).squeeze(1)
        outs.extend(diff.cpu().tolist())
    return outs, float(np.mean(outs)) if outs else 0.0


def capture_sparse(model, tokenizer, items, layer_mod, sae, mean_t, std_t, device,
                   batch_size=16):
    """跑任务 prompt, 收集该层编码后的 sparse 矩阵 (行=token)."""
    all_sparse = []
    for s in range(0, len(items), batch_size):
        chunk = items[s:s + batch_size]
        input_ids, attn = _encode_with_prefix(tokenizer, chunk, device)
        cap = {}

        def hk(module, args, output):
            hs = output[0] if isinstance(output, (tuple, list)) else output
            cap["h"] = hs.detach().float()

        handle = layer_mod.register_forward_hook(hk)
        try:
            with torch.no_grad():
                model(input_ids=input_ids, attention_mask=attn)
        finally:
            handle.remove()
        h = cap["h"]
        mask = attn.unsqueeze(-1).float()
        with torch.no_grad():
            normed = (h - mean_t) / std_t
            sparse, _ = sae.encode(normed)
        n_tok = int(mask.sum().item())
        all_sparse.append((sparse.cpu().numpy().reshape(-1, sparse.shape[-1]),
                           mask.cpu().numpy().reshape(-1) > 0, n_tok))
    return all_sparse


def tfidf_coherence(texts):
    """top 激活文本的 TF-IDF 平均两两余弦 (auto-interp 代理, 无 LLM)."""
    if len(texts) < 2:
        return None
    docs = [re.findall(r"[a-z0-9]+", t.lower()) for t in texts]
    vocab = sorted({w for d in docs for w in d})
    if not vocab:
        return None
    vi = {w: i for i, w in enumerate(vocab)}
    df = np.zeros(len(vocab))
    for d in docs:
        for w in set(d):
            df[vi[w]] += 1
    idf = np.log((1.0 + len(docs)) / (1.0 + df)) + 1.0
    vecs = np.zeros((len(docs), len(vocab)))
    for i, d in enumerate(docs):
        if not d:
            continue
        for w in d:
            vecs[i, vi[w]] += 1.0
        vecs[i] /= len(d)
    vecs *= idf
    nrm = np.linalg.norm(vecs, axis=1, keepdims=True)
    nrm[nrm == 0] = 1.0
    vecs /= nrm
    sims = vecs @ vecs.T
    iu = np.triu_indices(len(docs), k=1)
    return float(sims[iu].mean())


# ----------------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------------
def load_per_layer_sae(base_dir, m, layer, device, sae_root=None):
    """sae_root 允许指定替代 SAE 目录（用于 SAE 训练种子方差实验, C14）。"""
    root = base_dir / "output" / "phaseStep2" / (sae_root or "layer_sae")
    p = root / m / f"layer{layer}" / "best_sae.pt"
    if not p.exists():
        return None
    ckpt = torch.load(p, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    sae = _mod05.BatchTopKSAE(cfg["input_dim"], cfg["hidden_dim"], cfg["k"]).to(device)
    sae.load_state_dict(ckpt["model_state_dict"])
    sae.eval()
    mean = np.asarray(ckpt["mean"], dtype=np.float32)
    std = np.asarray(ckpt["std"], dtype=np.float32)
    return sae, mean, std


def load_corpus(base_dir, config, m, layer):
    """返回该层语料 (texts, raw_activations (N, hidden))."""
    act_dir = base_dir / apply_path_template(config["paths"]["text_activation_dir"], m)
    js = sorted(act_dir.glob("text_activations_*.json"))
    if not js:
        return None, None
    with open(js[-1], "r", encoding="utf-8") as f:
        data = json.load(f)
    texts, acts = [], []
    for rec in data:
        if rec.get("layer") != layer:
            continue
        a = np.asarray(rec["activations"], dtype=np.float32)
        for r in a:
            acts.append(r)
            texts.append(rec.get("text", ""))
    if not acts:
        return None, None
    return texts, np.stack(acts, axis=0)


def run_one(args, config, base_dir, m, layer, task):
    _, model_cfg = resolve_model_cfg(config, m)
    out_dir = (base_dir / "output" / "phaseStep2"
               / ("causal_validity" + getattr(args, "out_tag", "")) / m / f"layer{layer}")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / f"{task}.json"
    resume = ResumeManager(out_dir, f"36_{task}_layer{layer}")
    if resume.is_completed() and out_file.exists() and not args.force:
        print(f"  [跳过] {m} layer{layer} {task} 已完成")
        return json.load(open(out_file, encoding="utf-8"))

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    result = {"model": m, "layer": layer, "task": task,
              "timestamp": datetime.now().isoformat()}

    loaded = load_per_layer_sae(base_dir, m, layer, device, getattr(args, "sae_root", None))
    if loaded is None:
        result["error"] = "缺少 per-layer SAE (先跑 35_train_per_layer_sae.py)"
        _save(out_file, result)
        resume.mark_completed({"error": result["error"]})
        return result
    sae, mean_np, std_np = loaded
    mean_t = torch.tensor(mean_np, dtype=torch.float32, device=device)
    std_t = torch.tensor(std_np, dtype=torch.float32, device=device)

    model_path = base_dir / model_cfg["local_path"]
    if not model_path.exists():
        for p in (base_dir / "models").glob("*/*"):
            if p.is_dir() and (p / "config.json").exists():
                model_path = p
                break
    print(f"\n[{m} L{layer} {task}] 加载基座模型...")
    if model_needs_fp8(model_cfg, model_path):
        # 原生 fp8 模型: finegrained-fp8 kernel 已在本地缓存, 置离线模式即可让
        # kernels/utils.py 跳过 publisher 信任联网校验 (其读取 HF_HUB_OFFLINE)。
        try:
            from huggingface_hub import constants as _hf_const
            _hf_const.HF_HUB_OFFLINE = True
            print("  [fp8] 已置 HF_HUB_OFFLINE=True 以绕过 kernel 信任联网检查")
        except Exception as _e:
            print(f"  [fp8] 设置离线模式失败: {_e}")
        try:
            os.environ.setdefault(
                "TRITON_CACHE_DIR",
                os.path.join(os.path.expanduser("~"), ".cache", "triton"))
            print(f"  [fp8] TRITON_CACHE_DIR={os.environ['TRITON_CACHE_DIR']}")
        except Exception as _e:
            print(f"  [fp8] 设置 TRITON_CACHE_DIR 失败: {_e}")
        try:
            import huggingface_hub._snapshot_download as _sd
            # 本地缓存只含 build/torch-cuda 变体, 但 tree listing 记录了全量文件,
            # 离线时 snapshot_download 会因 "快照不完整" 直接报错。
            # read_tree_cache 返回 None 即跳过该完整性检查 (仅离线使用缓存, 不影响下载)。
            _sd.read_tree_cache = lambda *a, **k: None
            print("  [fp8] 已跳过 snapshot 完整性检查 (离线仅用 torch-cuda 变体)")
        except Exception as _e:
            print(f"  [fp8] patch read_tree_cache 失败: {_e}")
    model, tokenizer = load_model_for_generation(base_dir, model_cfg, model_path)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    layers_mod = find_decoder_layers(model)
    result["decoder_layers"] = len(layers_mod)

    probe = "When Mary and John went to the store, John gave a drink to"
    lmap = calibrate_layer_map(model, tokenizer, layers_mod, device, probe)
    result["layer_map"] = lmap
    if str(layer) not in lmap:
        result["error"] = f"层对齐失败: hidden_states[{layer}] 未匹配到任何 block"
        _save(out_file, result)
        resume.mark_completed({"error": result["error"]})
        del model
        return result
    block_idx = lmap[str(layer)]["block_idx"]
    layer_mod = layers_mod[block_idx]
    result["block_idx"] = block_idx
    result["align_max_abs_diff"] = lmap[str(layer)]["max_abs_diff"]
    print(f"  [对齐] hidden_states[{layer}] <-> layers_mod[{block_idx}] "
          f"(max|diff|={lmap[str(layer)]['max_abs_diff']:.2e})")

    # ---- 任务构造 ----
    builder = TASK_BUILDERS[task]
    items, task_meta = builder(tokenizer, args.n_prompts, seed=args.seed)
    result["task_meta"] = task_meta
    if not items:
        result["error"] = f"任务构造失败: {task_meta}"
        _save(out_file, result)
        resume.mark_completed({"error": result["error"]})
        del model
        return result

    # ---- 基线行为 + 筛查 ----
    b0_list, b0 = batch_behavior(model, tokenizer, items, layer_mod, None, device,
                                 args.batch_size)
    acc0 = float(np.mean([1.0 if v > 0 else 0.0 for v in b0_list]))
    result["b0"] = b0
    result["b0_acc"] = acc0
    result["at_chance"] = bool(acc0 <= args.chance_thresh)
    print(f"  [筛查] B0={b0:.3f} acc={acc0:.3f} (随机={args.chance_thresh})")
    if result["at_chance"]:
        result["note"] = "任务处于随机水平, 跳过干预"
        _save(out_file, result)
        resume.mark_completed({"at_chance": True, "b0_acc": acc0})
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return result

    # ---- hook 恒等性检查 ----
    id_ok = _identity_check(model, tokenizer, items[:4], layer_mod, sae,
                           mean_t, std_t, device, block_idx)
    result["hook_identity_ok"] = id_ok
    print(f"  [hook] 恒等性(alpha=1)检查: {id_ok}")

    # ---- 批量一致性检查 (left-padding 是否等价于逐条前向) ----
    single_vals, _ = batch_behavior(model, tokenizer, items[:4], layer_mod, None, device, 1)
    batch_vals, _ = batch_behavior(model, tokenizer, items[:4], layer_mod, None, device, 4)
    max_dev = float(np.max(np.abs(np.array(single_vals) - np.array(batch_vals))))
    result["batching_check"] = {"single": single_vals, "batch": batch_vals,
                                "max_abs_dev": max_dev}
    print(f"  [batch] 单条 vs 批量 max|Δ|={max_dev:.2e}")

    # ---- 任务激活 + 特征选择 ----
    packs = capture_sparse(model, tokenizer, items, layer_mod, sae, mean_t, std_t, device,
                           args.batch_size)
    S = np.concatenate([p[0] for p in packs], axis=0)      # (T, hidden_dim)
    M = np.concatenate([p[1] for p in packs], axis=0)      # (T,) bool
    T = int(M.sum())
    S = S[M]
    hidden_dim = S.shape[1]
    act_freq = (S > 0).mean(axis=0)
    mean_act = S.mean(axis=0)
    max_act = S.max(axis=0)
    sel = np.argsort(mean_act)[::-1][:args.n_features]
    print(f"  [特征] 任务 token 数={T}, 候选 top-{args.n_features} 平均激活 "
          f"{mean_act[sel].min():.3f}~{mean_act[sel].max():.3f}")

    # ---- 语料侧指标 (M4 需要重构基线 / M5 上下文 / M6 频率) ----
    corpus_texts, corpus_acts = load_corpus(base_dir, config, m, layer)
    ctx_coherence = {}
    corpus_freq = None
    recon_gain = {}
    if corpus_acts is not None:
        ca = torch.tensor(corpus_acts, dtype=torch.float32, device=device)
        cn = (ca - mean_t) / std_t
        with torch.no_grad():
            cs, _ = sae.encode(cn)
            base_recon = sae.decode(cs)
            base_mse = float(((base_recon - cn) ** 2).mean().item())
        cs_np = cs.cpu().numpy()
        cn_np = cn.cpu().numpy()
        base_recon_np = base_recon.cpu().numpy()
        # decoder: nn.Linear(hidden_dim, input_dim) -> weight (input_dim, hidden_dim)
        # 特征 f 的 decoder 方向 = W[:, f]
        W = sae.decoder.weight.detach().float().cpu().numpy()  # (input_dim, hidden_dim)
        corpus_freq = (cs_np > 0).mean(axis=0)
        for f in sel.tolist():
            # M4: 移除该特征后的重构 MSE 增量
            delta = cs_np[:, f:f + 1] * W[:, f][None, :]
            new_recon = base_recon_np - delta
            recon_gain[f] = float(((new_recon - cn_np) ** 2).mean().item() - base_mse)
            # M5: top-10 激活文本的 TF-IDF 一致性
            order = np.argsort(cs_np[:, f])[::-1][:10]
            ctx = [corpus_texts[i] for i in order if cs_np[i, f] > 0]
            ctx_coherence[f] = tfidf_coherence(ctx)
        del ca, cn, cs, base_recon, cs_np, cn_np, base_recon_np, W
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ---- 单特征干预 ----
    feats = []
    for f in sel.tolist():
        rec = {
            "feature_idx": int(f),
            "M1_act_freq_task": float(act_freq[f]),
            "M2_mean_act_task": float(mean_act[f]),
            "M3_max_act_task": float(max_act[f]),
            "M4_recon_mse_gain": recon_gain.get(f),
            "M5_ctx_coherence": ctx_coherence.get(f),
            "M6_corpus_freq": (float(corpus_freq[f]) if corpus_freq is not None else None),
        }
        for mode in ("ablate", "steer"):
            alpha = args.alpha if mode == "steer" else 1.0
            hk = make_feature_hook(sae, mean_t, std_t, f, mode, alpha)
            vals, bm = batch_behavior(model, tokenizer, items, layer_mod, hk, device,
                                      args.batch_size)
            rec[f"C_{mode}"] = float(abs(bm - b0))
            rec[f"B_{mode}"] = bm
            rec[f"flip_{mode}"] = float(np.mean(
                [1.0 if (v > 0) != (v0 > 0) else 0.0 for v, v0 in zip(vals, b0_list)]))
        feats.append(rec)
    print(f"  [干预] {len(feats)} 个特征完成; "
          f"C_ablate 最大={max(r['C_ablate'] for r in feats):.3f}, "
          f"C_steer 最大={max(r['C_steer'] for r in feats):.3f}")

    # ---- 基线 ----
    rng = np.random.RandomState(args.seed)
    rand_feats = rng.choice(hidden_dim, size=min(args.n_random, hidden_dim), replace=False)
    rb = []
    for f in rand_feats.tolist():
        hk = make_feature_hook(sae, mean_t, std_t, int(f), "ablate", 1.0)
        _, bm = batch_behavior(model, tokenizer, items, layer_mod, hk, device, args.batch_size)
        rb.append(abs(bm - b0))
    rd = []
    for i in range(args.n_random_dir):
        d = torch.randn(mean_np.shape[0], device=device)
        pair = int(rand_feats[i % len(rand_feats)])
        hk = make_norm_matched_random_hook(sae, mean_t, std_t, d, pair)
        _, bm = batch_behavior(model, tokenizer, items, layer_mod, hk, device, args.batch_size)
        rd.append(abs(bm - b0))
    _, b_all = batch_behavior(model, tokenizer, items, layer_mod,
                              make_all_feature_ablate_hook(sae, mean_t, std_t), device,
                              args.batch_size)
    _, b_mean = batch_behavior(model, tokenizer, items, layer_mod,
                               make_mean_ablate_hook(mean_t), device, args.batch_size)
    result["baselines"] = {
        "random_feature_ablate": {"mean": float(np.mean(rb)), "std": float(np.std(rb)),
                                  "max": float(np.max(rb)), "n": len(rb)},
        "random_dir_ablate": {"mean": float(np.mean(rd)), "std": float(np.std(rd)),
                              "max": float(np.max(rd)), "n": len(rd)},
        "all_feature_ablate": {"B": b_all, "C": float(abs(b_all - b0))},
        "mean_ablate": {"B": b_mean, "C": float(abs(b_mean - b0))},
    }
    print(f"  [基线] 随机特征 C={result['baselines']['random_feature_ablate']['mean']:.4f} "
          f"随机方向 C={result['baselines']['random_dir_ablate']['mean']:.4f} "
          f"全特征 C={result['baselines']['all_feature_ablate']['C']:.3f} "
          f"mean-ablate C={result['baselines']['mean_ablate']['C']:.3f}")

    # ---- 相关性 ----
    metrics = ["M1_act_freq_task", "M2_mean_act_task", "M3_max_act_task",
               "M4_recon_mse_gain", "M5_ctx_coherence", "M6_corpus_freq"]
    corr = {}
    for mm in metrics:
        for cc in ("C_ablate", "C_steer"):
            xs, ys = [], []
            for r in feats:
                if r.get(mm) is None:
                    continue
                xs.append(r[mm]); ys.append(r[cc])
            corr[f"{mm}_vs_{cc}"] = {"spearman": spearman(xs, ys),
                                     "pearson": pearson(xs, ys), "n": len(xs)}
    result["correlations"] = corr
    result["features"] = feats
    _save(out_file, result)
    resume.mark_completed({"n_features": len(feats), "b0_acc": acc0})

    print("  [相关] " + "; ".join(
        f"{k}={v['spearman']:.3f}" for k, v in corr.items() if k.endswith("C_ablate")))

    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return result


def _identity_check(model, tokenizer, items, layer_mod, sae, mean_t, std_t, device,
                    block_idx):
    """alpha=1 时 hook 必须等价于恒等映射: 比较有无 hook 的 logits."""
    _, b_no = batch_behavior(model, tokenizer, items, layer_mod, None, device, len(items))
    hk = make_feature_hook(sae, mean_t, std_t, 0, "steer", 1.0)
    _, b_hk = batch_behavior(model, tokenizer, items, layer_mod, hk, device, len(items))
    return bool(abs(b_no - b_hk) < 1e-3)


def _save(path, obj):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Step2-2: SAE 因果效度主实验")
    parser.add_argument("--config", type=str, default="config.json")
    parser.add_argument("--model", type=str, default=None)
    parser.add_argument("--layers", type=str, default=None, help="逗号分隔, 默认 target_layers 中间层")
    parser.add_argument("--tasks", type=str, default="ioi,gt")
    parser.add_argument("--n_prompts", type=int, default=200)
    parser.add_argument("--n_features", type=int, default=200)
    parser.add_argument("--n_random", type=int, default=200)
    parser.add_argument("--n_random_dir", type=int, default=100)
    parser.add_argument("--alpha", type=float, default=3.0)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--chance_thresh", type=float, default=0.60)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out_tag", type=str, default="",
                        help="输出目录后缀, 如 _s43 -> causal_validity_s43 (多种子不覆盖)")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--sae_root", type=str, default=None,
                        help="替代 SAE 根目录名 (相对 output/phaseStep2), 默认 layer_sae")
    args = parser.parse_args()

    config = load_config(args.config)
    base_dir = Path(args.config).parent
    tasks = [t.strip() for t in args.tasks.split(",") if t.strip()]

    models = [args.model] if args.model else list(config["models"].keys())
    out_root = (base_dir / "output" / "phaseStep2"
                / ("causal_validity" + args.out_tag))
    out_root.mkdir(parents=True, exist_ok=True)
    log_dir = out_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_saver = LogSaver(str(log_dir / f"36_causal_validity_{ts}.log"))
    sys.stdout = log_saver

    summary_path = out_root / "summary.json"
    summary = {"timestamp": ts, "runs": {}}
    if summary_path.exists():
        try:
            with open(summary_path, "r", encoding="utf-8") as f:
                prev = json.load(f)
            if isinstance(prev.get("runs"), dict):
                summary["runs"] = prev["runs"]
        except Exception:
            pass
    try:
        for m in models:
            _, model_cfg = resolve_model_cfg(config, m)
            if args.layers:
                layers = [int(x) for x in args.layers.split(",")]
            else:
                layers = [model_cfg["target_layers"][1]]
            for layer in layers:
                for task in tasks:
                    key = f"{m}|L{layer}|{task}"
                    try:
                        res = run_one(args, config, base_dir, m, layer, task)
                        summary["runs"][key] = {
                            "error": res.get("error"),
                            "at_chance": res.get("at_chance"),
                            "b0_acc": res.get("b0_acc"),
                            "align_max_abs_diff": res.get("align_max_abs_diff"),
                            "correlations": res.get("correlations"),
                            "baselines": res.get("baselines"),
                        }
                    except Exception as e:
                        import traceback
                        traceback.print_exc()
                        print(f"[失败] {key}: {e}")
                        summary["runs"][key] = {"error": str(e)[:400]}
                    _save(summary_path, summary)
    finally:
        log_saver.close()
        sys.stdout = log_saver.terminal

    print(f"[完成] 汇总: {out_root / 'summary.json'}")


if __name__ == "__main__":
    main()
