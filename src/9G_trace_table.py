# -*- coding: utf-8 -*-
"""9G — 生成数字溯源表：每个进 PDF 的数字 -> 哪个 JSON 的哪个键

读取 output/phaseStep2/gen_registry.json（9D 写出的宏登记表），
按宏名前缀映射到来源文件，输出 Markdown 表。
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent
BASE = ROOT / "output" / "phaseStep2"
OUT = ROOT / "_paper_build" / "reviews_20261103" / "10_数字溯源表.md"

SRC = {
    "EoneCombos": ("cfse_e1_batchnoise/e1_summary.json", "len(results)"),
    "EoneFeatRho": ("cfse_e1_batchnoise/e1_aggregate.json",
                    "aggregate.median_feat_spearman_same_dtype"),
    "EoneCellR": ("cfse_e1_batchnoise/e1_aggregate.json",
                  "aggregate.median_cell_pearson_same_dtype"),
    "EoneNoise": ("cfse_e1_batchnoise/e1_aggregate.json",
                  "aggregate.median_cell_ratio_same_dtype"),
    "EoneBatchOne": ("cfse_e1_batchnoise/e1_aggregate.json",
                     "aggregate.median_feat_spearman_batch1"),
    "EoneCrossDtype": ("cfse_e1_batchnoise/e1_aggregate.json",
                       "aggregate.median_feat_spearman_cross_dtype"),
    "EoneMedEff": ("cfse_e1_batchnoise/e1_summary.json",
                   "results[].median_abs_effect_f32_b16"),
    "EoneWorst": ("cfse_e1_batchnoise/e1_aggregate.json",
                  "results[].same_dtype_batch_ge8.median_feat_spearman"),
    "EoneFp": ("cfse_e1_fp32/<model>_l<layer>_<task>.json",
               "min_cell_pearson / pairs[fp32_*_vs_fp32_*].median_abs_cell_diff"),
    "EoneBf": ("cfse_e1_fp32/<model>_l<layer>_<task>.json", "fp32_vs_bf16_b16"),
    "Efour": ("cfse_e4_dose/e4_summary.json", "results[].curve[].rho_all"),
    "Ethree": ("cfse_e3_saefree/e3_summary.json", "results[].rho / delta_sae_minus_coord"),
    "Efive": ("cfse_e5_cekl/e5_summary.json", "results[].pred_rho / effect_corr / rel"),
}


def main():
    reg = json.load(open(BASE / "gen_registry.json", encoding="utf-8"))
    rows = []
    for k in sorted(reg):
        src = ("", "")
        for pre, s in SRC.items():
            if k.startswith(pre):
                src = s
                break
        val = reg[k]
        rows.append(f"| `\\{k}` | {val} | `{src[0]}` | `{src[1]}` |")
    head = ("# 数字溯源表\n\n"
            "本表由 `9G_trace_table.py` 从 `output/phaseStep2/gen_registry.json` 生成。"
            "PDF 里出现的每一个新增数字都是一个 LaTeX 宏（`sections/gen_numbers.tex`），"
            "宏值由 `9D_emit_macros.py` 从对应 JSON 计算，正文不含手写数字。\n\n"
            "| 宏 | 值 | 来源文件 | 键路径 |\n|---|---|---|---|\n")
    txt = head + "\n".join(rows) + "\n"
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(txt, encoding="utf-8")
    print(f"{len(rows)} 条 -> {OUT}")


if __name__ == "__main__":
    main()
