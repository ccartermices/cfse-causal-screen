import os
import sys
import json
import argparse
import time
from pathlib import Path
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed


class LogSaver:
    def __init__(self, log_path):
        self.terminal = sys.stdout
        self.log = open(log_path, "w", encoding="utf-8")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        self.terminal.flush()
        self.log.flush()

    def close(self):
        self.log.close()


class ResumeManager:
    def __init__(self, output_dir, script_name):
        self.output_dir = Path(output_dir)
        self.script_name = script_name
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
        state = self.load_state()
        return state.get("status") == "completed"

    def mark_completed(self, result_info=None):
        state = self.load_state()
        state["status"] = "completed"
        state["completed_at"] = datetime.now().isoformat()
        if result_info:
            state["result"] = result_info
        self.save_state(state)

    def mark_step_done(self, step_name, step_data=None):
        state = self.load_state()
        if "steps_done" not in state:
            state["steps_done"] = {}
        state["steps_done"][step_name] = {"done": True, "timestamp": datetime.now().isoformat()}
        if step_data:
            state["steps_done"][step_name].update(step_data)
        self.save_state(state)

    def is_step_done(self, step_name):
        state = self.load_state()
        return state.get("steps_done", {}).get(step_name, {}).get("done", False)

    def reset(self):
        if self.state_file.exists():
            self.state_file.unlink()


def download_modelscope_dataset(dataset_path, local_dir, samples=None):
    from modelscope.msdatasets import MsDataset

    print(f"[下载] ModelScope 数据集: {dataset_path}")
    os.makedirs(local_dir, exist_ok=True)

    parts = dataset_path.split("/")
    namespace = parts[0] if len(parts) == 2 else None
    name = parts[1] if len(parts) == 2 else parts[0]

    ds = MsDataset.load(name, namespace=namespace, split="train")

    count = 0
    total = samples or len(ds)

    for item in ds:
        if samples and count >= samples:
            break

        save_item = {}
        for key, val in item.items():
            try:
                json.dumps(val)
                save_item[key] = val
            except (TypeError, ValueError):
                save_item[key] = str(val)

        out_path = os.path.join(local_dir, f"sample_{count:06d}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(save_item, f, ensure_ascii=False, indent=2)

        count += 1
        if count % 200 == 0:
            print(f"  已下载: {count}/{total}")

    print(f"[完成] {dataset_path}: {count} 条 -> {local_dir}")
    return count


def download_hf_dataset(dataset_path, local_dir, samples=None, subset=None):
    from datasets import load_dataset

    desc = f"{dataset_path}/{subset}" if subset else dataset_path
    print(f"[下载] HuggingFace 数据集: {desc}")
    os.makedirs(local_dir, exist_ok=True)

    if subset:
        ds = load_dataset(dataset_path, subset, split="train", streaming=True)
    else:
        ds = load_dataset(dataset_path, split="train", streaming=True)

    count = 0
    total = samples or "全部"

    for item in ds:
        if samples and count >= samples:
            break

        save_item = {}
        for key, val in item.items():
            try:
                json.dumps(val)
                save_item[key] = str(val) if not isinstance(val, (str, int, float, bool, list, dict)) else val
            except (TypeError, ValueError):
                save_item[key] = str(val)

        out_path = os.path.join(local_dir, f"sample_{count:06d}.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(save_item, f, ensure_ascii=False, indent=2)

        count += 1
        if count % 200 == 0:
            print(f"  已下载: {count}/{total}")

    print(f"[完成] {desc}: {count} 条 -> {local_dir}")
    return count


def download_file(url, dest_path, desc="", retries=3):
    import requests

    dest_path = Path(dest_path)
    if dest_path.exists():
        return True

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = dest_path.with_suffix(dest_path.suffix + ".tmp")

    for attempt in range(retries):
        try:
            resp = requests.get(url, stream=True, timeout=60)
            if resp.status_code == 200:
                total_size = int(resp.headers.get("content-length", 0))
                downloaded = 0
                with open(tmp_path, "wb") as f:
                    for chunk in resp.iter_content(chunk_size=65536):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)
                tmp_path.rename(dest_path)
                if desc:
                    size_mb = dest_path.stat().st_size / (1024 * 1024)
                    print(f"  [OK] {desc} ({size_mb:.1f}MB)")
                return True
            else:
                if attempt < retries - 1:
                    time.sleep(2 * (attempt + 1))
        except Exception as e:
            if attempt < retries - 1:
                print(f"  [重试 {attempt+2}/{retries}] {desc}: {e}")
                time.sleep(3 * (attempt + 1))
            else:
                print(f"  [失败] {desc}: {e}")

    if tmp_path.exists():
        tmp_path.unlink()
    return False


def download_coco_images(image_dir, year="2017", split="train", max_images=None):
    print(f"\n[COCO] 下载 COCO {year} {split} 图片...")
    image_dir = Path(image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)

    zip_name = f"{split}{year}.zip"
    zip_path = image_dir / zip_name

    urls = [
        f"http://images.cocodataset.org/zips/{split}{year}.zip",
        f"https://bj.bcebos.com/v1/ai-studio-online/1b5e0c1e1c1c4c1c1c1c1c1c1c1c1c1c/{split}{year}.zip",
    ]

    if not zip_path.exists():
        print(f"[COCO] 下载 {zip_name} (~18GB, 请耐心等待)...")
        downloaded = False
        for url in urls:
            print(f"  尝试: {url}")
            if download_file(url, str(zip_path), desc=zip_name, retries=2):
                downloaded = True
                break
        if not downloaded:
            print(f"[COCO] 自动下载失败，请手动下载:")
            print(f"  wget http://images.cocodataset.org/zips/{split}{year}.zip")
            print(f"  放到: {image_dir}")
            return 0

    extract_dir = image_dir / f"{split}{year}"
    if not extract_dir.exists() or not any(extract_dir.iterdir()):
        print(f"[COCO] 解压 {zip_name}...")
        import zipfile
        with zipfile.ZipFile(str(zip_path), "r") as zf:
            zf.extractall(str(image_dir))
        print(f"[COCO] 解压完成")

    images = list(extract_dir.glob("*.jpg"))
    if max_images and len(images) > max_images:
        images = images[:max_images]

    print(f"[COCO] 可用图片: {len(images)} 张")
    return len(images)


def download_coco_annotations(data_dir, year="2017"):
    print(f"\n[COCO] 下载标注文件...")
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    ann_file = data_dir / f"annotations/captions_train{year}.json"
    if ann_file.exists():
        print(f"[COCO] 标注已存在: {ann_file}")
        return str(ann_file)

    ann_zip = data_dir / "annotations_trainval2017.zip"
    ann_url = f"http://images.cocodataset.org/annotations/annotations_trainval{year}.zip"

    if not ann_zip.exists():
        print(f"  下载标注文件...")
        if not download_file(ann_url, str(ann_zip), desc="COCO annotations"):
            print(f"[COCO] 标注下载失败，将使用 ModelScope 数据集")
            return None

    import zipfile
    with zipfile.ZipFile(str(ann_zip), "r") as zf:
        zf.extractall(str(data_dir))

    if ann_file.exists():
        print(f"[COCO] 标注就绪: {ann_file}")
        return str(ann_file)
    return None


def download_sharegpt4v_images(meta_dir, image_dir, samples=None, num_workers=4):
    print(f"\n[ShareGPT4V] 下载图片...")
    meta_dir = Path(meta_dir)
    image_dir = Path(image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)

    json_files = sorted(meta_dir.glob("sample_*.json"))
    if not json_files:
        print(f"[ShareGPT4V] 未找到元数据文件，先下载元数据...")
        return 0

    print(f"[ShareGPT4V] 元数据文件: {len(json_files)} 个")

    print(f"[ShareGPT4V] 分析元数据结构...")
    sample_keys = set()
    sample_item = {}
    for jf in json_files[:5]:
        try:
            with open(jf, "r", encoding="utf-8") as f:
                item = json.load(f)
            sample_keys.update(item.keys())
            if not sample_item:
                sample_item = {k: type(v).__name__ for k, v in item.items()}
        except Exception:
            continue
    print(f"  字段: {sorted(sample_keys)}")
    print(f"  类型: {sample_item}")

    if samples:
        json_files = json_files[:samples]

    tasks = []
    for jf in json_files:
        try:
            with open(jf, "r", encoding="utf-8") as f:
                item = json.load(f)
        except Exception:
            continue

        img_url = None
        for key in ("image", "img", "image_url", "url", "image_path", "path", "conversations"):
            val = item.get(key, "")
            if isinstance(val, str) and val.startswith(("http://", "https://")):
                img_url = val
                break
            if isinstance(val, list) and len(val) > 0:
                first = val[0]
                if isinstance(first, dict):
                    for v in first.values():
                        if isinstance(v, str) and v.startswith(("http://", "https://")):
                            img_url = v
                            break
                if img_url:
                    break

        if not img_url:
            for val in item.values():
                if isinstance(val, str) and val.startswith(("http://", "https://")) and any(
                    ext in val.lower() for ext in (".jpg", ".jpeg", ".png", ".webp")
                ):
                    img_url = val
                    break

        if img_url:
            idx = jf.stem.replace("sample_", "")
            ext = ".jpg"
            for e in (".png", ".webp", ".jpeg"):
                if e in img_url.lower():
                    ext = e
                    break
            dest = image_dir / f"img_{idx}{ext}"
            tasks.append((img_url, str(dest), f"img_{idx}"))

    if not tasks:
        print(f"[ShareGPT4V] 未找到图片URL，尝试从 HuggingFace 数据集下载...")
        return download_sharegpt4v_from_hf(image_dir, samples)

    print(f"[ShareGPT4V] 待下载: {len(tasks)} 张图片")
    success = 0

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = {}
        for url, dest, desc in tasks:
            if Path(dest).exists():
                success += 1
                continue
            future = executor.submit(download_file, url, dest, desc, 2)
            futures[future] = desc

        for i, future in enumerate(as_completed(futures)):
            desc = futures[future]
            try:
                if future.result():
                    success += 1
            except Exception:
                pass
            if (i + 1) % 100 == 0:
                print(f"  进度: {i+1}/{len(tasks)} (成功: {success})")

    print(f"[ShareGPT4V] 图片下载完成: {success}/{len(tasks)}")
    return success


def download_sharegpt4v_from_hf(image_dir, meta_dir=None, samples=None):
    try:
        from datasets import load_dataset
        print(f"  从 HuggingFace 加载 ShareGPT4V...")
        ds = load_dataset("Lin-Chen/ShareGPT4V", "ShareGPT4V", split="train", streaming=True)
    except Exception as e:
        print(f"  [失败] 无法加载 HF 数据集: {e}")
        print(f"  [提示] ShareGPT4V 图片需手动下载或使用其他数据集替代")
        return 0

    image_dir = Path(image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)
    if meta_dir:
        meta_dir = Path(meta_dir)
        meta_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    img_count = 0
    debug_types = set()

    for item in ds:
        if samples and count >= samples:
            break

        image_data = item.get("image")
        if image_data is not None:
            if count < 3:
                debug_types.add(type(image_data).__name__)
                print(f"  [调试] sample {count}: image type={type(image_data).__name__}, "
                      f"has save={hasattr(image_data, 'save')}, "
                      f"is bytes={isinstance(image_data, bytes)}, "
                      f"is dict={isinstance(image_data, dict)}, "
                      f"is str={isinstance(image_data, str)}")
                if isinstance(image_data, str):
                    print(f"    value preview: {str(image_data)[:200]}")

            img_path = image_dir / f"img_{count:06d}.jpg"
            if not img_path.exists():
                try:
                    saved = False
                    if hasattr(image_data, "save"):
                        image_data.save(str(img_path), "JPEG")
                        saved = True
                    elif isinstance(image_data, bytes):
                        with open(img_path, "wb") as f:
                            f.write(image_data)
                        saved = True
                    elif isinstance(image_data, str):
                        if image_data.startswith(("http://", "https://")):
                            saved = download_file(image_data, str(img_path), f"img_{count:06d}", 2)
                        elif os.path.exists(image_data):
                            import shutil
                            shutil.copy2(image_data, str(img_path))
                            saved = True
                        else:
                            filename = os.path.basename(image_data)
                            coco_img = image_dir.parent / "coco_caption" / "train2017" / filename
                            if coco_img.exists():
                                import shutil
                                shutil.copy2(str(coco_img), str(img_path))
                                saved = True
                            else:
                                coco_url = f"http://images.cocodataset.org/train2017/{filename}"
                                saved = download_file(coco_url, str(img_path), f"img_{count:06d}", 2)
                    elif isinstance(image_data, dict):
                        if "bytes" in image_data:
                            with open(img_path, "wb") as f:
                                f.write(image_data["bytes"])
                            saved = True
                        elif "path" in image_data and os.path.exists(image_data["path"]):
                            import shutil
                            shutil.copy2(image_data["path"], str(img_path))
                            saved = True
                    elif hasattr(image_data, "tobytes"):
                        raw = image_data.tobytes()
                        if raw:
                            with open(img_path, "wb") as f:
                                f.write(raw)
                            saved = True

                    if not saved:
                        try:
                            from PIL import Image
                            if isinstance(image_data, Image.Image):
                                image_data.save(str(img_path), "JPEG")
                                saved = True
                        except Exception:
                            pass

                    if saved:
                        img_count += 1
                except Exception as e:
                    if count < 5:
                        print(f"  [保存失败] sample {count}: {e}")

        if meta_dir:
            save_item = {}
            for key, val in item.items():
                if key == "image":
                    continue
                try:
                    json.dumps(val)
                    save_item[key] = val
                except (TypeError, ValueError):
                    save_item[key] = str(val)
            if img_path.exists():
                save_item["local_image_path"] = str(img_path)
            out_path = meta_dir / f"sample_{count:06d}.json"
            with open(out_path, "w", encoding="utf-8") as f:
                json.dump(save_item, f, ensure_ascii=False, indent=2)

        count += 1
        if count % 200 == 0:
            print(f"  已处理: {count} (图片: {img_count})")

    print(f"[ShareGPT4V-HF] 图片: {img_count}/{count}")
    if debug_types:
        print(f"[ShareGPT4V-HF] 遇到的image类型: {debug_types}")
    return img_count


def download_coco_with_annotations(data_dir, image_dir, samples=None):
    print(f"\n[COCO] 下载 COCO 数据集 (标注+图片)...")
    data_dir = Path(data_dir)
    image_dir = Path(image_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    image_dir.mkdir(parents=True, exist_ok=True)

    ds = None
    for ds_name in ["HuggingFaceM4/COCO", "ydshieh/coco_dataset"]:
        try:
            from datasets import load_dataset
            print(f"  尝试 HuggingFace: {ds_name}...")
            ds = load_dataset(ds_name, split="train", streaming=True)
            print(f"  [OK] 加载成功: {ds_name}")
            break
        except Exception as e:
            print(f"  [失败] {ds_name}: {e}")

    if ds is None:
        try:
            from modelscope.msdatasets import MsDataset
            print(f"  尝试 ModelScope: modelscope/coco_2017...")
            ds = MsDataset.load("coco_2017", namespace="modelscope", split="train")
            print(f"  [OK] ModelScope 加载成功")
        except Exception as e:
            print(f"  [失败] ModelScope: {e}")

    if ds is None:
        print(f"[COCO] 所有数据源均失败，跳过 COCO 图片下载")
        print(f"[提示] 可手动下载: wget http://images.cocodataset.org/zips/train2017.zip")
        return 0

    count = 0
    img_count = 0
    tasks = []
    debug_printed = False

    print(f"  读取标注并收集图片URL...")
    for item in ds:
        if samples and count >= samples:
            break

        if not debug_printed:
            print(f"  [调试] COCO item keys: {list(item.keys())[:15]}")
            debug_printed = True

        save_item = {}
        img_url = None

        for key, val in item.items():
            if key in ("url", "image_url", "image", "coco_url", "flickr_url"):
                if isinstance(val, str) and val.startswith(("http://", "https://")):
                    img_url = val
                elif hasattr(val, "save"):
                    img_path = image_dir / f"coco_{count:08d}.jpg"
                    if not img_path.exists():
                        try:
                            val.save(str(img_path), "JPEG")
                            save_item["local_image_path"] = str(img_path)
                            img_count += 1
                        except Exception:
                            pass
                    else:
                        save_item["local_image_path"] = str(img_path)
                        img_count += 1
                continue

            try:
                json.dumps(val)
                save_item[key] = val
            except (TypeError, ValueError):
                save_item[key] = str(val)

        if img_url:
            img_path = image_dir / f"coco_{count:08d}.jpg"
            if img_path.exists():
                save_item["local_image_path"] = str(img_path)
                img_count += 1
            else:
                tasks.append((img_url, str(img_path), f"coco_{count:08d}"))

        out_path = data_dir / f"sample_{count:08d}.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(save_item, f, ensure_ascii=False, indent=2)

        count += 1
        if count % 1000 == 0:
            print(f"  已读取: {count} (待下载图片: {len(tasks)})")

    print(f"[COCO] 标注: {count} 条, 待下载图片: {len(tasks)} 张")

    if tasks:
        print(f"[COCO] 开始下载图片 (预计每张~200KB, 总计~{len(tasks)*0.2:.0f}MB)...")
        success = 0
        with ThreadPoolExecutor(max_workers=8) as executor:
            futures = {}
            for url, dest, desc in tasks:
                future = executor.submit(download_file, url, dest, desc, 2)
                futures[future] = desc

            for i, future in enumerate(as_completed(futures)):
                try:
                    if future.result():
                        success += 1
                except Exception:
                    pass
                if (i + 1) % 500 == 0:
                    print(f"  进度: {i+1}/{len(tasks)} (成功: {success})")

        img_count += success
        print(f"[COCO] 图片下载: {success}/{len(tasks)}")

    print(f"[COCO] 完成: {count} 条标注, {img_count} 张图片")
    return count


def main():
    parser = argparse.ArgumentParser(description="阶段1: 数据准备 - 下载数据集(含图片)")
    parser.add_argument("--config", type=str, default="config.json")
    parser.add_argument("--samples", type=int, default=None, help="覆盖每类数据集的样本数")
    parser.add_argument("--skip_multimodal", action="store_true", help="跳过多模态数据集")
    parser.add_argument("--skip_text", action="store_true", help="跳过文本数据集")
    parser.add_argument("--skip_images", action="store_true", help="跳过图片下载(仅保存元数据)")
    parser.add_argument("--only", type=str, default=None, help="只下载指定数据集(逗号分隔), 如 sharegpt4v,coco_caption")
    parser.add_argument("--force", action="store_true", help="强制重新运行，忽略断点续运行状态")
    parser.add_argument("--coco_year", type=str, default="2017", help="COCO年份: 2014/2017")
    parser.add_argument("--num_workers", type=int, default=4, help="图片下载并发数")
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        config = json.load(f)

    base_dir = Path(args.config).parent
    data_dir = base_dir / config["paths"]["data_dir"]
    data_dir.mkdir(parents=True, exist_ok=True)

    resume = ResumeManager(data_dir, "01_download")
    if args.force:
        resume.reset()
        print("[强制] 重置断点续运行状态")

    log_dir = data_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_saver = LogSaver(str(log_dir / f"01_download_{timestamp}.log"))
    sys.stdout = log_saver

    if resume.is_completed():
        print("[跳过] 数据下载已完成")
        print("  如需重新运行，请使用 --force 参数")
        log_saver.close()
        sys.stdout = log_saver.terminal
        return

    image_dir = base_dir / config["paths"].get("image_dir", "./data/images")
    phase1 = config["phase1_data"]
    total = 0

    only_datasets = None
    if args.only:
        only_datasets = set(d.strip() for d in args.only.split(","))
        print(f"[过滤] 只下载: {only_datasets}")

    if not args.skip_text:
        print(f"\n{'='*60}")
        print(f"下载文本数据集")
        print(f"{'='*60}")
        for name, cfg in phase1.get("text_datasets", {}).items():
            if only_datasets and name not in only_datasets:
                continue
            local_dir = str(data_dir / "text" / name)
            samples = args.samples or cfg.get("samples")
            if cfg.get("type") == "modelscope":
                total += download_modelscope_dataset(cfg["path"], local_dir, samples)
            else:
                total += download_hf_dataset(cfg["path"], local_dir, samples, subset=cfg.get("subset"))

    if not args.skip_multimodal:
        print(f"\n{'='*60}")
        print(f"下载多模态数据集 (含图片)")
        print(f"{'='*60}")

        for name, cfg in phase1.get("multimodal_datasets", {}).items():
            if only_datasets and name not in only_datasets:
                continue
            local_dir = str(data_dir / "multimodal" / name)
            img_dir = str(image_dir / name)
            samples = args.samples or cfg.get("samples")
            ds_type = cfg.get("type", "modelscope")

            if args.skip_images:
                if ds_type == "modelscope":
                    total += download_modelscope_dataset(cfg["path"], local_dir, samples)
                else:
                    total += download_hf_dataset(cfg["path"], local_dir, samples, subset=cfg.get("subset"))
                continue

            if name == "coco_caption":
                total += download_coco_with_annotations(local_dir, img_dir, samples)
            elif name == "sharegpt4v":
                img_downloaded = download_sharegpt4v_from_hf(img_dir, local_dir, samples)
                total += img_downloaded
            else:
                if ds_type == "modelscope":
                    total += download_modelscope_dataset(cfg["path"], local_dir, samples)
                else:
                    total += download_hf_dataset(cfg["path"], local_dir, samples, subset=cfg.get("subset"))

    print(f"\n{'='*60}")
    print(f"阶段1 数据下载完成! 总计: {total} 条")
    print(f"数据目录: {data_dir}")
    print(f"图片目录: {image_dir}")
    print(f"{'='*60}")

    resume.mark_completed({"total": total})

    log_saver.close()
    sys.stdout = log_saver.terminal


if __name__ == "__main__":
    main()
