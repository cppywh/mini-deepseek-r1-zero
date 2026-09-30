"""无卡下载模型与 GSM8K；优先复用旧缓存，不加载模型、不训练。"""

import argparse
import os
import shutil
from pathlib import Path

from project_paths import require_space, setup_paths


def reuse_model_cache(model_id: str, hub: Path) -> None:
    """输入模型 ID 与新 hub 目录，将系统盘已有模型缓存复制到数据盘；保留源文件。"""
    name = "models--" + model_id.replace("/", "--")
    old = Path.home() / ".cache" / "huggingface" / "hub" / name
    new = hub / name
    if not old.is_dir() or new.exists() or old.resolve() == new.resolve():
        return
    size = sum(p.stat().st_size for p in old.rglob("*") if p.is_file() and not p.is_symlink())
    require_space(hub, size / 2**30 + 8)
    print("复用旧缓存:", model_id, flush=True)
    shutil.copytree(old, new, symlinks=True)


def main() -> None:
    """默认准备 Base 和 smoke 模型；--comparison 额外下载两个评估模型。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", action="store_true", help="额外下载 Distill 和 Instruct；会增加磁盘使用")
    args = parser.parse_args()
    paths = setup_paths()
    from huggingface_hub import snapshot_download
    from datasets import load_dataset

    models = ["Qwen/Qwen3-0.6B-Base", "Qwen/Qwen3-1.7B-Base"]
    if args.comparison:
        models += ["deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B", "Qwen/Qwen3-1.7B"]
    for model_id in models:
        reuse_model_cache(model_id, paths["hub"])
        # 下载一个模型后，仍需给训练检查点留下至少8GiB空间。
        require_space(paths["root"], 12)
        print("准备模型:", model_id, "下载源:", os.environ["HF_ENDPOINT"], flush=True)
        snapshot_download(model_id, cache_dir=str(paths["hub"]),
                          allow_patterns=["*.json", "*.safetensors", "*.txt", "*.jinja", "*.model"])
    load_dataset("openai/gsm8k", "main", cache_dir=str(paths["datasets"]))
    require_space(paths["root"], 8)
    print("模型与数据准备完成。恢复 GPU 后再运行 smoke。")


if __name__ == "__main__":
    main()
