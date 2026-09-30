"""统一本地与易贝云的数据路径；在导入 Hugging Face 库之前调用。"""

import ctypes
import os
import sysconfig
import shutil
import sys
import tempfile
from pathlib import Path


def setup_paths() -> dict[str, Path]:
    """读取可选 GRPO_PROJECT_ROOT，返回项目、输出及缓存目录，并设置缓存环境变量。"""
    preload_cuda_runtime()
    override = os.environ.get("GRPO_PROJECT_ROOT")
    if override:
        root = Path(override).expanduser().resolve()
    elif os.name == "nt":
        root = Path(__file__).resolve().parent / "grpo_project"
    else:
        volume = Path("/root/blockdata")
        if not volume.is_mount():
            raise RuntimeError("/root/blockdata 尚未挂载数据盘；先挂载，避免写入30GB系统盘")
        root = volume / "grpo_project"

    paths = {
        "root": root,
        "outputs": root / "outputs",
        "hf_home": root / "cache" / "huggingface",
        "hub": root / "cache" / "huggingface" / "hub",
        "datasets": root / "cache" / "huggingface" / "datasets",
        "tmp": root / "tmp",
    }
    if "huggingface_hub" in sys.modules and os.environ.get("HF_HOME") != str(paths["hf_home"]):
        raise RuntimeError("当前内核已使用旧缓存路径，请重启内核后从第一个单元运行")
    env = {
        "HF_HOME": paths["hf_home"],
        "HF_HUB_CACHE": paths["hub"],
        "HF_DATASETS_CACHE": paths["datasets"],
        "XDG_CACHE_HOME": root / "cache",
        "TORCH_HOME": root / "cache" / "torch",
        "TORCH_EXTENSIONS_DIR": root / "cache" / "torch_extensions",
        "TRITON_CACHE_DIR": root / "cache" / "triton",
        "VLLM_CACHE_ROOT": root / "cache" / "vllm",
        "TMPDIR": paths["tmp"],
    }
    for directory in [*paths.values(), *env.values()]:
        directory.mkdir(parents=True, exist_ok=True)
    os.environ.update({key: str(value) for key, value in env.items()})
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    tempfile.tempdir = None
    print("项目目录:", root)
    print("可用空间: %.1f GiB" % (shutil.disk_usage(root).free / 2**30))
    return paths


def require_space(root: Path, needed_gib: float) -> None:
    """输入输出目录及最低剩余 GiB；空间不足时停止，避免保存到一半报错。"""
    free = shutil.disk_usage(root).free / 2**30
    if free < needed_gib:
        raise RuntimeError(f"数据盘剩余 {free:.1f} GiB，至少需要 {needed_gib:.1f} GiB；请扩容或先归档旧实验")


def preload_cuda_runtime() -> None:
    """Linux 下预加载已安装的 CUDA 13 runtime；输入无，输出无，兼容已有 Jupyter 内核启动方式。"""
    if sys.platform != "linux":
        return
    library = Path(sysconfig.get_paths()["purelib"]) / "nvidia/cu13/lib/libcudart.so.13"
    if library.is_file():
        ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)
