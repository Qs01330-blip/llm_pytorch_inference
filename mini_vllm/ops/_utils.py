import os
import sys
import subprocess

_ext_loaded = None


def _add_dll_dirs():
    """Add DLL directories for Windows to find CUDA and Torch libraries."""
    if sys.platform != "win32":
        return
    # PyTorch lib directory
    try:
        import torch
        torch_lib = os.path.join(os.path.dirname(torch.__file__), "lib")
        if os.path.isdir(torch_lib):
            os.add_dll_directory(torch_lib)
    except Exception:
        pass
    # Find CUDA from nvcc in PATH
    try:
        nvcc_path = subprocess.check_output(
            ["where", "nvcc"], stderr=subprocess.DEVNULL
        ).decode().strip().split("\r\n")[0]
        cuda_bin = os.path.dirname(nvcc_path)
        if os.path.isdir(cuda_bin):
            os.add_dll_directory(cuda_bin)
    except Exception:
        pass


# Set DLL dirs at import time so CUDA extension can be loaded
_add_dll_dirs()


def _load_ext():
    """Load the CUDA extension via torch.ops.load_library."""
    _add_dll_dirs()
    import torch, glob

    # Search locations: package dir, cwd, and project root (parent of mini_vllm/)
    pkg_dir = os.path.dirname(os.path.abspath(__file__))
    search_dirs = [
        os.path.dirname(pkg_dir),          # mini_vllm/
        os.getcwd(),                        # wherever the user runs from
    ]
    # Also try project root (parent of mini_vllm if it's a subfolder)
    project_root = os.path.dirname(os.path.dirname(pkg_dir))
    if project_root not in search_dirs:
        search_dirs.append(project_root)

    for d in search_dirs:
        pyd_files = glob.glob(os.path.join(d, "_C*.pyd"))
        if pyd_files:
            try:
                torch.ops.load_library(pyd_files[0])
                return True
            except Exception:
                pass

    # Fallback: try import (works on Linux with PYBIND11_MODULE)
    try:
        import mini_vllm._C  # noqa: F401
        return True
    except (ImportError, Exception):
        pass
    return False


def is_ext_loaded() -> bool:
    """Check if the CUDA C++ extension is loaded (independent of device)."""
    global _ext_loaded
    if _ext_loaded is None:
        if os.environ.get("MINI_VLLM_BACKEND", "").lower() == "pytorch":
            _ext_loaded = False
        else:
            _ext_loaded = _load_ext()
    return _ext_loaded
