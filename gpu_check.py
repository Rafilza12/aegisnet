"""
gpu_check.py – Quick CUDA / device sanity check.

Run this FIRST after setting up the environment to confirm PyTorch
can see your GPU and tensors can be allocated on it.

Usage:
    python gpu_check.py
"""

import sys
import torch


def main():
    print("=" * 55)
    print(" AegisNet – GPU / Environment Check")
    print("=" * 55)

    # Python & PyTorch versions
    print(f"Python  : {sys.version.split()[0]}")
    print(f"PyTorch : {torch.__version__}")

    # CUDA availability
    cuda_available = torch.cuda.is_available()
    print(f"CUDA    : {'Available ✓' if cuda_available else 'Not available ✗'}")

    if not cuda_available:
        print("\n[WARNING] No GPU detected – training will run on CPU.")
        print("  → Install the CUDA-enabled build of PyTorch:")
        print("    pip install torch torchvision torchaudio "
              "--index-url https://download.pytorch.org/whl/cu118")
        return

    # GPU details
    gpu_count = torch.cuda.device_count()
    print(f"GPUs    : {gpu_count}")
    for i in range(gpu_count):
        name   = torch.cuda.get_device_name(i)
        mem_gb = torch.cuda.get_device_properties(i).total_memory / 1e9
        print(f"  [{i}] {name}  –  {mem_gb:.1f} GB VRAM")

    # Allocate a small tensor to verify memory is accessible
    device = torch.device("cuda:0")
    dummy  = torch.randn(1024, 1024, device=device)
    print(f"\nTest tensor allocated on {device}: shape={list(dummy.shape)}, "
          f"dtype={dummy.dtype}")
    del dummy
    torch.cuda.empty_cache()

    print("\n[OK] GPU is ready for AegisNet training.")
    print("=" * 55)


if __name__ == "__main__":
    main()
