import time
from pathlib import Path
import os

import torch

os.environ.setdefault("TRITON_CACHE_DIR", str(Path(".triton-cache").resolve()))

import triton
import triton.language as tl


@triton.jit
def vector_add_kernel(a_ptr, b_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    a = tl.load(a_ptr + offsets, mask=mask)
    b = tl.load(b_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, a + b, mask=mask)


def triton_vector_add(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    out = torch.empty_like(a)
    n_elements = out.numel()
    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)
    vector_add_kernel[grid](a, b, out, n_elements, BLOCK_SIZE=1024)
    return out


def time_cpu(fn, warmup=5, repeats=30) -> float:
    for _ in range(warmup):
        fn()

    start = time.perf_counter()
    for _ in range(repeats):
        fn()
    end = time.perf_counter()
    return (end - start) * 1000 / repeats


def time_gpu(fn, warmup=10, repeats=100) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(repeats):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / repeats


if not torch.cuda.is_available():
    raise RuntimeError("CUDA is not available on this machine/runtime.")


n = 50_000_000
a_cpu = torch.rand(n, dtype=torch.float32)
b_cpu = torch.rand(n, dtype=torch.float32)

# Move inputs once so GPU timing measures kernel execution, not CPU-to-GPU copy time.
a_gpu = a_cpu.cuda()
b_gpu = b_cpu.cuda()

cpu_ms = time_cpu(lambda: a_cpu + b_cpu)
gpu_ms = time_gpu(lambda: triton_vector_add(a_gpu, b_gpu))

cpu_out = a_cpu + b_cpu
gpu_out = triton_vector_add(a_gpu, b_gpu)
max_diff = (cpu_out.cuda() - gpu_out).abs().max().item()

print(f"Elements: {n:,}")
print(f"CPU time: {cpu_ms:.3f} ms")
print(f"Triton GPU kernel time: {gpu_ms:.3f} ms")
print(f"Speedup: {cpu_ms / gpu_ms:.1f}x")
print(f"Max difference: {max_diff}")
