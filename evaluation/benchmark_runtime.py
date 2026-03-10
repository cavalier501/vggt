"""
usage:
  python evaluation/benchmark_runtime.py --mode aggregator --frames 1 2 4 8 10 20 50 100 200 --model_path /path/to/model.pt
  python evaluation/benchmark_runtime.py --mode full --frames 1 2 4 8 10 20 50 100 200 --model_path /path/to/model.pt
"""

import argparse
import json
import os
import random
import sys
import time
from datetime import datetime
from statistics import mean, pstdev

import numpy as np
import torch


import torch_npu  # noqa: F401
from torch_npu.contrib import transfer_to_npu  # noqa: F401



REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from demo_model_loader import load_vggt_weights
from vggt.models.vggt import VGGT


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark VGGT runtime and memory usage")
    parser.add_argument("--mode", choices=["aggregator", "full"], required=True, help="Benchmark mode")
    parser.add_argument(
        "--frames",
        nargs="+",
        type=int,
        default=[1, 2, 4, 8, 10, 20, 50, 100, 200],
        help="Frame counts to benchmark",
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size for synthetic inputs")
    parser.add_argument("--height", type=int, default=518, help="Input image height")
    parser.add_argument("--width", type=int, default=518, help="Input image width")
    parser.add_argument("--warmup", type=int, default=3, help="Warmup iterations per frame count")
    parser.add_argument("--repeat", type=int, default=10, help="Measured iterations per frame count")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--output_json",
        type=str,
        default="evaluation/benchmark_runtime_results.json",
        help="Path to save benchmark results as JSON",
    )
    parser.add_argument("--model_path", type=str, default=None, help="Optional local path to model.pt")
    return parser.parse_args()


def set_random_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_runtime_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


def get_inference_dtype():
    return torch.bfloat16


def synchronize_device():
    try:
        if hasattr(torch, "npu") and callable(getattr(torch.npu, "is_available", None)) and torch.npu.is_available():
            torch.npu.synchronize()
            return
    except Exception:
        pass

    if torch.cuda.is_available():
        torch.cuda.synchronize()



def empty_cache():
    try:
        if hasattr(torch, "npu") and callable(getattr(torch.npu, "is_available", None)) and torch.npu.is_available():
            torch.npu.empty_cache()
            return
    except Exception:
        pass

    if torch.cuda.is_available():
        torch.cuda.empty_cache()



def reset_peak_memory_stats():
    try:
        if hasattr(torch, "npu") and callable(getattr(torch.npu, "is_available", None)) and torch.npu.is_available():
            torch.npu.reset_peak_memory_stats()
            return True
    except Exception:
        return False




def get_peak_memory_reserved():
    try:
        if hasattr(torch, "npu") and callable(getattr(torch.npu, "is_available", None)) and torch.npu.is_available():
            return float(torch.npu.max_memory_reserved())
    except Exception:
        return None




def bytes_to_gb(num_bytes):
    if num_bytes is None:
        return None
    return num_bytes / (1024 ** 3)


def run_forward(model, images, mode, dtype):
    # Input conventions:
    # - Aggregator.forward expects [B, S, 3, H, W]. See vggt/models/aggregator.py.
    # - VGGT.forward accepts [S, 3, H, W] or [B, S, 3, H, W]. See vggt/models/vggt.py.
    # This benchmark always constructs batched inputs as [B, S, 3, H, W], default B=1.
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            if mode == "aggregator":
                return model.aggregator(images)
            return model(images)



def load_model(device, model_path):
    print("Initializing and loading VGGT model...")
    model = VGGT()
    model = load_vggt_weights(model, model_path)
    model.eval()
    model = model.to(device)
    return model



def build_input_tensor(batch_size, frames, height, width, device):
    return torch.rand(batch_size, frames, 3, height, width, device=device, dtype=torch.float32)



def benchmark_one_setting(model, batch_size, frames, height, width, mode, warmup, repeat, device, dtype):
    images = build_input_tensor(batch_size, frames, height, width, device)

    print(f"Running warmup for mode={mode}, frames={frames}")
    for _ in range(warmup):
        run_forward(model, images, mode, dtype)
        synchronize_device()

    empty_cache()
    memory_supported = reset_peak_memory_stats()

    timings = []
    peak_memories = []

    for _ in range(repeat):
        if memory_supported:
            reset_peak_memory_stats()

        synchronize_device()
        start = time.perf_counter()
        run_forward(model, images, mode, dtype)
        synchronize_device()
        end = time.perf_counter()

        timings.append(end - start)
        if memory_supported:
            peak_memories.append(get_peak_memory_reserved())

    valid_peak_memories = [m for m in peak_memories if m is not None]
    peak_memory_bytes = max(valid_peak_memories, default=None)
    peak_memory_mean_bytes = mean(valid_peak_memories) if valid_peak_memories else None
    peak_memory_std_bytes = pstdev(valid_peak_memories) if len(valid_peak_memories) > 1 else 0.0 if valid_peak_memories else None

    result = {
        "mode": mode,
        "frames": frames,
        "batch_size": batch_size,
        "time_mean_s": mean(timings),
        "peak_mem_gb": bytes_to_gb(peak_memory_bytes),
        "peak_mem_mean_gb": bytes_to_gb(peak_memory_mean_bytes),
        "peak_mem_std_gb": bytes_to_gb(peak_memory_std_bytes),
        "time_std_s": pstdev(timings) if len(timings) > 1 else 0.0,
        "time_min_s": min(timings),
        "repeat": repeat,
        "warmup": warmup,
        "dtype": str(dtype).replace("torch.", ""),
        "device": device,
        "height": height,
        "width": width,
        "memory_api": "peak_reserved",
        "memory_supported": memory_supported and peak_memory_bytes is not None,
    }
    return result



def print_results_table(results):
    headers = [
        "mode",
        "frames",
        "time_mean_s",
        "peak_mem_gb",
        "peak_mem_mean_gb",
        "peak_mem_std_gb",
        "time_std_s",
        "time_min_s",
        "repeat",
        "warmup",
        "dtype",
        "device",
    ]

    rows = []
    for item in results:
        rows.append(
            [
                item["mode"],
                str(item["frames"]),
                f'{item["time_mean_s"]:.4f}',
                "null" if item["peak_mem_gb"] is None else f'{item["peak_mem_gb"]:.4f}',
                "null" if item["peak_mem_mean_gb"] is None else f'{item["peak_mem_mean_gb"]:.4f}',
                "null" if item["peak_mem_std_gb"] is None else f'{item["peak_mem_std_gb"]:.4f}',
                f'{item["time_std_s"]:.4f}',
                f'{item["time_min_s"]:.4f}',
                str(item["repeat"]),
                str(item["warmup"]),
                item["dtype"],
                item["device"],
            ]
        )

    widths = []
    for idx, header in enumerate(headers):
        widths.append(max(len(header), *(len(row[idx]) for row in rows)))

    header_line = "  ".join(header.ljust(widths[idx]) for idx, header in enumerate(headers))
    sep_line = "  ".join("-" * widths[idx] for idx in range(len(headers)))

    print(header_line)
    print(sep_line)
    for row in rows:
        print("  ".join(row[idx].ljust(widths[idx]) for idx in range(len(headers))))



def save_results_json(args, results, device, dtype):
    output_path = args.output_json
    if not os.path.isabs(output_path):
        output_path = os.path.join(REPO_ROOT, output_path)

    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    payload = {
        "timestamp": datetime.now().isoformat(),
        "mode": args.mode,
        "frames": args.frames,
        "batch_size": args.batch_size,
        "height": args.height,
        "width": args.width,
        "warmup": args.warmup,
        "repeat": args.repeat,
        "seed": args.seed,
        "device": device,
        "dtype": str(dtype).replace("torch.", ""),
        "model_path": args.model_path,
        "results": results,
    }

    with open(output_path, "w", encoding="utf-8") as fout:
        json.dump(payload, fout, indent=2)

    print(f"Saved benchmark results to: {output_path}")



def main():
    args = parse_args()
    set_random_seeds(args.seed)

    device = get_runtime_device()
    dtype = get_inference_dtype()

    print(f"Using device: {device}")
    print(f"Using dtype: {dtype}")
    print(f"Benchmark mode: {args.mode}")

    model = load_model(device, args.model_path)

    results = []
    for frames in args.frames:
        print("=" * 80)
        print(
            f"Benchmarking mode={args.mode}, frames={frames}, "
            f"input_shape=({args.batch_size}, {frames}, 3, {args.height}, {args.width})"
        )
        result = benchmark_one_setting(
            model=model,
            batch_size=args.batch_size,
            frames=frames,
            height=args.height,
            width=args.width,
            mode=args.mode,
            warmup=args.warmup,
            repeat=args.repeat,
            device=device,
            dtype=dtype,
        )
        results.append(result)

    print("=" * 80)
    print_results_table(results)
    save_results_json(args, results, device, dtype)

    if any(not item["memory_supported"] for item in results):
        print("[WARN] Peak memory API was unavailable for part or all of this run; peak_mem_gb may be null.")


if __name__ == "__main__":
    main()
