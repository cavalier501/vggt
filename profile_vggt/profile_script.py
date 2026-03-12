"""
usage:
  python profile/profile.py --image_folder path/to/images --model_path /path/to/model.pt
  python profile/profile.py --image_folder path/to/images --model_path /path/to/model.pt --mode full --enable_postprocess
  python profile/profile.py --image_folder path/to/images --model_path /path/to/model.pt --mode aggregator
"""

import argparse
import glob
import os
import sys
from datetime import datetime

import numpy as np
import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu  # noqa: F401

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)

from demo_model_loader import load_vggt_weights
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri


def parse_args():
    parser = argparse.ArgumentParser(description="Profile VGGT on NPU with real images")
    parser.add_argument("--image_folder", type=str, required=True, help="Path to folder containing images")
    parser.add_argument("--model_path", type=str, default=None, help="Optional local path to model.pt")
    parser.add_argument("--mode", choices=["aggregator", "full"], default="aggregator", help="Profile aggregator or full model forward")
    parser.add_argument("--enable_postprocess", action="store_true", help="Run demo_viser-style postprocess and visualization after profiling")
    parser.add_argument("--profile_dir", type=str, default="profile_vggt/profiling_files", help="Directory to store profiler traces")
    parser.add_argument("--warmup_steps", type=int, default=2, help="Profiler warmup steps")
    parser.add_argument("--active_steps", type=int, default=3, help="Profiler active steps")
    parser.add_argument("--schedule_repeat", type=int, default=1, help="Profiler schedule repeat count")
    parser.add_argument("--record_shapes", action=argparse.BooleanOptionalAction, default=True, help="Record operator shapes")
    parser.add_argument("--profile_memory", action=argparse.BooleanOptionalAction, default=True, help="Record operator memory")
    parser.add_argument("--with_stack", action="store_true", default=False, help="Record operator stacks")
    parser.add_argument("--with_flops", action="store_true", default=False, help="Record FLOPs metadata")
    parser.add_argument("--profiler_level", choices=["Level0", "Level1", "Level2"], default="Level1", help="Profiler detail level")
    parser.add_argument("--aic_metrics", default="PipeUtilization", help="torch_npu.profiler.AiCMetrics enum name")
    parser.add_argument("--use_point_map", action="store_true", help="Use point map instead of depth-based points in viser")
    parser.add_argument("--background_mode", action="store_true", help="Run the viser server in background mode")
    parser.add_argument("--port", type=int, default=8080, help="Port number for the viser server")
    parser.add_argument("--conf_threshold", type=float, default=25.0, help="Initial percentage of low-confidence points to filter out")
    parser.add_argument("--mask_sky", action="store_true", help="Apply sky segmentation to filter out sky points")
    return parser.parse_args()


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



def create_profiler_output_dir(base_dir, mode):
    if not os.path.isabs(base_dir):
        base_dir = os.path.join(REPO_ROOT, base_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(base_dir, f"{timestamp}_{mode}")
    os.makedirs(output_dir, exist_ok=True)
    return output_dir



def load_model(device, model_path):
    print("Initializing and loading VGGT model...")
    model = VGGT()
    model = load_vggt_weights(model, model_path)
    model.eval()
    model = model.to(device)
    return model



def load_images(image_folder, device):
    image_names = sorted(glob.glob(os.path.join(image_folder, "*")))
    if len(image_names) == 0:
        raise ValueError(f"No images found in {image_folder}")
    print(f"Found {len(image_names)} images")
    images = load_and_preprocess_images(image_names, mode="pad").to(device)
    print(f"Preprocessed images shape: {images.shape}")
    return image_names, images



def get_profiler_enums(args):
    profiler_level = getattr(torch_npu.profiler.ProfilerLevel, args.profiler_level)
    aic_metrics = getattr(torch_npu.profiler.AiCMetrics, args.aic_metrics)
    return profiler_level, aic_metrics



def build_profiler(args, output_dir):
    profiler_level, aic_metrics = get_profiler_enums(args)
    experimental_config = torch_npu.profiler._ExperimentalConfig(
        profiler_level=profiler_level,
        aic_metrics=aic_metrics,
        msprof_tx=True,
    )
    return torch_npu.profiler.profile(
        activities=[
            torch_npu.profiler.ProfilerActivity.CPU,
            torch_npu.profiler.ProfilerActivity.NPU,
        ],
        schedule=torch_npu.profiler.schedule(
            wait=0,
            warmup=args.warmup_steps,
            active=args.active_steps,
            repeat=args.schedule_repeat,
            skip_first=0,
        ),
        on_trace_ready=torch_npu.profiler.tensorboard_trace_handler(output_dir),
        record_shapes=args.record_shapes,
        profile_memory=args.profile_memory,
        with_stack=args.with_stack,
        with_modules=False,
        with_flops=args.with_flops,
        experimental_config=experimental_config,
    )



def get_mstx():
    try:
        return torch_npu.npu.mstx()
    except Exception as exc:
        print(f"[WARN] mstx is unavailable, profiling will continue without custom marks: {exc}")
        return None



def mstx_mark(mstx, label):
    if mstx is None:
        return
    try:
        mstx.mark(label)
    except Exception:
        pass



def run_profile_loop(model, images, mode, dtype, prof, mstx, total_steps):
    print(f"Profiling mode={mode} for {total_steps} steps")
    with prof:
        for step in range(total_steps):
            mstx_mark(mstx, f"profile_step_{step}_start")
            with torch.no_grad():
                with torch.cuda.amp.autocast(dtype=dtype):
                    if mode == "aggregator":
                        model.aggregator(images[None])
                    else:
                        model(images)
            synchronize_device()
            prof.step()
            mstx_mark(mstx, f"profile_step_{step}_end")



def run_full_forward_for_postprocess(model, images, dtype, mstx):
    mstx_mark(mstx, "postprocess_full_forward_start")
    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            predictions = model(images)
    synchronize_device()
    mstx_mark(mstx, "postprocess_full_forward_end")
    return predictions



def run_postprocess_and_visualize(predictions, images, args, mstx):
    from demo_viser import viser_wrapper

    mstx_mark(mstx, "pose_decode_start")
    extrinsic, intrinsic = pose_encoding_to_extri_intri(predictions["pose_enc"], images.shape[-2:])
    predictions["extrinsic"] = extrinsic
    predictions["intrinsic"] = intrinsic
    mstx_mark(mstx, "pose_decode_end")

    mstx_mark(mstx, "tensor_to_numpy_start")
    for key in list(predictions.keys()):
        if isinstance(predictions[key], torch.Tensor):
            predictions[key] = predictions[key].cpu().numpy().squeeze(0)
    mstx_mark(mstx, "tensor_to_numpy_end")

    mstx_mark(mstx, "viser_launch_start")
    viser_wrapper(
        predictions,
        port=args.port,
        init_conf_threshold=args.conf_threshold,
        use_point_map=args.use_point_map,
        background_mode=args.background_mode,
        mask_sky=args.mask_sky,
        image_folder=args.image_folder,
    )
    mstx_mark(mstx, "viser_launch_end")



def main():
    args = parse_args()
    device = get_runtime_device()
    dtype = get_inference_dtype()
    mstx = get_mstx()

    print(f"Using device: {device}")
    print(f"Using dtype: {dtype}")
    print(f"Profile mode: {args.mode}")

    model = load_model(device, args.model_path)

    mstx_mark(mstx, "load_images_start")
    image_names, images = load_images(args.image_folder, device)
    mstx_mark(mstx, "load_images_end")
    _ = image_names

    output_dir = create_profiler_output_dir(args.profile_dir, args.mode)
    profiler = build_profiler(args, output_dir)
    total_steps = args.warmup_steps + args.active_steps * args.schedule_repeat

    run_profile_loop(model, images, args.mode, dtype, profiler, mstx, total_steps)
    print(f"Profiler traces saved to: {output_dir}")

    if args.enable_postprocess:
        print("Running postprocess and visualization after profiling...")
        predictions = run_full_forward_for_postprocess(model, images, dtype, mstx)
        run_postprocess_and_visualize(predictions, images, args, mstx)


if __name__ == "__main__":
    main()

