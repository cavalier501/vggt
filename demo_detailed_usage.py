import torch
import torch_npu
from torch_npu.contrib import transfer_to_npu
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images
from vggt.utils.pose_enc import pose_encoding_to_extri_intri

import os
import argparse

def get_all_files_paths(directory):
    file_paths = []
    for root, dirs, files in os.walk(directory):
        for file in files:
            file_path = os.path.join(root, file)
            file_paths.append(file_path)
    return file_paths

def detailed_usage(pt_path, images_path):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    # bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+)
    dtype = torch.bfloat16

    # Initialize the model and load the pretrained weights.
    # This will automatically download the model weights the first time it's run, which may take a while.
    model = VGGT()
    model.load_state_dict(torch.load(pt_path))
    model = model.to(device)

    # Load and preprocess example images (replace with your own image paths)
    image_names = get_all_files_paths(images_path)
    images = load_and_preprocess_images(image_names).to(device)

    with torch.no_grad():
        with torch.cuda.amp.autocast(dtype=dtype):
            images = images[None]  # add batch dimension
            aggregated_tokens_list, ps_idx = model.aggregator(images)

        # Predict Cameras
        pose_enc = model.camera_head(aggregated_tokens_list)[-1]
        # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
        extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])

        # Predict Depth Maps
        depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, ps_idx)

        # Predict Point Maps
        point_map, point_conf = model.point_head(aggregated_tokens_list, images, ps_idx)

        # Predict Tracks
        # choose your own points to track, with shape (N, 2) for one scene
        query_points = torch.FloatTensor([[100.0, 200.0],
                                            [60.72, 259.94]]).to(device)
        track_list, vis_score, conf_score = model.track_head(aggregated_tokens_list, images, ps_idx, query_points=query_points[None])

def parse_args():
    parser = argparse.ArgumentParser(description='VGGT quick start')
    parser.add_argument('--images_path', type=str, default='examples/kitchen/images', help='images_path')
    parser.add_argument('--pt_path', type=str, default='model.pt', help='pretrained_model')
    args = parser.parse_args()

    return args

def main():
    args = parse_args()
    detailed_usage(args.pt_path, args.images_path)

if __name__ == '__main__':
    main()