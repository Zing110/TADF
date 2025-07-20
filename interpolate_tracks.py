import os
import torch
import argparse
import cotracker.predictor as predictor
import imageio.v3 as iio
import torch.nn.functional as F
import numpy as np


import torch
import torch.nn.functional as F
import numpy as np

def interpolate_tracks(pred_tracks, target_height, target_width):
    """
    Interpolate the grid-based tracks to the target size (H, W).
    
    Args:
        pred_tracks (numpy.ndarray): Shape (num_frames, grid_h, grid_w, 2)
        target_height (int): Target height (H) for interpolation.
        target_width (int): Target width (W) for interpolation.
    
    Returns:
        torch.Tensor: Interpolated tracks with shape (num_frames, H, W, 2).
    """
    num_frames, grid_h, grid_w, _ = pred_tracks.shape
    
    # Convert to tensor and permute dimensions to [Batch, 2, H, W]
    pred_tracks_tensor = torch.tensor(pred_tracks, dtype=torch.float32)
    pred_tracks_tensor = pred_tracks_tensor.permute(0, 3, 1, 2)  # [num_frames, 2, grid_h, grid_w]

    # Generate normalized grid [-1, 1] for target size
    grid_y, grid_x = torch.meshgrid(
        torch.linspace(-1, 1, target_height),
        torch.linspace(-1, 1, target_width),
        indexing='ij'  # 确保坐标顺序正确
    )
    grid = torch.stack((grid_x, grid_y), dim=-1)  # [H, W, 2]

    # Expand grid to match batch size
    grid = grid.unsqueeze(0).repeat(num_frames, 1, 1, 1)  # [num_frames, H, W, 2]

    # Apply grid_sample
    interpolated_tracks = F.grid_sample(
        pred_tracks_tensor, 
        grid, 
        mode='bilinear', 
        align_corners=True
    )  # 输出形状 [num_frames, 2, H, W]

    # Permute back to [num_frames, H, W, 2]
    interpolated_tracks = interpolated_tracks.permute(0, 2, 3, 1)
    return interpolated_tracks

def process_video(frames, cotracker, save_dir):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    video = torch.tensor(frames).permute(0, 3, 1, 2)[None].float().to(device)

    # Get predictions from cotracker
    pred_tracks, pred_visibility, grid_pts = cotracker(video, grid_size=30)

    # Convert to numpy and check shape
    pred_tracks_np = pred_tracks[0].detach().cpu().numpy()
    print(f"Shape of pred_tracks_np: {pred_tracks_np.shape}")

    # Validate grid size
    H, W = frames.shape[1], frames.shape[2]
    expected_grid_size = 30 * 30
    if pred_tracks_np.shape[1] != expected_grid_size:
        raise ValueError(f"Expected {expected_grid_size} grid points, got {pred_tracks_np.shape[1]}")

    # Reshape to (num_frames, 30, 30, 2)
    num_frames = pred_tracks_np.shape[0]
    pred_tracks_reshaped = pred_tracks_np.reshape(num_frames, 30, 30, 2)

    # Interpolate tracks
    interpolated_tracks = interpolate_tracks(pred_tracks_reshaped, H, W)

    # Save results
    output_dict = {
        'pred_tracks': pred_tracks.detach().cpu(),
        'pred_visibility': pred_visibility.detach().cpu(),
        'grid_pts': grid_pts.detach().cpu() if grid_pts is not None else None,
        'interpolated_tracks': interpolated_tracks
    }

    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'cotracker_output.pt')
    torch.save(output_dict, save_path)
    print(f"Saved results to {save_path}")

def process_full_video(video_path, save_dir, cotracker):
    os.makedirs(save_dir, exist_ok=True)
    frames = iio.imread(video_path, plugin="FFMPEG")
    print("Processing full video...")
    process_video(frames, cotracker, save_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--video-path', type=str, required=True, help='Path to the input video file')
    parser.add_argument('--save-dir', type=str, required=True, help='Directory to save the results')
    args = parser.parse_args()

    # Load the CoTracker model
    cotracker = predictor.CoTrackerPredictor('/bd_targaryen/users/omnimotion/CoTracker/co-tracker/checkpoints/cotracker2.pth')
    cotracker.to('cuda' if torch.cuda.is_available() else 'cpu')

    # Process the entire video and save results with interpolation
    process_full_video(args.video_path, args.save_dir, cotracker)
