import torch
import trimesh
import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import glob
import os
import argparse

def load_ply_sequence(ply_folder):
    ply_files = sorted(glob.glob(os.path.join(ply_folder, "*color.ply")),
                      key=lambda x: int(os.path.basename(x).split('_')[0]))
    return [trimesh.load(f, process=False) for f in ply_files]  

def load_tensor_data(deform_pt_path, x_pt_path):
    deform_tensor = torch.load(deform_pt_path)  # (N, H, W, 3)
    x_tensor = torch.load(x_pt_path)            # (N, H, W, 3)
    
    assert deform_tensor.ndim == 4, f"Deform tensor should be 4 dims，but now {deform_tensor.ndim}"
    assert x_tensor.ndim == 4, f"X tensor should be 4 dims，but now{x_tensor.ndim}"
    
    return x_tensor, deform_tensor

def map_displacement_to_mesh(original_vertices, x_frame, deform_frame, method='nearest'):
    h, w, _ = x_frame.shape
    x_flat = x_frame.reshape(-1, 3)
    deform_flat = deform_frame.reshape(-1, 3)
    displacement = np.linalg.norm(deform_flat, axis=1)
    
    tree = cKDTree(x_flat)
    _, indices = tree.query(original_vertices, k=1, workers=-1)  
    return displacement[indices]

def generate_colored_ply(mesh, displacement, save_path, hist_save_path):
    valid_mask = ~np.isnan(displacement)
    if np.sum(valid_mask) == 0:
        print(f"warning: {save_path} doesn't have valid data，skip saving")
        return
    
    displacement_valid = displacement[valid_mask] * 1000 # Amplifying
    
    eps = 1e-8
    vmin, vmax = np.min(displacement), np.max(displacement)
    displacement_norm = (displacement - vmin) / (vmax - vmin + eps)
    
    colors = plt.cm.viridis(displacement_norm)[:, :3]
    colors = (colors * 255).astype(np.uint8)
    
    colored_mesh = mesh.copy()
    colored_mesh.visual.vertex_colors = colors
    
    colored_mesh.export(save_path, file_type="ply", encoding="binary")
    
    plt.figure(figsize=(10, 6))
    hist = plt.hist(displacement_valid, 
                   bins=np.linspace(0, np.max(displacement_valid), 50),
                   density=True,
                   edgecolor='black', 
                   alpha=0.7)
    plt.title(f"Displacement Distribution (max={np.max(displacement_valid):.2f})")
    plt.xlabel("Displacement")
    plt.ylabel("Probability Density")
    plt.grid(alpha=0.3)
    plt.savefig(hist_save_path, dpi=150, bbox_inches='tight')
    plt.close()

def process_sequence(ply_sequence, x_seq, deform_seq, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    
    for t, ply in enumerate(ply_sequence):
        print(f"process {t+1}/{len(ply_sequence)}th frame...")
        
        current_x = x_seq[t]
        current_deform = deform_seq[t]
        
        displacement = map_displacement_to_mesh(
            ply.vertices,
            current_x,
            current_deform
        )
        
        frame_str = f"{t:03d}"
        ply_path = os.path.join(output_dir, f"frame_{frame_str}_color.ply")
        hist_path = os.path.join(output_dir, f"frame_{frame_str}_hist.png")
        
        generate_colored_ply(ply, displacement, ply_path, hist_path)

def main():
    parser = argparse.ArgumentParser(description='Process mesh sequence with displacement visualization')
    parser.add_argument('--ply-folder', type=str, default="./endosurf/logs/endosurf/base-scared2019-dataset_2_keyframe_1_disparity/demo/iter_00100000/all_3d_thresh_0_res_128",
                       help='Relative path to folder containing PLY files')
    parser.add_argument('--deform-pt', type=str, default="./endosurf/logs/endosurf/interpolated-scared2019-dataset_2_keyframe_1_disparity-test/out_deform.pt",
                       help='Relative path to deformation tensor file (.pt)')
    parser.add_argument('--x-pt', type=str, default="./endosurf/logs/endosurf/interpolated-scared2019-dataset_2_keyframe_1_disparity-test/out_x_0.pt",
                       help='Relative path to x tensor file (.pt)')
    parser.add_argument('--output-dir', type=str, default='./heatmap_ply_d2k1',
                       help='Relative path for output directory')
    
    args = parser.parse_args()
    
 
    ply_folder = os.path.abspath(args.ply_folder)
    deform_pt_path = os.path.abspath(args.deform_pt)
    x_pt_path = os.path.abspath(args.x_pt)
    output_dir = os.path.abspath(args.output_dir)

    print("Load PLY files...")
    ply_meshes = load_ply_sequence(ply_folder)
    print(f"Successfully loaded {len(ply_meshes)} PLY files")
    
    print("Start process...")
    x_seq, deform_seq = load_tensor_data(deform_pt_path, x_pt_path)
    
    process_sequence(ply_meshes, x_seq, deform_seq, output_dir)

if __name__ == "__main__":
    main()

