import torch
import trimesh
import numpy as np
from scipy.spatial import cKDTree
import matplotlib.pyplot as plt
import glob
import os

def load_ply_sequence(ply_folder):
    """加载PLY格式的网格序列"""
    ply_files = sorted(glob.glob(os.path.join(ply_folder, "*color.ply")),
                      key=lambda x: int(os.path.basename(x).split('_')[0]))
    return [trimesh.load(f, process=False) for f in ply_files]  # 禁用自动处理保持原始数据

def load_tensor_data(deform_pt_path, x_pt_path):
    """加载变形张量和坐标数据（保持原功能不变）"""
    deform_tensor = torch.load(deform_pt_path)  # (N, H, W, 3)
    x_tensor = torch.load(x_pt_path)            # (N, H, W, 3)
    
    # 维度验证
    assert deform_tensor.ndim == 4, f"Deform tensor应为4维，实际为{deform_tensor.ndim}"
    assert x_tensor.ndim == 4, f"X tensor应为4维，实际为{x_tensor.ndim}"
    
    return x_tensor, deform_tensor

def map_displacement_to_mesh(original_vertices, x_frame, deform_frame, method='nearest'):
    """位移映射方法（优化最近邻匹配性能）"""
    h, w, _ = x_frame.shape
    x_flat = x_frame.reshape(-1, 3)
    deform_flat = deform_frame.reshape(-1, 3)
    displacement = np.linalg.norm(deform_flat, axis=1)
    
    # 使用KDTree进行快速匹配
    tree = cKDTree(x_flat)
    _, indices = tree.query(original_vertices, k=1, workers=-1)  # 启用多线程
    return displacement[indices]

def generate_colored_ply(mesh, displacement, save_path, hist_save_path):
    """生成带颜色映射的PLY文件"""
    # 数据验证
    valid_mask = ~np.isnan(displacement)
    if np.sum(valid_mask) == 0:
        print(f"警告: {save_path} 无有效数据，跳过保存")
        return
    
    # 转换为毫米单位
    displacement_valid = displacement[valid_mask] * 1000
    
    # 创建颜色映射（优化颜色过渡）
    eps = 1e-8
    vmin, vmax = np.min(displacement), np.max(displacement)
    displacement_norm = (displacement - vmin) / (vmax - vmin + eps)
    
    # 使用matplotlib的viridis色彩映射[1,5](@ref)
    colors = plt.cm.viridis(displacement_norm)[:, :3]
    colors = (colors * 255).astype(np.uint8)
    
    # 创建PLY网格对象
    colored_mesh = mesh.copy()
    colored_mesh.visual.vertex_colors = colors
    
    # 保存为PLY格式（优化导出参数）
    colored_mesh.export(save_path, file_type="ply", encoding="binary")
    
    # 生成统计直方图（优化可视化）
    plt.figure(figsize=(10, 6))
    hist = plt.hist(displacement_valid, 
                   bins=np.linspace(0, np.max(displacement_valid), 50),
                   density=True,
                   edgecolor='black', 
                   alpha=0.7)
    plt.title(f"Displacement Distribution (max={np.max(displacement_valid):.2f}mm)")
    plt.xlabel("Displacement (mm)")
    plt.ylabel("Probability Density")
    plt.grid(alpha=0.3)
    plt.savefig(hist_save_path, dpi=150, bbox_inches='tight')
    plt.close()

def process_sequence(ply_sequence, x_seq, deform_seq, output_dir="./heatmap_ply_d2k1"):
    """处理PLY文件序列"""
    os.makedirs(output_dir, exist_ok=True)
    
    for t, ply in enumerate(ply_sequence):
        print(f"处理第 {t+1}/{len(ply_sequence)} 帧...")
        
        # 获取当前帧数据
        current_x = x_seq[t]
        current_deform = deform_seq[t]
        
        # 计算位移映射
        displacement = map_displacement_to_mesh(
            ply.vertices,
            current_x,
            current_deform
        )
        
        # 生成输出路径
        frame_str = f"{t:03d}"
        ply_path = os.path.join(output_dir, f"frame_{frame_str}_color.ply")
        hist_path = os.path.join(output_dir, f"frame_{frame_str}_hist.png")
        
        # 保存结果
        generate_colored_ply(ply, displacement, ply_path, hist_path)

def main():
    # 参数配置
    ply_folder = "/bd_targaryen/users/omnimotion/ZeqingTracker/Surgical-Visual-Force-Field/endosurf/logs/endosurf/base-scared2019-dataset_2_keyframe_1_disparity/demo/iter_00100000/all_3d_thresh_0_res_128"
    deform_pt_path = "/bd_targaryen/users/omnimotion/ZeqingTracker/endosurf/logs/endosurf/interpolated-scared2019-dataset_2_keyframe_1_disparity-test/out_deform.pt"
    x_pt_path = "/bd_targaryen/users/omnimotion/ZeqingTracker/endosurf/logs/endosurf/interpolated-scared2019-dataset_2_keyframe_1_disparity-test/out_x_0.pt"
    
    # 加载数据
    print("正在加载PLY序列...")
    ply_meshes = load_ply_sequence(ply_folder)
    print(f"成功加载 {len(ply_meshes)} 个PLY文件")
    
    print("加载张量数据...")
    x_seq, deform_seq = load_tensor_data(deform_pt_path, x_pt_path)
    
    # 处理序列
    process_sequence(ply_meshes, x_seq, deform_seq)

if __name__ == "__main__":
    main()