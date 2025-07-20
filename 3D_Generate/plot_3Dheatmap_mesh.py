import torch
import trimesh
import numpy as np
from scipy.spatial import cKDTree  # 用于坐标匹配的备选方案
import matplotlib.pyplot as plt
import glob
import os

def load_mesh_sequence(mesh_folder):
    """加载按序号排列的网格文件序列"""
    mesh_files = sorted(glob.glob(os.path.join(mesh_folder, "*_color.obj")),
                      key=lambda x: int(os.path.basename(x).split('_')[0]))
    return [trimesh.load(f) for f in mesh_files]

def load_tensor_data(deform_pt_path, x_pt_path):
    """加载变形张量和坐标数据"""
    deform_tensor = torch.load(deform_pt_path)  # (N, H, W, 3)
    x_tensor = torch.load(x_pt_path)            # (N, H, W, 3)
    
    # 维度验证
    assert deform_tensor.ndim == 4, f"Deform tensor应为4维，实际为{deform_tensor.ndim}"
    assert x_tensor.ndim == 4, f"X tensor应为4维，实际为{x_tensor.ndim}"
    
    return x_tensor, deform_tensor

def map_displacement_to_mesh(original_vertices, x_frame, deform_frame, method='raw'):
    """
    位移映射方法选择器
    method可选: 
    - 'raw'    直接使用原始数据（需严格对齐）
    - 'nearest' 最近邻匹配（推荐）
    """
    h, w, _ = x_frame.shape
    x_flat = x_frame.reshape(-1, 3)        # (H*W, 3)
    deform_flat = deform_frame.reshape(-1, 3)
    displacement = np.linalg.norm(deform_flat, axis=1)
    
    if method == 'raw':
        # 直接匹配模式
        assert len(original_vertices) == len(x_flat), \
            f"顶点数不匹配: 网格{len(original_vertices)} vs 数据{len(x_flat)}"
        
        if not np.allclose(original_vertices, x_flat, atol=1e-6):
            print("警告：检测到坐标不严格对齐，建议使用method='nearest'")
        
        return displacement
    elif method == 'nearest':
        # 最近邻匹配模式
        tree = cKDTree(x_flat)
        _, indices = tree.query(original_vertices, k=1)
        return displacement[indices]
    else:
        raise ValueError(f"不支持的method参数: {method}")

def generate_colored_mesh(mesh, displacement, save_path, hist_save_path):
    """生成带颜色映射的网格"""
    # 数据清洗
    valid_mask = ~np.isnan(displacement)
    if np.sum(valid_mask) == 0:
        print(f"警告: {save_path} 无有效数据，跳过保存")
        return
    
    displacement_valid = displacement[valid_mask] * 1000  # 转毫米
    
    # 统计信息
    stats = {
        "min": np.min(displacement_valid),
        "max": np.max(displacement_valid),
        "mean": np.mean(displacement_valid),
        "std": np.std(displacement_valid)
    }
    
    # 颜色映射
    eps = 1e-8
    displacement_norm = (displacement - np.min(displacement)) / (
        np.max(displacement) - np.min(displacement) + eps)
    colors = plt.cm.viridis(displacement_norm)[:, :3]
    colors = (colors * 255).astype(np.uint8)
    
    # 创建彩色网格
    colored_mesh = mesh.copy()
    colored_mesh.visual.vertex_colors = colors
    
    # 创建对比视图
    #combined = trimesh.util.concatenate([
    #    mesh,
    #    colored_mesh.apply_translation([3, 0, 0])
    #])

    combined = trimesh.util.concatenate([
        colored_mesh
    ])

    # 保存结果
    combined.export(save_path)
    
    # 生成直方图
    plt.figure(figsize=(10, 6))
    plt.hist(displacement_valid, bins=50, density=True, 
            edgecolor='black', alpha=0.7)
    plt.title(f"Displacement Distribution\nμ={stats['mean']:.2f} ± {stats['std']:.2f} mm")
    plt.xlabel("Displacement (mm)")
    plt.ylabel("Probability Density")
    plt.savefig(hist_save_path, dpi=150, bbox_inches='tight')
    plt.close()

def process_sequence(mesh_sequence, x_seq, deform_seq, output_dir="output", method='nearest'):
    """处理时间序列"""
    os.makedirs(output_dir, exist_ok=True)
    
    for t, mesh in enumerate(mesh_sequence):
        print(f"处理第 {t+1}/{len(mesh_sequence)} 帧...")
        
        # 获取当前帧数据
        current_x = x_seq[t]      # 转为numpy数组
        current_deform = deform_seq[t]
        
        # 计算位移映射
        displacement = map_displacement_to_mesh(
            mesh.vertices,
            current_x,
            current_deform,
            method=method
        )
        
        # 生成输出路径
        frame_str = f"{t:03d}"
        mesh_path = os.path.join(output_dir, f"frame_{frame_str}_colored.obj")
        hist_path = os.path.join(output_dir, f"frame_{frame_str}_hist.png")
        
        # 保存结果
        generate_colored_mesh(mesh, displacement, mesh_path, hist_path)

def main():
    # 参数配置
    mesh_folder = "/bd_targaryen/users/omnimotion/ZeqingTracker/Surgical-Visual-Force-Field/endosurf/logs/endosurf/base-scared2019-dataset_1_keyframe_1_disparity/demo/iter_00100000/all_3d_thresh_0_res_128"
    deform_pt_path = "/bd_targaryen/users/omnimotion/ZeqingTracker/endosurf/logs/endosurf/interpolated-endonerf-pulling_soft_tissues-test/out_deform.pt"
    x_pt_path = "/bd_targaryen/users/omnimotion/ZeqingTracker/endosurf/logs/endosurf/interpolated-endonerf-pulling_soft_tissues-test/out_x_0.pt"
    
    # 加载数据
    print("正在加载网格序列...")
    meshes = load_mesh_sequence(mesh_folder)
    print(f"成功加载 {len(meshes)} 个网格帧")
    
    print("加载张量数据...")
    x_seq, deform_seq = load_tensor_data(deform_pt_path, x_pt_path)
    print(f"数据维度 - X: {x_seq.shape}, Deform: {deform_seq.shape}")
    
    # 数据一致性检查
    assert len(meshes) == x_seq.shape[0], "网格帧数与X数据不匹配"
    assert len(meshes) == deform_seq.shape[0], "网格帧数与Deform数据不匹配"
    
    # 处理序列（method可选 'raw' 或 'nearest'）
    process_sequence(meshes, x_seq, deform_seq, 
                    output_dir="./heatmap_results_pulling",
                    method='nearest')

if __name__ == "__main__":
    main()