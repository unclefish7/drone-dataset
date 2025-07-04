import open3d as o3d
import numpy as np
import os
from collections import defaultdict

def analyze_voxel_point_counts(root_dir):
    voxel_size = [0.4, 0.4, 3]
    pc_range = [-102.4, -102.4, 0.5, 102.4, 102.4, 3.5]
    x_min, y_min, z_min, x_max, y_max, z_max = pc_range
    voxel_x, voxel_y, voxel_z = voxel_size

    all_voxel_point_counts = []
    file_pillar_counts = []  # 记录每个文件的非空pillar数

    # 递归遍历所有 .pcd 文件
    pcd_file_list = []
    for root, _, files in os.walk(root_dir):
        for fname in files:
            if fname.endswith('.pcd'):
                pcd_file_list.append(os.path.join(root, fname))

    total_files = len(pcd_file_list)
    if total_files == 0:
        print("❗ 没有找到任何 .pcd 文件。")
        return

    for idx, file_path in enumerate(pcd_file_list):
        progress = (idx + 1) / total_files * 100
        print(f"\r进度：{progress:5.1f}% - 正在处理：{os.path.basename(file_path)}", end="")

        try:
            pcd = o3d.io.read_point_cloud(file_path)
            points = np.asarray(pcd.points)
            if points.shape[0] == 0:
                continue

            # 筛选 ROI 范围内的点
            mask = (
                (points[:, 0] >= x_min) & (points[:, 0] < x_max) &
                (points[:, 1] >= y_min) & (points[:, 1] < y_max) &
                (points[:, 2] >= z_min) & (points[:, 2] < z_max)
            )
            points = points[mask]
            if points.shape[0] == 0:
                continue

            # 将每个点映射到 voxel 索引
            voxel_indices = np.floor((points - [x_min, y_min, z_min]) / voxel_size).astype(np.int32)
            voxel_counts = defaultdict(int)
            for idx in voxel_indices:
                voxel_key = tuple(idx)
                voxel_counts[voxel_key] += 1

            # 记录当前文件的非空pillar数
            current_file_pillar_count = len(voxel_counts)
            file_pillar_counts.append(current_file_pillar_count)
            
            # 统计所有非空 voxel 的点数
            all_voxel_point_counts.extend(voxel_counts.values())

        except Exception as e:
            print(f"\n⚠️ 读取失败: {file_path}，错误：{e}")

    print()  # 换行

    if not all_voxel_point_counts:
        print("❗ 没有任何非空 pillar 数据。")
        return

    counts_array = np.array(all_voxel_point_counts)
    file_pillar_counts_array = np.array(file_pillar_counts)

    print("\n========== Pillar统计结果 ==========")
    print(f"处理的pcd文件数：{len(file_pillar_counts)}")
    print(f"总非空 pillar 数：{len(counts_array)}")
    print(f"平均每个pcd文件的非空pillar数：{file_pillar_counts_array.mean():.2f}")
    print(f"每个pcd文件非空pillar数范围：{file_pillar_counts_array.min()} - {file_pillar_counts_array.max()}")
    
    print("\n========== Points per Pillar统计 ==========")
    print(f"最少points_per_pillar：{counts_array.min()}")
    print(f"最多points_per_pillar：{counts_array.max()}")
    print(f"平均points_per_pillar：{counts_array.mean():.2f}")
    print(f"中位数points_per_pillar：{np.median(counts_array):.2f}")
    print(f"标准差points_per_pillar：{counts_array.std():.2f}")

# 示例调用
if __name__ == "__main__":
    dataset_root = r"E:\datasets\OPV2V\train"  # 替换为你的根目录
    analyze_voxel_point_counts(dataset_root)
