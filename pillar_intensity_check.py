import open3d as o3d
import numpy as np
import os
from collections import defaultdict

def analyze_voxel_point_counts(root_dir):
    voxel_size = [0.4, 0.4, 10.0]
    pc_range = [-100, -100, -10, 100, 100, 10]
    x_min, y_min, z_min, x_max, y_max, z_max = pc_range
    voxel_x, voxel_y, voxel_z = voxel_size

    all_voxel_point_counts = []

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

            # 统计所有非空 voxel 的点数
            all_voxel_point_counts.extend(voxel_counts.values())

        except Exception as e:
            print(f"\n⚠️ 读取失败: {file_path}，错误：{e}")

    print()  # 换行

    if not all_voxel_point_counts:
        print("❗ 没有任何非空 pillar 数据。")
        return

    counts_array = np.array(all_voxel_point_counts)

    print("\n========== 每个非空 pillar 内的点数统计 ==========")
    print(f"总非空 pillar 数：{len(counts_array)}")
    print(f"最小点数/voxel：{counts_array.min()}")
    print(f"最大点数/voxel：{counts_array.max()}")
    print(f"平均点数/voxel：{counts_array.mean():.2f}")
    print(f"中位数点数/voxel：{np.median(counts_array):.2f}")

# 示例调用
if __name__ == "__main__":
    dataset_root = r"D:\datasets\mydataset\test"  # 替换为你的根目录
    analyze_voxel_point_counts(dataset_root)
