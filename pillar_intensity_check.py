import open3d as o3d
import numpy as np
import os

def compute_voxel_occupancy(points, voxel_size, pc_range):
    x_min, y_min, z_min, x_max, y_max, z_max = pc_range
    voxel_x, voxel_y, voxel_z = voxel_size

    mask = (
        (points[:, 0] >= x_min) & (points[:, 0] < x_max) &
        (points[:, 1] >= y_min) & (points[:, 1] < y_max) &
        (points[:, 2] >= z_min) & (points[:, 2] < z_max)
    )
    points = points[mask]
    if points.shape[0] == 0:
        return 0, 0, 0

    voxel_indices = np.floor((points - [x_min, y_min, z_min]) / voxel_size).astype(np.int32)
    voxel_keys = set(tuple(vidx) for vidx in voxel_indices)

    nx = int((x_max - x_min) / voxel_x)
    ny = int((y_max - y_min) / voxel_y)
    nz = int((z_max - z_min) / voxel_z)
    total_voxels = nx * ny * nz

    return len(voxel_keys), total_voxels, len(points)

def analyze_dataset_voxel_occupancy(root_dir, max_uav_per_scene=4):
    voxel_size = [0.4, 0.4, 10.0]
    pc_range = [-100, -100, -10, 100, 100, 10]

    # === 预扫描所有 .pcd 文件路径 ===
    pcd_file_list = []
    for split in ['train', 'test', 'validate']:
        split_dir = os.path.join(root_dir, split)
        if not os.path.exists(split_dir):
            continue
        for scene in os.listdir(split_dir):
            scene_dir = os.path.join(split_dir, scene)
            if not os.path.isdir(scene_dir):
                continue
            uav_folders = [f for f in os.listdir(scene_dir) if os.path.isdir(os.path.join(scene_dir, f))]
            uav_folders = uav_folders[:max_uav_per_scene]
            for uav_folder in uav_folders:
                uav_dir = os.path.join(scene_dir, uav_folder)
                for fname in os.listdir(uav_dir):
                    if fname.endswith(".pcd"):
                        pcd_file_list.append(os.path.join(uav_dir, fname))

    # === 开始处理并显示进度 ===
    total_files = len(pcd_file_list)
    if total_files == 0:
        print("❗ 没有找到任何 .pcd 文件。")
        return

    ratios = []
    file_count = 0
    max_ratio = -1
    min_ratio = 1
    total_points = 0

    for idx, file_path in enumerate(pcd_file_list):
        progress = (idx + 1) / total_files * 100
        print(f"\r进度：{progress:5.1f}% - 正在处理：{os.path.basename(file_path)}", end="")

        try:
            pcd = o3d.io.read_point_cloud(file_path)
            points = np.asarray(pcd.points)
            if points.shape[0] == 0:
                continue
            occupied, total, inroi = compute_voxel_occupancy(points, voxel_size, pc_range)
            if total == 0:
                continue
            ratio = occupied / total
            ratios.append(ratio)
            file_count += 1
            total_points += inroi
            max_ratio = max(max_ratio, ratio)
            min_ratio = min(min_ratio, ratio)
        except Exception as e:
            print(f"\n⚠️ 读取失败: {file_path}，错误：{e}")

    print()  # 换行

    if file_count == 0:
        print("❗ 没有成功分析的点云文件。")
        return

    avg_ratio = sum(ratios) / len(ratios)

    print("\n========== 数据集稀疏度分析结果 ==========")
    print(f"总分析文件数：{file_count}")
    print(f"平均 ROI 点数/帧：{total_points // file_count}")
    print(f"最大非空pillar占比：{max_ratio:.4%}")
    print(f"最小非空pillar占比：{min_ratio:.4%}")
    print(f"平均非空pillar占比：{avg_ratio:.4%}")

# 示例调用
if __name__ == "__main__":
    dataset_root = r"D:\datasets\mydataset"  # 替换为你的数据集根目录
    analyze_dataset_voxel_occupancy(dataset_root, max_uav_per_scene=3)
