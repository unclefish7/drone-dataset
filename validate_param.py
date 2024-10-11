import os
import yaml
import open3d as o3d
import numpy as np
import cv2
import matplotlib.pyplot as plt

# 定义文件夹路径
DATASET_FOLDER = r'C:\Users\uncle\_Projects\Carla\CARLA_Latest\WindowsNoEditor\myDemo\dataset\4'

# 遍历文件夹中的yaml文件
def load_yaml(yaml_path):
    with open(yaml_path, 'r') as file:
        data = yaml.safe_load(file)
    return data

def load_point_cloud(pcd_path):
    pcd = o3d.io.read_point_cloud(pcd_path)
    return np.asarray(pcd.points)

def visualize_point_cloud(point_cloud):
    # 使用Open3D可视化点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
    o3d.visualization.draw_geometries([pcd], window_name='Point Cloud Visualization')

def validate_intrinsics_extrinsics(yaml_data, image_path, point_cloud):
    # 从yaml数据中提取内参和外参
    if 'intrinsic' not in yaml_data or 'extrinsic' not in yaml_data:
        print(f"Missing intrinsic or extrinsic data in {image_path}")
        return False

    intrinsics = np.array(yaml_data['intrinsic']).reshape(3, 3)
    extrinsics = np.array(yaml_data['extrinsic']).reshape(4, 4)

    # 读取图像并进行检测
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image at path {image_path} could not be loaded.")
    img_height, img_width = image.shape[:2]

    # 验证相机内参（比如校验矩阵特征）
    if not (intrinsics.shape == (3, 3)):
        print(f"Invalid intrinsics matrix size in {image_path}")
        return False

    # 验证外参（通过点云数据的映射到图像进行验证）
    # 将点云转换为齐次坐标
    point_cloud_h = np.hstack((point_cloud, np.ones((point_cloud.shape[0], 1))))
    # 使用外参将点云从世界坐标变换到相机坐标
    cam_points = (extrinsics @ point_cloud_h.T).T
    # 使用内参将相机坐标变换到像素坐标
    pixel_coords = (intrinsics @ cam_points[:, :3].T).T
    pixel_coords /= pixel_coords[:, 2].reshape(-1, 1)
    # 过滤有效的像素坐标
    valid_mask = (pixel_coords[:, 0] >= 0) & (pixel_coords[:, 0] < img_width) & (pixel_coords[:, 1] >= 0) & (pixel_coords[:, 1] < img_height)
    if not np.any(valid_mask):
        print(f"Invalid extrinsics or intrinsics in {image_path}")
        return False

    # 可视化结果
    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.scatter(pixel_coords[valid_mask, 0], pixel_coords[valid_mask, 1], s=1, c='red', label='Projected Points')
    plt.title(f"Projection of Point Cloud on Image: {image_path}")
    plt.legend()
    plt.show()

    return True

# 主函数
for frame_file in os.listdir(DATASET_FOLDER):
    if frame_file.endswith('.yaml'):
        frame = frame_file.split('.')[0]
        yaml_path = os.path.join(DATASET_FOLDER, frame_file)
        pcd_path = os.path.join(DATASET_FOLDER, f"{frame}.pcd")
        yaml_data = load_yaml(yaml_path)
        point_cloud = load_point_cloud(pcd_path)

        # 可视化点云
        visualize_point_cloud(point_cloud)

        for cam_index in range(5):
            image_path = os.path.join(DATASET_FOLDER, f"{frame}_camera{cam_index}.png")
            try:
                if f'camera{cam_index}' in yaml_data:
                    camera_data = yaml_data[f'camera{cam_index}']
                    if not validate_intrinsics_extrinsics(camera_data, image_path, point_cloud):
                        print(f"Validation failed for {image_path}")
                    else:
                        print(f"Validation succeeded for {image_path}")
                else:
                    print(f"Camera data for camera{cam_index} not found in {yaml_path}")
            except FileNotFoundError as e:
                print(e)