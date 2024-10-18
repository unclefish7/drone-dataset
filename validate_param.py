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
    point_cloud = np.asarray(pcd.points)
    return point_cloud

def visualize_point_cloud(point_cloud):
    # 使用Open3D可视化点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)
        # 将激光雷达位置添加为参考
    o3d.visualization.draw_geometries([pcd], window_name='Point Cloud Visualization')
    print("Coordinate frame legend: X-axis (Red), Y-axis (Green), Z-axis (Blue)")

def adjust_point_cloud_to_world(point_cloud, lidar_pose):
    # 将点云转换到世界坐标系，考虑激光雷达的位姿
    x, y, z, roll, yaw, pitch = lidar_pose
    # 计算旋转矩阵
    rotation = o3d.geometry.get_rotation_matrix_from_xyz([roll, pitch, yaw])
    
    # 先进行旋转
    rotated_points = point_cloud @ rotation.T  # 点云乘以旋转矩阵
    
    # 再进行平移
    translated_points = rotated_points + np.array([x, y, z])
    
    return translated_points

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
    # 第一步：外参变换（从世界坐标到相机坐标）
    # 提取外参的旋转矩阵和平移向量
    x, y, z, roll, yaw, pitch = lidar_pose
    # 计算旋转矩阵
    rotation = o3d.geometry.get_rotation_matrix_from_xyz([roll, pitch, yaw])
    translation = extrinsics[:3, 3]
    translation = translation.reshape(3, 1)
    
    # 对点云进行旋转操作
    cam_points = rotation @ point_cloud.T  # 点云乘以外参的旋转矩阵
    # 然后进行平移操作
    cam_points += translation  # 加上外参的平移向量

    # 第二步：内参变换（从相机坐标到像素坐标）
    pixel_coords = intrinsics @ cam_points  # 点云乘以内参矩阵
    # 将像素坐标除以z值（归一化）
    pixel_coords /= pixel_coords[2, :].reshape(1, -1)  # 保持列向量操作，归一化
    # 转置像素坐标回到原本的 (N, 3) 形状
    pixel_coords = pixel_coords.T  # 现在是 (N, 3)，方便后续过滤操作
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

        # 读取激光雷达的位姿
        if 'lidar_pose' in yaml_data:
            lidar_pose = yaml_data['lidar_pose']
            # 调整点云基准点到世界坐标系
            point_cloud_world = adjust_point_cloud_to_world(point_cloud, lidar_pose)

            # 可视化交换后的点云
            visualize_point_cloud(point_cloud_world)

            for cam_index in range(5):
                image_path = os.path.join(DATASET_FOLDER, f"{frame}_camera{cam_index}.png")
                try:
                    if f'camera{cam_index}' in yaml_data:
                        camera_data = yaml_data[f'camera{cam_index}']
                        if not validate_intrinsics_extrinsics(camera_data, image_path, point_cloud_world):
                            print(f"Validation failed for {image_path}")
                        else:
                            print(f"Validation succeeded for {image_path}")
                    else:
                        print(f"Camera data for camera{cam_index} not found in {yaml_path}")
                except FileNotFoundError as e:
                    print(e)