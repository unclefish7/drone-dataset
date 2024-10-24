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

def visualize_point_cloud(point_cloud, lidar_pose):
    # 使用Open3D可视化点云
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(point_cloud)

    # 提取激光雷达位姿
    x, y, z, roll, yaw, pitch = lidar_pose

    # 创建激光雷达的坐标系原点
    lidar_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0, origin=[x, y, z])

    # 创建世界坐标系原点
    world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=2.0, origin=[0, 0, 0])

    # 可视化点云和坐标系
    o3d.visualization.draw_geometries([pcd, world_frame], window_name='Point Cloud Visualization')

    print("Coordinate frame legend:")
    print("- Red axis: X")
    print("- Green axis: Y")
    print("- Blue axis: Z")

def adjust_point_cloud_to_world(point_cloud, lidar_pose):
    # 将点云转换到世界坐标系，考虑激光雷达的位姿
    x, y, z, roll, yaw, pitch = lidar_pose
    # 计算旋转矩阵
    roll_rad = np.deg2rad(roll)
    pitch_rad = np.deg2rad(pitch)
    yaw_rad = np.deg2rad(yaw)

    # 计算绕 X 轴的旋转矩阵（Roll）
    R_x = np.array([
        [1, 0, 0],
        [0, np.cos(roll_rad), -np.sin(roll_rad)],
        [0, np.sin(roll_rad), np.cos(roll_rad)]
    ])

    # 计算绕 Y 轴的旋转矩阵（Pitch）
    R_y = np.array([
        [np.cos(pitch_rad), 0, np.sin(pitch_rad)],
        [0, 1, 0],
        [-np.sin(pitch_rad), 0, np.cos(pitch_rad)]
    ])

    # 计算绕 Z 轴的旋转矩阵（Yaw）
    R_z = np.array([
        [np.cos(yaw_rad), -np.sin(yaw_rad), 0],
        [np.sin(yaw_rad), np.cos(yaw_rad), 0],
        [0, 0, 1]
    ])

    # 组合旋转矩阵（ZYX 顺序）
    rotation = R_z @ R_y @ R_x

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

    intrinsics = np.array(yaml_data['intrinsic'])
    extrinsics = np.array(yaml_data['extrinsic'])
    x1, y1, z1, roll, yaw, pitch = yaml_data['cords']

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
    rotation = extrinsics[:3, :3]  # 旋转矩阵
    translation = extrinsics[:3, 3]  # 平移向量
    translation = translation.reshape(3, 1)
    
    # 对点云进行旋转和平移操作
    cam_points = rotation @ point_cloud.T + translation  # 形状 (3, N)

    Xc = cam_points[0, :]  # 相机坐标系的x坐标
    Yc = cam_points[1, :]  # 相机坐标系的y坐标
    Zc = cam_points[2, :]  # 相机坐标系的z坐标

    # 仅考虑在相机前方的点（Zc > 0）
    valid_depth = Zc > 0
    if not np.any(valid_depth):
        print(f"No points are in front of the camera in {image_path}")
        return False

    Xc = Xc[valid_depth]
    Yc = Yc[valid_depth]
    Zc = Zc[valid_depth]

    # 防止除零错误
    Zc[Zc == 0] = 1e-6

    # 计算归一化的图像坐标
    x = Xc / Zc
    y = Yc / Zc

    # 构建归一化的像素坐标
    image_points = np.vstack((x, y, np.ones_like(x)))  # 形状 (3, N')

    # 第二步：内参变换（从归一化图像坐标到像素坐标）
    pixel_coords = intrinsics @ image_points  # 形状 (3, N')

    # 归一化齐次坐标
    u = pixel_coords[0, :] / pixel_coords[2, :]
    v = pixel_coords[1, :] / pixel_coords[2, :]

    # 组合 u 和 v 得到像素坐标
    pixel_points = np.vstack((u, v)).T  # 形状 (N', 2)

    # 过滤有效的像素坐标
    valid_mask = (u >= 0) & (u < img_width) & (v >= 0) & (v < img_height)
    if not np.any(valid_mask):
        print(f"No valid projected points in {image_path}")
        return False

    # 可视化结果
    plt.figure(figsize=(10, 6))
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.scatter(u[valid_mask], v[valid_mask], s=1, c='red', label='Projected Points')
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
            visualize_point_cloud(point_cloud_world, lidar_pose=lidar_pose)

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