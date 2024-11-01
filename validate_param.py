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
    lidar_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10.0, origin=[x, y, z])

    # 创建世界坐标系原点
    world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10.0, origin=[0, 0, 0])

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

    # 读取图像并检查
    image = cv2.imread(image_path)
    if image is None:
        raise FileNotFoundError(f"Image at path {image_path} could not be loaded.")
    img_height, img_width = image.shape[:2]

    # 验证内参矩阵大小
    if intrinsics.shape != (3, 3):
        print(f"Invalid intrinsics matrix size in {image_path}")
        return False

    # 提取外参的旋转矩阵和平移向量
    rotation = extrinsics[:3, :3]
    translation = extrinsics[:3, 3].reshape(3, 1)
    
    # 可视化原始点云
    original_pcd = o3d.geometry.PointCloud()
    original_pcd.points = o3d.utility.Vector3dVector(point_cloud)
    world_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10.0, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([original_pcd, world_frame], window_name="Original Point Cloud (World Coordinates)")

    # 第一步旋转：将点云从世界坐标系旋转到相机坐标系
    rotated_points = rotation @ point_cloud.T
    rotated_pcd = o3d.geometry.PointCloud()
    rotated_pcd.points = o3d.utility.Vector3dVector(rotated_points.T)
    o3d.visualization.draw_geometries([rotated_pcd, world_frame], window_name="Rotated Point Cloud")

    # 第二步平移：应用平移到旋转后的点云
    cam_points = rotated_points + translation  # (3, N)
    cam_point_cloud = cam_points.T  # 转置为 (N, 3)
    cam_pcd = o3d.geometry.PointCloud()
    cam_pcd.points = o3d.utility.Vector3dVector(cam_point_cloud)

    # 可视化相机坐标系下的点云以及坐标原点
    cam_frame = o3d.geometry.TriangleMesh.create_coordinate_frame(size=10.0, origin=[0, 0, 0])
    o3d.visualization.draw_geometries([cam_pcd, cam_frame], window_name="Translated Point Cloud (Camera Coordinates)")
    
    # 仅考虑在相机前方的点
    Zc = cam_points[2, :]
    valid_depth = Zc > 0
    if not np.any(valid_depth):
        print(f"No points are in front of the camera in {image_path}")
        return False

    Xc = cam_points[0, valid_depth]
    Yc = cam_points[1, valid_depth]
    Zc = cam_points[2, valid_depth]

    # 防止除零错误
    Zc[Zc == 0] = 1e-6

    # 计算归一化的图像坐标
    x = Xc / Zc
    y = Yc / Zc

    # 像素坐标
    image_points = np.vstack((x, y, np.ones_like(x)))
    pixel_coords = intrinsics @ image_points
    u = pixel_coords[0, :] / pixel_coords[2, :]
    v = pixel_coords[1, :] / pixel_coords[2, :]

    # 过滤有效的像素坐标
    valid_mask = (u >= 0) & (u < img_width) & (v >= 0) & (v < img_height)
    if not np.any(valid_mask):
        print(f"No valid projected points in {image_path}")
        return False

    # 可视化投影到图像的结果
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