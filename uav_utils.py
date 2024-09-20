import carla
import numpy as np
import open3d as o3d
import yaml
import os
import time
import math
from scipy.spatial.transform import Rotation as R

def transform_to_matrix(transform):
    """
    将 CARLA 的 Transform 对象转换为 4x4 的变换矩阵。

    参数：
    - transform：carla.Transform 对象。

    返回：
    - transform_matrix：4x4 的 numpy 数组，表示变换矩阵。
    """
    # 获取位置信息
    location = transform.location
    x, y, z = location.x, -location.y, location.z  # 关于 y 的符号后面讨论

    # 获取旋转角度并转换为弧度
    roll = math.radians(transform.rotation.roll)    # Roll 对应于绕 X 轴旋转
    pitch = math.radians(transform.rotation.pitch)  # Pitch 对应于绕 Y 轴旋转
    yaw = math.radians(transform.rotation.yaw)      # Yaw 对应于绕 Z 轴旋转

    # 计算绕 X 轴的旋转矩阵（Roll）
    Rx = np.array([
        [1, 0, 0],
        [0, math.cos(roll), -math.sin(roll)],
        [0, math.sin(roll), math.cos(roll)]
    ])

    # 计算绕 Y 轴的旋转矩阵（Pitch）
    Ry = np.array([
        [math.cos(pitch), 0, math.sin(pitch)],
        [0, 1, 0],
        [-math.sin(pitch), 0, math.cos(pitch)]
    ])

    # 计算绕 Z 轴的旋转矩阵（Yaw）
    Rz = np.array([
        [math.cos(yaw), -math.sin(yaw), 0],
        [math.sin(yaw), math.cos(yaw), 0],
        [0, 0, 1]
    ])

    # 组合旋转矩阵（注意乘法顺序）
    rotation_matrix = np.dot(Rz, np.dot(Ry, Rx))

    # 构造 4x4 的变换矩阵
    transform_matrix = np.identity(4)
    transform_matrix[:3, :3] = rotation_matrix
    transform_matrix[:3, 3] = [x, y, z]

    return transform_matrix

def compute_matrix_world_to_given(world_origin, direction_x, direction_z=np.array([0, 0, 1])):
    """
    计算从世界坐标系到给定坐标系的变换矩阵。

    参数：
    - world_origin：给定坐标系的原点在世界坐标系中的位置（numpy 数组）。
    - direction_x：给定坐标系的 x 轴方向向量（numpy 数组）。
    - direction_z：给定坐标系的 z 轴方向向量，默认为 [0, 0, 1]。

    返回：
    - T_world_to_given：4x4 的 numpy 数组，从世界坐标系到给定坐标系的变换矩阵。
    """
    # 归一化方向向量
    direction_x = direction_x / np.linalg.norm(direction_x)
    direction_z = direction_z / np.linalg.norm(direction_z)

    # 根据右手定则计算 y 轴方向
    direction_y = np.cross(direction_z, direction_x)
    direction_y = direction_y / np.linalg.norm(direction_y)

    # 构造旋转矩阵
    rotation_matrix = np.column_stack((direction_x, direction_y, direction_z))

    # 构造从给定坐标系到世界坐标系的变换矩阵（T_given_to_world）
    T_given_to_world = np.identity(4)
    T_given_to_world[:3, :3] = rotation_matrix
    T_given_to_world[:3, 3] = world_origin

    # 计算 T_world_to_given，即 T_given_to_world 的逆矩阵
    T_world_to_given = np.linalg.inv(T_given_to_world)

    return T_world_to_given

class UAV:
    def __init__(self, world, location, uav_id, yaw_angle=0):
        """
        初始化 UAV 类。

        参数：
        - world：CARLA 的 world 对象。
        - location：无人机的初始位置（carla.Location 对象）。
        - uav_id：无人机的唯一 ID。
        - yaw_angle：无人机的初始偏航角。
        """
        self.world = world
        self.location = location
        self.yaw_angle = yaw_angle
        self.uav_id = uav_id  # 无人机的唯一 ID

        # 数据保存的根目录
        self.rootDir = fr'C:\Users\uncle\_Projects\Carla\CARLA_Latest\WindowsNoEditor\myDemo\dataset\{self.uav_id}'

        self.static_actor = None  # 静态演员，表示无人机的位置
        self.sensors = []         # 存储所有传感器的列表
        self.lidar_sensor = None  # 激光雷达传感器

        self.sensors_data_counter = 0  # 已接收到的传感器数据计数
        self.total_sensors = 6         # 总的传感器数量

        self.world_origin = [0, 0, 0]  # 给定坐标系的原点
        self.direction_x = [1, 0, 0]   # 给定坐标系的 x 轴方向向量

        # 传感器采集间隔设置
        self.ticks_per_capture = 5  # 每隔多少个 tick 采集一次数据
        self.sensors_capture_intervals = self.ticks_per_capture * world.get_settings().fixed_delta_seconds
        self.tick_counter = 0  # tick 计数器

        # 移动设置
        self.move_enabled = False               # 移动开关，默认关闭
        self.delta_location = carla.Location(0, 0, 0)  # 移动的位移向量
        self.noise_std = 0                      # 随机扰动的标准差

        # 传感器激活标志
        self.rgb_sensors_active = True  # 是否激活 RGB 相机传感器
        self.dot_sensors_active = True  # 是否激活激光雷达传感器

        # 生成无人机并附加传感器
        self.spawn_uav()

    def spawn_uav(self):
        """
        生成无人机并附加传感器。
        """
        # 传感器参数设置
        image_size_x = 800
        image_size_y = 450
        pitch_degree = -45  # 相机俯仰角
        fov = 90            # 视场角
        capture_intervals = self.sensors_capture_intervals  # 采集间隔

        # 不同方向的设置
        directions = ["East", "South", "West", "North"]
        yaw_angles = [0, 90, 180, 270]
        sensor_offset = [
            [1, 0, -1],   # East: 沿 x 正方向偏移
            [0, 1, -1],   # South: 沿 y 正方向偏移
            [-1, 0, -1],  # West: 沿 x 负方向偏移
            [0, -1, -1]   # North: 沿 y 负方向偏移
        ]

        # 生成静态演员，表示无人机的位置
        static_blueprint = self.world.get_blueprint_library().find('static.prop.box01')
        spawn_point = carla.Transform(self.location, carla.Rotation(yaw=self.yaw_angle))
        self.static_actor = self.world.spawn_actor(static_blueprint, spawn_point)

        # 创建垂直向下的 RGB 相机
        rgb_blueprint = self.world.get_blueprint_library().find('sensor.camera.rgb')
        rgb_blueprint.set_attribute('image_size_x', str(image_size_x))
        rgb_blueprint.set_attribute('image_size_y', str(image_size_y))
        rgb_blueprint.set_attribute('fov', str(fov))
        rgb_blueprint.set_attribute('sensor_tick', str(capture_intervals))

        # 设置相机的变换（位置和旋转）
        rgb_transform = carla.Transform(carla.Location(x=0, y=0, z=-1), carla.Rotation(pitch=-90))
        rgb_sensor = self.world.spawn_actor(rgb_blueprint, rgb_transform, self.static_actor)
        if self.rgb_sensors_active:
            rgb_sensor.listen(lambda data: self.process_image(data, "down", "rgb"))
        self.sensors.append(rgb_sensor)

        # 创建激光雷达传感器
        lidar_blueprint = self.world.get_blueprint_library().find('sensor.lidar.ray_cast')
        lidar_blueprint.set_attribute("channels", '128')
        lidar_blueprint.set_attribute('range', '200.0')
        lidar_blueprint.set_attribute('rotation_frequency', '10.0')
        lidar_blueprint.set_attribute('horizontal_fov', '360.0')
        lidar_blueprint.set_attribute('upper_fov', '0.0')
        lidar_blueprint.set_attribute('lower_fov', '-90.0')
        lidar_blueprint.set_attribute('points_per_second', '1000000')
        lidar_blueprint.set_attribute('sensor_tick', str(capture_intervals))

        # 设置激光雷达的变换
        lidar_transform = carla.Transform(carla.Location(x=0, y=0, z=-1), carla.Rotation(pitch=0))
        lidar_sensor = self.world.spawn_actor(lidar_blueprint, lidar_transform, self.static_actor)
        if self.dot_sensors_active:
            lidar_sensor.listen(lambda data: self.process_dot_image(data, "down", "dot"))
        # 存储激光雷达传感器
        self.lidar_sensor = lidar_sensor

        # 创建四个不同方向的 RGB 相机
        for direction, yaw, offset in zip(directions, yaw_angles, sensor_offset):
            adjusted_yaw = yaw - self.yaw_angle  # 根据无人机的朝向调整传感器的朝向

            rgb_blueprint = self.world.get_blueprint_library().find('sensor.camera.rgb')
            rgb_blueprint.set_attribute('image_size_x', str(image_size_x))
            rgb_blueprint.set_attribute('image_size_y', str(image_size_y))
            rgb_blueprint.set_attribute('fov', str(fov))
            rgb_blueprint.set_attribute('sensor_tick', str(capture_intervals))

            rgb_transform = carla.Transform(
                carla.Location(x=offset[0], y=offset[1], z=offset[2]),
                carla.Rotation(yaw=adjusted_yaw, pitch=pitch_degree)
            )
            rgb_sensor = self.world.spawn_actor(rgb_blueprint, rgb_transform, self.static_actor)
            if self.rgb_sensors_active:
                rgb_sensor.listen(lambda data, dir=direction: self.process_image(data, dir, "rgb"))
            self.sensors.append(rgb_sensor)

    def process_image(self, image, direction, sensor_type):
        """
        处理并保存传感器的图像数据。

        参数：
        - image：传感器返回的图像数据。
        - direction：图像的方向标签。
        - sensor_type：传感器类型（"rgb"）。
        """
        # 生成文件名并保存图像
        file_name = os.path.join(self.rootDir, f'{sensor_type}_{direction}_{image.frame:06d}.png')
        image.save_to_disk(file_name)
        # 增加数据计数器并检查是否需要保存参数
        self.sensors_data_counter += 1
        self.check_and_save_yaml(image.frame)

    def process_dot_image(self, image, direction, sensor_type):
        """
        处理并保存激光雷达的点云数据。

        参数：
        - image：传感器返回的点云数据。
        - direction：数据的方向标签。
        - sensor_type：传感器类型（"dot"）。
        """
        # 将原始数据转换为 numpy 数组
        data = np.copy(np.frombuffer(image.raw_data, dtype=np.dtype('f4')))
        data = np.reshape(data, (int(data.shape[0] / 4), 4))

        # 提取 XYZ 坐标
        points = data[:, :3]
        # 翻转 y 轴以匹配坐标系
        points[:, 1] = -points[:, 1]

        # 创建点云并保存为 PCD 文件
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd_file_name = os.path.join(self.rootDir, f'{sensor_type}_{direction}_{image.frame:06d}.pcd')
        o3d.io.write_point_cloud(pcd_file_name, pcd)
        # 增加数据计数器并检查是否需要保存参数
        self.sensors_data_counter += 1
        self.check_and_save_yaml(image.frame)

    def get_intrinsics(self, sensor):
        """
        计算传感器的内参矩阵。

        参数：
        - sensor：传感器对象。

        返回：
        - intrinsics：3x3 的 numpy 数组，表示内参矩阵。
        """
        if not sensor.is_alive:
            print(f"Sensor {sensor.id} has been destroyed.")
            return None

        fov = float(sensor.attributes['fov'])
        image_width = int(sensor.attributes['image_size_x'])
        image_height = int(sensor.attributes['image_size_y'])

        # 计算焦距
        fx = image_width / (2.0 * np.tan(fov * np.pi / 360.0))
        fy = image_height / (2.0 * np.tan(fov * np.pi / 360.0))

        # 构造内参矩阵
        intrinsics = np.array([
            [fx, 0, image_width / 2],
            [0, fy, image_height / 2],
            [0, 0, 1]
        ])
        
        return intrinsics

    def get_sensor_extrinsics_and_pose(self, T_sensor_to_world, T_world_to_given):
        """
        计算传感器相对于给定坐标系的外参矩阵和位姿。

        参数：
        - T_sensor_to_world：传感器相对于世界坐标系的变换矩阵。
        - T_world_to_given：世界坐标系相对于给定坐标系的变换矩阵。

        返回：
        - extrinsics：4x4 的 numpy 数组，表示外参矩阵。
        - pose：长度为 6 的 numpy 数组，包含 [x, y, z, roll, pitch, yaw]，角度以度为单位。
        """
        # 计算传感器相对于给定坐标系的变换矩阵
        T_sensor_to_given = np.dot(T_world_to_given, T_sensor_to_world)
        
        # 提取平移向量
        translation_vector = T_sensor_to_given[:3, 3]
        x, y, z = translation_vector

        # 提取旋转矩阵
        rotation_matrix = T_sensor_to_given[:3, :3]
        
        # 将旋转矩阵转换为欧拉角（roll, pitch, yaw），以度为单位
        r = R.from_matrix(rotation_matrix)
        roll, pitch, yaw = r.as_euler('xyz', degrees=True)
        
        # 组合位姿数组
        pose = np.array([x, y, z, roll, pitch, yaw])
        
        # 外参矩阵就是 T_sensor_to_given
        extrinsics = T_sensor_to_given
        
        return extrinsics, pose

    def check_and_save_yaml(self, frame):
        """
        检查是否已接收到所有传感器的数据，并保存参数到 YAML 文件。

        参数：
        - frame：当前帧号。
        """
        if self.sensors_data_counter == self.total_sensors:
            # 创建一个字典用于存储所有相机参数
            camera_params = {}

            for idx, sensor in enumerate(self.sensors):

                if not sensor.is_alive:
                    print(f"Sensor {sensor.id} has been destroyed.")
                    return None
                
                camera_id = f'camera{idx}'  # 动态生成相机 ID

                # 计算变换矩阵
                T_sensor_to_world = transform_to_matrix(sensor.get_transform())
                T_world_to_given = compute_matrix_world_to_given(self.world_origin, self.direction_x)

                # 获取外参和位姿
                extrinsics, pose = self.get_sensor_extrinsics_and_pose(T_sensor_to_world, T_world_to_given)
                intrinsics = self.get_intrinsics(sensor)

                # 将相机的参数添加到字典中
                camera_params[camera_id] = {
                    'cords': pose.tolist(),
                    'extrinsic': extrinsics.tolist(),
                    'intrinsic': intrinsics.tolist()
                }
            
            # 生成 YAML 文件的路径
            yaml_file = os.path.join(self.rootDir, f'yaml_frame{frame:06d}.yaml')
            
            # 将所有相机的参数写入一个 YAML 文件
            self.save_camera_params_to_yaml(camera_params, yaml_file)

            # 保存完毕后重置计数器
            self.sensors_data_counter = 0

    def save_camera_params_to_yaml(self, camera_params, yaml_file):
        """
        将相机参数保存到 YAML 文件。

        参数：
        - camera_params：包含相机参数的字典。
        - yaml_file：YAML 文件的路径。
        """
        # 获取目录路径
        directory = os.path.dirname(yaml_file)

        # 如果目录不存在，则创建
        if not os.path.exists(directory):
            os.makedirs(directory)

        # 写入文件
        try:
            with open(yaml_file, 'w') as file:
                yaml.dump(camera_params, file, default_flow_style=False)
        except FileNotFoundError as e:
            print(f"Error: {e}")

    def move(self):
        """
        控制无人机移动，并添加随机扰动。
        只有在移动开关打开时才会移动。
        """
        # 生成移动的随机扰动
        noise = np.random.normal(0, self.noise_std, size=3)

        delta_location_with_noise = carla.Location(
            x=self.delta_location.x + noise[0],
            y=self.delta_location.y + noise[1],
            z=self.delta_location.z + noise[2]
        )

        # 更新无人机的位置
        new_location = self.static_actor.get_location() + delta_location_with_noise
        self.static_actor.set_location(new_location)

    def update(self):
        """
        每次 tick 调用该方法来检查是否需要移动无人机。
        """
        if self.move_enabled:
            self.tick_counter += 1
            if self.tick_counter >= self.ticks_per_capture:
                self.tick_counter = 0
                self.move()

    # 设置各种参数的函数
    def set_world_origin(self, origin):
        self.world_origin = origin

    def set_direction_x(self, direction_x):
        self.direction_x = direction_x

    def set_sensors_capture_intervals(self, intervals):
        self.sensors_capture_intervals = intervals

    def enable_movement(self, enabled=True):
        self.move_enabled = enabled

    def set_ticks_per_move(self, ticks):
        self.ticks_per_move = ticks

    def set_delta_location(self, delta_location):
        self.delta_location = delta_location

    def set_noise_std(self, noise_std):
        self.noise_std = noise_std

    def destroy(self):
        """
        清理资源，销毁传感器和静态演员。
        """
        time.sleep(1)

        # 销毁所有传感器
        for sensor in self.sensors:
            if sensor.is_alive:
                sensor.stop()
                sensor.destroy()

        # 销毁静态演员
        if self.static_actor is not None and self.static_actor.is_alive:
            self.static_actor.destroy()
