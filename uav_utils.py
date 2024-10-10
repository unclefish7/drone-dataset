import carla
import numpy as np
import open3d as o3d
import yaml
import os
import time
import math
from scipy.spatial.transform import Rotation as R
from queue import Queue
from queue import Empty

class RGBData:
    def __init__(self, sensor_id, data):
        self.sensor_id = sensor_id  # 传感器编号
        self.data = data              # RGB数据

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
        self.total_sensors = 0

        self.rgb_data_list = []  # 存储 RGB 数据的列表
        self.lidar_data = None

        self.sensor_queue = Queue()  # 传感器数据队列

        # 传感器采集间隔设置
        self.sensors_capture_intervals = 0.1  # 传感器采集间隔（秒）
        self.ticks_per_capture = self.sensors_capture_intervals / world.get_settings().fixed_delta_seconds
        self.tick_counter = 0  # tick 计数器

        # 计数器，所有传感器都存好数据以后才进行下一个tick
        self.if_saved = False

        # 移动设置
        self.move_enabled = False               # 移动开关，默认关闭
        self.delta_location = carla.Location(0, 0, 0)  # 移动的位移向量
        self.noise_std = 0                      # 随机扰动的标准差

        # 传感器激活标志
        self.rgb_sensors_active = True  # 是否激活 RGB 相机传感器
        self.dot_sensors_active = True  # 是否激活激光雷达传感器

        if self.rgb_sensors_active:
            self.total_sensors += 5

        if self.dot_sensors_active:
            self.total_sensors += 1

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
        directions = ["camera1", "camera2", "camera3", "camera4"]
        yaw_angles = [0, 90, 180, 270]
        sensor_offset = [
            [1, 0, -1],   # Front: 沿 x 正方向偏移
            [0, 1, -1],   # Right: 沿 y 正方向偏移
            [-1, 0, -1],  # Back: 沿 x 负方向偏移
            [0, -1, -1]   # Left: 沿 y 负方向偏移
        ]

        # 生成静态演员，表示无人机的位置
        static_blueprint = self.world.get_blueprint_library().find('static.prop.box01')
        spawn_point = carla.Transform(self.location, carla.Rotation(yaw=self.yaw_angle))
        self.static_actor = self.world.spawn_actor(static_blueprint, spawn_point)

        # 创建激光雷达传感器
        lidar_blueprint = self.world.get_blueprint_library().find('sensor.lidar.ray_cast')
        lidar_blueprint.set_attribute("channels", '128')
        lidar_blueprint.set_attribute('range', '100.0')
        lidar_blueprint.set_attribute('rotation_frequency', '10.0')
        lidar_blueprint.set_attribute('horizontal_fov', '150.0')
        lidar_blueprint.set_attribute('upper_fov', '50.0')
        lidar_blueprint.set_attribute('lower_fov', '-60.0')
        lidar_blueprint.set_attribute('points_per_second', '1000000')
        lidar_blueprint.set_attribute('sensor_tick', str(capture_intervals))

        # 设置激光雷达的变换
        lidar_transform = carla.Transform(carla.Location(x=0, y=0, z=-1), carla.Rotation(pitch=-60))
        lidar_sensor = self.world.spawn_actor(lidar_blueprint, lidar_transform, self.static_actor)

        # 创建垂直向下的 RGB 相机
        rgb_blueprint = self.world.get_blueprint_library().find('sensor.camera.rgb')
        rgb_blueprint.set_attribute('image_size_x', str(image_size_x))
        rgb_blueprint.set_attribute('image_size_y', str(image_size_y))
        rgb_blueprint.set_attribute('fov', str(fov))
        rgb_blueprint.set_attribute('sensor_tick', str(capture_intervals))

        # 设置相机的变换（位置和旋转）
        rgb_transform = carla.Transform(carla.Location(x=0, y=0, z=-1), carla.Rotation(pitch=-90))
        rgb_sensor = self.world.spawn_actor(rgb_blueprint, rgb_transform, self.static_actor)

        # 监听传感器数据
        if self.dot_sensors_active:
            lidar_sensor.listen(lambda data: self.process_lidar(data))

        if self.rgb_sensors_active:
            rgb_sensor.listen(lambda data: self.process_rgb("camera0", data))
        
        self.lidar_sensor = lidar_sensor
        self.sensors.append(rgb_sensor)

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
                rgb_sensor.listen(lambda data, dir=direction: self.process_rgb(dir, data))
            self.sensors.append(rgb_sensor)

    def process_lidar(self, data):
        # 暂存LiDAR数据
        self.sensor_queue.put((data, "lidar"))
        self.sensors_data_counter += 1
        # if self.sensors_data_counter == self.total_sensors:
        #     self.sensors_data_counter = 0
        #     self.write_all_data()

    def process_rgb(self, sensor_id, data):
        # 暂存RGB数据
        self.sensor_queue.put((data, sensor_id))
        self.sensors_data_counter += 1
        # if self.sensors_data_counter == self.total_sensors:
        #     self.sensors_data_counter = 0
        #     self.write_all_data()

    def process_image(self, image, direction):
        """
        处理并保存传感器的图像数据。

        参数：
        - image：传感器返回的图像数据。
        - direction：图像的方向标签。
        """
        # 生成文件名并保存图像
        if image != None:
            file_name = os.path.join(self.rootDir, f'{image.frame}_{direction}.png')
            image.save_to_disk(file_name)

    def process_dot_image(self, image):
        """
        处理并保存激光雷达的点云数据。

        参数：
        - image：传感器返回的点云数据。
        """
        # 将原始数据转换为 numpy 数组
        if image != None:
            data = np.copy(np.frombuffer(image.raw_data, dtype=np.dtype('f4')))
            data = np.reshape(data, (int(data.shape[0] / 4), 4))

            # 提取 XYZ 坐标
            points = data[:, :3]
            # 翻转 y 轴以匹配坐标系
            points[:, 1] = -points[:, 1]

            # 创建点云并保存为 PCD 文件
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd_file_name = os.path.join(self.rootDir, f'{image.frame + 1}.pcd')
            o3d.io.write_point_cloud(pcd_file_name, pcd)

    def get_intrinsics(self):
        """
        计算传感器的内参矩阵。

        参数：
        - sensor：传感器对象。

        返回：
        - intrinsics：3x3 的 numpy 数组，表示内参矩阵。
        """
        sensor = self.sensors[0]

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

    def get_sensor_extrinsics_and_pose(self, data):
        """
        计算传感器相对于给定坐标系的外参矩阵和位姿。

        参数：
        - T_sensor_to_world：传感器相对于世界坐标系的变换矩阵。

        返回：
        - extrinsics：4x4 的 numpy 数组，表示外参矩阵。
        - pose：长度为 6 的 numpy 数组，包含 [x, y, z, roll, pitch, yaw]，角度以度为单位。
        """

        sensor_transform = data.transform

        # 计算传感器相对于给定坐标系的变换矩阵
        extrinsics = np.array(sensor_transform.get_matrix())
        # 计算位姿
        x = sensor_transform.location.x
        y = sensor_transform.location.y
        z = sensor_transform.location.z

        # 提取旋转（以度为单位）
        roll = sensor_transform.rotation.roll
        yaw = sensor_transform.rotation.yaw
        pitch = sensor_transform.rotation.pitch

        pose = np.array([x, y, z, roll, yaw, pitch])
        
        return extrinsics, pose
    
    def get_lidar_pose(self, data):
        sensor_transform = data.transform

        x = sensor_transform.location.x
        y = sensor_transform.location.y
        z = sensor_transform.location.z

        roll = sensor_transform.rotation.roll
        yaw = sensor_transform.rotation.yaw
        pitch = sensor_transform.rotation.pitch

        pose = np.array([x, y, z, roll, yaw, pitch])

        return pose

    def check_and_save_all(self):
        """
        保存参数到 YAML 文件。
        """
        # 创建一个字典用于存储所有相机参数
        camera_params = {}

        yaml_file = None

        if self.sensors_data_counter < self.total_sensors:
            return

        try:
            for _ in range(6):
                data = self.sensor_queue.get(True, 1.0)

                if data is None:
                    continue
                
                if data[1] == "lidar":
                    # 生成 YAML 文件的路径
                    yaml_file = os.path.join(self.rootDir, f'{data[0].frame + 1}.yaml')

                    lidar_pose = self.get_lidar_pose(data[0])

                    camera_params['lidar_pose'] = lidar_pose.tolist()

                    self.process_dot_image(data[0])

                if data[1] != "lidar":
                    camera_id = data[1]

                    # 获取外参和位姿
                    extrinsics, pose = self.get_sensor_extrinsics_and_pose(data[0])
                    intrinsics = self.get_intrinsics()

                    # 将相机的参数添加到字典中
                    camera_params[camera_id] = {
                        'cords': pose.tolist(),
                        'extrinsic': extrinsics.tolist(),
                        'intrinsic': intrinsics.tolist()
                    }

                    # 保存相机图像
                    self.process_image(data[0], camera_id)
                
            # 将所有相机的参数写入一个 YAML 文件
            if yaml_file is not None:
                self.save_camera_params_to_yaml(camera_params, yaml_file)
                self.sensors_data_counter = 0
        except Empty:
            print("Queue is empty")

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

        每次 tick 会统一处理并写入所有传感器的数据。
        """
        self.check_and_save_all()


        if self.move_enabled:
            self.tick_counter += 1
            if self.tick_counter >= self.ticks_per_capture:
                self.tick_counter = 0
                self.move()

    # 设置各种参数的函数
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
