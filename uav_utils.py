import carla
import numpy as np
import open3d as o3d
from PIL import Image
import yaml
import os
import time
import math

def transform_to_matrix(transform):
    # 获取位置
    location = transform.location
    x, y, z = location.x, location.y, location.z
    
    # 获取旋转角度并转换为弧度
    pitch = math.radians(transform.rotation.pitch)
    yaw = math.radians(transform.rotation.yaw)
    roll = math.radians(transform.rotation.roll)

    # 计算旋转矩阵 (3x3)
    Rx = np.array([
        [1, 0, 0],
        [0, math.cos(pitch), -math.sin(pitch)],
        [0, math.sin(pitch), math.cos(pitch)]
    ])
    
    Ry = np.array([
        [math.cos(roll), 0, math.sin(roll)],
        [0, 1, 0],
        [-math.sin(roll), 0, math.cos(roll)]
    ])
    
    Rz = np.array([
        [math.cos(yaw), -math.sin(yaw), 0],
        [math.sin(yaw), math.cos(yaw), 0],
        [0, 0, 1]
    ])

    # 组合旋转矩阵
    rotation_matrix = np.dot(Rz, np.dot(Ry, Rx))

    # 构造 4x4 的变换矩阵
    transform_matrix = np.identity(4)
    transform_matrix[:3, :3] = rotation_matrix
    transform_matrix[:3, 3] = [x, y, z]

    return transform_matrix


class UAV:
    def __init__(self, world, location, uav_id, yaw_angle=0):
        self.world = world
        self.location = location

        self.yaw_angle = yaw_angle
        self.uav_id = uav_id  # 添加UAV的唯一ID
        self.rootDir = fr'C:\Users\uncle\_Projects\Carla\CARLA_Latest\WindowsNoEditor\myDemo\dataset\{self.uav_id}'

        self.static_actor = None
        self.sensors = []
        self.lidar_sensor = None

        self.sensors_data_counter = 0  # 计数器
        self.total_sensors = 6  # 假设一共有6个传感器

        self.world_origin = [0, 0, 0]  # 世界坐标系的原点
        self.direction_x = [1, 0, 0]  # 世界坐标系的X轴方向向量

        self.ticks_per_capture = 5  # 每多少tick采集一次数据
        self.sensors_capture_intervals = self.ticks_per_capture * world.get_settings().fixed_delta_seconds  # 传感器的采集
        self.tick_counter = 0  # 初始化tick计数器

        self.move_enabled = False  # 移动开关，默认关闭
        self.delta_location = carla.Location(0, 0, 0)  # 默认的位移向量
        self.noise_std = 0  # 随机扰动的标准差

        # self.rgb_sensors_active = [True, True, True, True, True]
        self.rgb_sensors_active = False

        self.dot_sensors_active = False

        self.spawn_uav()

    def spawn_uav(self):
        image_size_x = 800
        image_size_y = 450
        pitch_degree = -45
        fov = 90
        capture_intervals = self.sensors_capture_intervals  # 无人机的移动频率应该和传感器的采集频率一致

        directions = ["East", "South", "West", "North"]
        yaw_angles = [0, 90, 180, 270]
        sensor_offset = [
            [1, 0, -1],  # East: 沿x正方向偏移
            [0, 1, -1],  # South: 沿y正方向偏移
            [-1, 0, -1],  # West: 沿x负方向偏移
            [0, -1, -1]   # North: 沿y负方向偏移
        ]


        static_blueprint = self.world.get_blueprint_library().find('static.prop.box01')
        spawn_point = carla.Transform(self.location, carla.Rotation(yaw=self.yaw_angle))
        self.static_actor = self.world.spawn_actor(static_blueprint, spawn_point)

        # 创建垂直向下的传感器
        rgb_blueprint = self.world.get_blueprint_library().find('sensor.camera.rgb')
        rgb_blueprint.set_attribute('image_size_x', str(image_size_x))
        rgb_blueprint.set_attribute('image_size_y', str(image_size_y))
        rgb_blueprint.set_attribute('fov', str(fov))
        rgb_blueprint.set_attribute('sensor_tick', str(capture_intervals))

        rgb_transform = carla.Transform(carla.Location(x=0, y=0, z=-1), carla.Rotation(pitch=-90))
        rgb_sensor = self.world.spawn_actor(rgb_blueprint, rgb_transform, self.static_actor)
        if self.rgb_sensors_active:
            rgb_sensor.listen(lambda data: self.process_image(data, "down", "rgb"))
        self.sensors.append(rgb_sensor)

        lidar_blueprint = self.world.get_blueprint_library().find('sensor.lidar.ray_cast')
        lidar_blueprint.set_attribute("channels", '128.0')
        lidar_blueprint.set_attribute('range', '200.0')
        lidar_blueprint.set_attribute('rotation_frequency', '10.0')
        # lidar_blueprint.set_attribute('horizontal_fov', '180.0')
        lidar_blueprint.set_attribute('horizontal_fov', '360.0')

        lidar_blueprint.set_attribute('upper_fov','0.0')
        lidar_blueprint.set_attribute('lower_fov', '-90.0')
        lidar_blueprint.set_attribute('points_per_second', '1000000')
        lidar_blueprint.set_attribute('sensor_tick', str(capture_intervals))

        lidar_transform = carla.Transform(carla.Location(x=0, y=0, z=-1), carla.Rotation(pitch=0))  # 根据需要调整位置
        # lidar_transform = carla.Transform(carla.Location(x=0, y=0, z=-1), carla.Rotation(pitch=-90))  # 根据需要调整位置
        lidar_sensor = self.world.spawn_actor(lidar_blueprint, lidar_transform, self.static_actor)
        if self.dot_sensors_active:
            lidar_sensor.listen(lambda data: self.process_dot_image(data, "down", "dot"))
        # lidar_sensor.listen(lambda data: self.draw_lidar(self.display,self.process_lidar_data(data)))

        self.lidar_sensor = lidar_sensor

        # 创建四个不同方向的传感器
        for direction, yaw, offset in zip(directions, yaw_angles, sensor_offset):
            adjusted_yaw = yaw - self.yaw_angle  # 根据无人机的朝向调整传感器的朝向

            rgb_blueprint = self.world.get_blueprint_library().find('sensor.camera.rgb')
            rgb_blueprint.set_attribute('image_size_x', str(image_size_x))
            rgb_blueprint.set_attribute('image_size_y', str(image_size_y))
            rgb_blueprint.set_attribute('fov', str(fov))
            rgb_blueprint.set_attribute('sensor_tick', str(capture_intervals))

            rgb_transform = carla.Transform(carla.Location(x=offset[0], y=offset[1], z=offset[2]), carla.Rotation(yaw=adjusted_yaw, pitch=pitch_degree))
            rgb_sensor = self.world.spawn_actor(rgb_blueprint, rgb_transform, self.static_actor)
            if self.rgb_sensors_active:
                rgb_sensor.listen(lambda data, dir=direction: self.process_image(data, dir, "rgb"))
            self.sensors.append(rgb_sensor)

    def process_image(self, image, direction, sensor_type):
        file_name = self.rootDir + r'\rgb_%s_%06d.png' % (direction, image.frame)
        image.save_to_disk(file_name)
        self.sensors_data_counter += 1
        self.check_and_save_yaml(image.frame)

    def process_dot_image(self, image, direction, sensor_type):
        data = np.copy(np.frombuffer(image.raw_data, dtype=np.dtype('f4')))
        data = np.reshape(data, (int(data.shape[0] / 4), 4))

        # 翻转x轴
        # points = data[:, :-1]
        points = data[:, :3]
        points[:, 1] = -points[:, 1]
        # 保存为PCD格式
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd_file_name = self.rootDir + r'\dot_%s_%06d.pcd' % (direction, image.frame)
        o3d.io.write_point_cloud(pcd_file_name, pcd)
        self.sensors_data_counter += 1
        self.check_and_save_yaml(image.frame)

    def get_intrinsics(self, sensor):
        if not sensor.is_alive:
            print(f"Sensor {sensor.id} has been destroyed.")
            return None

        fov = float(sensor.attributes['fov'])  # 获取FOV
        image_width = int(sensor.attributes['image_size_x'])  # 图像宽度
        image_height = int(sensor.attributes['image_size_y'])  # 图像高度

        # 分别计算fx和fy
        fx = image_width / (2.0 * np.tan(fov * np.pi / 360.0))
        fy = image_height / (2.0 * np.tan(fov * np.pi / 360.0))

        intrinsics = np.array([
            [fx, 0, image_width / 2],
            [0, fy, image_height / 2],
            [0, 0, 1]
        ])
        
        return intrinsics
    

    def calculate_extrinsics_with_x_direction(self, sensor, world_origin, direction_x):
        """
        计算传感器相对于世界坐标系的外参，通过世界坐标系的原点和X轴方向向量生成旋转矩阵。
        
        :param sensor: 传感器对象
        :param world_origin: 世界坐标系的原点 [x, y, z]
        :param direction_x: 世界坐标系的X轴方向向量 [x, y, z]
        :return: 传感器相对于世界坐标系的外参矩阵
        """

        if not sensor.is_alive:
            print(f"Sensor {sensor.id} has been destroyed.")
            return None
        
        # 获取传感器相对于static_actor的相对变换
        sensor_transform = sensor.get_transform()
        actor_transform = self.static_actor.get_transform()

        # 计算传感器相对于世界坐标系的位置
        sensor_world_location = actor_transform.transform(sensor_transform.location)

        # 计算传感器相对于世界坐标系的旋转
        sensor_world_rotation = actor_transform.rotation
        sensor_world_rotation.roll += sensor_transform.rotation.roll
        sensor_world_rotation.pitch += sensor_transform.rotation.pitch
        sensor_world_rotation.yaw += sensor_transform.rotation.yaw

        # 定义给定坐标系的方向向量
        direction_x = np.array(direction_x)
        direction_x = direction_x / np.linalg.norm(direction_x)  # 归一化
        direction_z = np.array([0, 0, 1])  # 竖直向上
        direction_y = np.cross(direction_z, direction_x)  # 计算与 direction_x 和 direction_z 垂直的方向 y

        # 构造给定坐标系的旋转矩阵
        rotation_matrix = np.array([direction_x, direction_y, direction_z]).T

        # 构造给定坐标系的变换矩阵
        world_to_custom_transform = np.eye(4)
        world_to_custom_transform[:3, :3] = rotation_matrix  # 设置旋转部分
        world_to_custom_transform[:3, 3] = np.array(world_origin)  # 设置平移部分

        # 将传感器位置从世界坐标系转换到自定义坐标系
        sensor_location_world = np.array([sensor_world_location.x, sensor_world_location.y, sensor_world_location.z, 1])
        sensor_world_matrix = np.eye(4)
        sensor_world_matrix[:3, 3] = sensor_location_world[:3]
        
        # 计算外参矩阵
        sensor_extrinsics = np.linalg.inv(world_to_custom_transform) @ sensor_world_matrix

        return sensor_extrinsics
    
    def calculate_world_coordinates(self, sensor, world_origin, direction_x):

        if not sensor.is_alive:
            print(f"Sensor {sensor.id} has been destroyed.")
            return None
    
        # 获取传感器相对于static_actor的相对变换
        sensor_transform = sensor.get_transform()
        actor_transform = self.static_actor.get_transform()

        sensor_world_location = actor_transform.transform(sensor_transform.location)

        # 计算传感器相对于世界坐标系的旋转
        sensor_world_rotation = actor_transform.rotation
        sensor_world_rotation.roll += sensor_transform.rotation.roll
        sensor_world_rotation.pitch += sensor_transform.rotation.pitch
        sensor_world_rotation.yaw += sensor_transform.rotation.yaw

        # 定义给定坐标系的方向向量
        direction_x = np.array(direction_x)
        direction_x = direction_x / np.linalg.norm(direction_x)  # 归一化
        direction_z = np.array([0, 0, 1])  # 竖直向上
        direction_y = np.cross(direction_z, direction_x)  # 计算与 direction_x 和 direction_z 垂直的方向 y

        # 构造给定坐标系的旋转矩阵
        rotation_matrix = np.array([direction_x, direction_y, direction_z]).T

        # 构造给定坐标系的变换矩阵
        world_to_custom_transform = np.eye(4)
        world_to_custom_transform[:3, :3] = rotation_matrix  # 设置旋转部分
        world_to_custom_transform[:3, 3] = np.array(world_origin)  # 设置平移部分

        # 将传感器位置从世界坐标系转换到自定义坐标系
        sensor_location_world = np.array([sensor_world_location.x, sensor_world_location.y, sensor_world_location.z, 1])
        custom_location = np.linalg.inv(world_to_custom_transform) @ sensor_location_world

        # 按照指定格式返回转换后的位姿
        cords = [
            float(custom_location[0]),  # x
            float(custom_location[1]),  # y
            float(custom_location[2]),  # z
            float(sensor_world_rotation.roll),  # roll
            float(sensor_world_rotation.yaw),   # yaw
            float(sensor_world_rotation.pitch)  # pitch
        ]

        return cords

    def check_and_save_yaml(self, frame):
        if self.sensors_data_counter == self.total_sensors:
            # 创建一个用于存储所有相机参数的字典
            camera_params = {}

            for idx, sensor in enumerate(self.sensors):

                if not sensor.is_alive:
                    print(f"Sensor {sensor.id} has been destroyed.")
                    return None
                
                camera_id = f'camera{idx}'  # 动态生成 camera_id
                extrinsics = self.calculate_extrinsics_with_x_direction(sensor, self.world_origin, self.direction_x)
                intrinsics = self.get_intrinsics(sensor)

                # 获取相机位姿
                cords = self.calculate_world_coordinates(sensor, self.world_origin, self.direction_x)

                # 将相机的参数添加到字典中
                camera_params[camera_id] = {
                    'cords': cords,
                    'extrinsic': extrinsics.tolist(),
                    'intrinsic': intrinsics.tolist()
                }
            
            # 生成YAML文件的路径
            yaml_file = self.rootDir + r'\yaml_frame%06d.yaml' % (frame)
            
            # 将所有相机的参数写入一个YAML文件
            self.save_camera_params_to_yaml(camera_params, yaml_file)

            # 重置计数器和帧数据
            self.sensors_data_counter = 0


    def save_camera_params_to_yaml(self, camera_params, yaml_file):
        # 获取文件目录路径
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
        只有在移动开关打开时才会移动，每隔指定的tick数进行一次移动。
        """
        noise = np.random.normal(0, self.noise_std, size=3)

        delta_location_with_noise = carla.Location(
            x=self.delta_location.x + noise[0],
            y=self.delta_location.y + noise[1],
            z=self.delta_location.z + noise[2]
        )

        new_location = self.static_actor.get_location() + delta_location_with_noise

        self.static_actor.set_location(new_location)

    def update(self):
        """
        每次tick调用该方法来检查是否需要移动无人机。
        """
        if self.move_enabled:
            self.tick_counter += 1
            if self.tick_counter >= self.ticks_per_capture:
                self.tick_counter = 0
                self.move()

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
        time.sleep(1)

        # 销毁所有传感器
        for sensor in self.sensors:
            sensor.destroy()
        # 销毁静态演员
        if self.static_actor is not None:
            self.static_actor.destroy()