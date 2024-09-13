import carla
import numpy as np
import open3d as o3d
from PIL import Image
import yaml
import os
import time


def depth_to_point_cloud(depth_map):
    fx = 2892.33
    fy = 2883.18
    cx = 823.205
    cy = 619.071

    h, w = depth_map.shape
    points = []
    for v in range(h):
        for u in range(w):
            Z = depth_map[v, u]
            X = (u - cx) * Z / fx
            Y = (v - cy) * Z / fy
            points.append([X, Y, Z])
    return np.array(points)

class UAV:
    def __init__(self, world, location, uav_id, yaw_angle=0):
        self.world = world
        self.location = location

        self.yaw_angle = yaw_angle
        self.uav_id = uav_id  # 添加UAV的唯一ID
        self.rootDir = fr'C:\Users\uncle\_Projects\Carla\CARLA_Latest\WindowsNoEditor\myDemo\dataset\{self.uav_id}'

        self.static_actor = None
        self.sensors = []

        self.sensors_data_counter = 0  # 计数器
        self.total_sensors = 5  # 假设一共有5个传感器
        # self.frame_data = {}  # 保存每一帧的数据

        self.world_origin = [0, 0, 0]  # 世界坐标系的原点
        self.direction_x = [1, 0, 0]  # 世界坐标系的X轴方向向量

        self.ticks_per_capture = 5  # 每多少tick采集一次数据
        self.sensors_capture_intervals = self.ticks_per_capture * world.get_settings().fixed_delta_seconds  # 传感器的采集
        self.tick_counter = 0  # 初始化tick计数器

        self.move_enabled = False  # 移动开关，默认关闭
        self.delta_location = carla.Location(0, 0, 0)  # 默认的位移向量
        self.noise_std = 0  # 随机扰动的标准差

        self.rgb_sensors_active = [True, True, True, True, True]
        # self.rgb_sensors_active = [False, False, False, True, True]

        self.spawn_uav()

    def spawn_uav(self):
        image_size_x = 800
        image_size_y = 450
        pitch_degree = -45
        fov = 90
        capture_intervals = self.sensors_capture_intervals  # 无人机的移动频率应该和传感器的采集频率一致

        directions = ["North", "East", "South", "West"]
        yaw_angles = [0, 90, 180, 270]

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
        if self.rgb_sensors_active[4]:
            rgb_sensor.listen(lambda data: self.process_image(data, "down", "rgb"))
        self.sensors.append(rgb_sensor)

        # 创建四个不同方向的传感器
        for direction, yaw in zip(directions, yaw_angles):
            adjusted_yaw = yaw - self.yaw_angle  # 根据无人机的朝向调整传感器的朝向

            rgb_blueprint = self.world.get_blueprint_library().find('sensor.camera.rgb')
            rgb_blueprint.set_attribute('image_size_x', str(image_size_x))
            rgb_blueprint.set_attribute('image_size_y', str(image_size_y))
            rgb_blueprint.set_attribute('fov', str(fov))
            rgb_blueprint.set_attribute('sensor_tick', str(capture_intervals))

            rgb_transform = carla.Transform(carla.Location(x=0, y=0, z=-1), carla.Rotation(yaw=adjusted_yaw, pitch=pitch_degree))
            rgb_sensor = self.world.spawn_actor(rgb_blueprint, rgb_transform, self.static_actor)
            if self.rgb_sensors_active[yaw // 90]:
                rgb_sensor.listen(lambda data, dir=direction: self.process_image(data, dir, "rgb"))
            self.sensors.append(rgb_sensor)

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
    

    def euler_to_rotation_matrix(self, roll, pitch, yaw):
        # 将欧拉角转换为弧度
        roll = np.radians(roll)
        pitch = np.radians(pitch)
        yaw = np.radians(yaw)

        # X轴旋转矩阵
        Rx = np.array([
            [1, 0, 0],
            [0, np.cos(roll), -np.sin(roll)],
            [0, np.sin(roll), np.cos(roll)]
        ])

        # Y轴旋转矩阵
        Ry = np.array([
            [np.cos(pitch), 0, np.sin(pitch)],
            [0, 1, 0],
            [-np.sin(pitch), 0, np.cos(pitch)]
        ])

        # Z轴旋转矩阵
        Rz = np.array([
            [np.cos(yaw), -np.sin(yaw), 0],
            [np.sin(yaw), np.cos(yaw), 0],
            [0, 0, 1]
        ])

        # 最终旋转矩阵
        return np.dot(Rz, np.dot(Ry, Rx))

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
        
        # 假设Z轴方向向量固定为 [0, 0, 1] (垂直向上)
        direction_z = np.array([0, 0, 1])
        
        # 通过X轴和Z轴计算Y轴方向向量
        direction_y = np.cross(direction_z, direction_x)
        
        # 归一化所有方向向量
        direction_x = direction_x / np.linalg.norm(direction_x)
        direction_y = direction_y / np.linalg.norm(direction_y)
        direction_z = direction_z / np.linalg.norm(direction_z)

        # 获取传感器的位姿并转换为旋转矩阵
        sensor_transform = sensor.get_transform()
        roll = sensor_transform.rotation.roll
        pitch = sensor_transform.rotation.pitch
        yaw = sensor_transform.rotation.yaw
        sensor_rotation_matrix = self.euler_to_rotation_matrix(roll, pitch, yaw)

        # 获取传感器的平移向量
        sensor_translation = np.array([sensor_transform.location.x, sensor_transform.location.y, sensor_transform.location.z])

        # 构建传感器在车辆坐标系中的外参矩阵
        sensor_extrinsics = np.hstack((sensor_rotation_matrix, sensor_translation.reshape(3, 1)))
        sensor_extrinsics = np.vstack((sensor_extrinsics, [0, 0, 0, 1]))

        # 构建世界坐标系的旋转矩阵
        rotation_matrix = np.array([direction_x, direction_y, direction_z]).T

        # 构建世界坐标系的变换矩阵
        translation = np.array(world_origin).reshape(3, 1)
        world_extrinsics = np.hstack((rotation_matrix, translation))
        world_extrinsics = np.vstack((world_extrinsics, [0, 0, 0, 1]))

        # 通过矩阵乘法得到传感器相对于世界坐标系的外参
        world_sensor_extrinsics = np.dot(world_extrinsics, sensor_extrinsics)

        return world_sensor_extrinsics


    def process_image(self, image, direction, sensor_type):
        file_name = self.rootDir + r'\rgb_%s_%06d.png' % (direction, image.frame)
        image.save_to_disk(file_name)
        # self.frame_data[f'{sensor_type}_{direction}'] = image.frame
        self.sensors_data_counter += 1
        self.check_and_save_yaml(image.frame)

    def process_depth_image(self, image, direction, sensor_type):
        file_name = self.rootDir + r'\depth_%s_%06d.png' % (direction, image.frame)
        image.convert(carla.ColorConverter.LogarithmicDepth)
        image.save_to_disk(file_name)
        depth_map = Image.open(file_name).convert("L")
        depth_map = np.array(depth_map)
        points = depth_to_point_cloud(depth_map)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd_file_name = self.rootDir + r'dot_%s_%06d.pcd' % (direction, image.frame)
        o3d.io.write_point_cloud(pcd_file_name, pcd)
        # self.frame_data[f'{sensor_type}_{direction}'] = image.frame
        # self.sensors_data_counter += 1
        # self.check_and_save_yaml(image.frame)
    
    def calculate_world_coordinates(self, sensor_transform, world_origin, direction_x):
        # 定义世界坐标系的Z轴为垂直向上
        direction_z = np.array([0, 0, 1])
        
        # 通过X轴和Z轴叉乘计算Y轴方向
        direction_y = np.cross(direction_z, direction_x)
        
        # 归一化所有方向向量
        direction_x = direction_x / np.linalg.norm(direction_x)
        direction_y = direction_y / np.linalg.norm(direction_y)
        direction_z = direction_z / np.linalg.norm(direction_z)

        # 构建世界坐标系的旋转矩阵
        world_rotation_matrix = np.array([direction_x, direction_y, direction_z]).T

        # 相机的局部坐标
        local_translation = np.array([sensor_transform.location.x, 
                                    sensor_transform.location.y, 
                                    sensor_transform.location.z])
        
        # 将局部坐标转换为世界坐标
        world_translation = np.dot(world_rotation_matrix, local_translation) + world_origin

        # 提取相机旋转信息
        roll = sensor_transform.rotation.roll
        pitch = sensor_transform.rotation.pitch
        yaw = sensor_transform.rotation.yaw

        world_translation = np.array(world_translation, dtype=float)
        
        # 返回转换后的坐标
        cords = [
            float(world_translation[0]),  # x
            float(world_translation[1]),  # y
            float(world_translation[2]),  # z
            float(roll),  # roll
            float(yaw),   # yaw
            float(pitch)  # pitch
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
                sensor_transform = sensor.get_transform()
                cords = self.calculate_world_coordinates(sensor_transform, self.world_origin, self.direction_x)

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
            # self.frame_data.clear()


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
        for sensor in self.sensors:
            sensor.destroy()
        if self.static_actor is not None:
            self.static_actor.destroy()
