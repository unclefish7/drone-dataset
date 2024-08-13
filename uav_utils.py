import carla
import numpy as np
import open3d as o3d
from PIL import Image

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
    def __init__(self, world, location, yaw_angle=0):
        self.world = world
        self.location = location
        self.yaw_angle = yaw_angle
        self.static_actor = None
        self.sensors = []
        self.ticks_per_move = 20  # 默认每20个tick移动一次
        self.tick_counter = 0  # 初始化tick计数器
        self.move_enabled = False  # 移动开关，默认关闭
        self.delta_location = carla.Location(0, 0, 0)  # 默认的位移向量
        self.noise_std = 0  # 随机扰动的标准差
        self.spawn_uav()

    def spawn_uav(self):
        image_size_x = 800
        image_size_y = 450
        pitch_degree = -45
        fov = 90
        capture_intervals = 5.0

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
        rgb_sensor.listen(lambda data: self.process_image(data, "down", "rgb"))
        self.sensors.append(rgb_sensor)

        depth_blueprint = self.world.get_blueprint_library().find('sensor.camera.depth')
        depth_blueprint.set_attribute('image_size_x', str(image_size_x))
        depth_blueprint.set_attribute('image_size_y', str(image_size_y))
        depth_blueprint.set_attribute('fov', str(fov))
        depth_blueprint.set_attribute('sensor_tick', str(capture_intervals))

        depth_transform = carla.Transform(carla.Location(x=0, y=0, z=-1), carla.Rotation(pitch=-90))
        depth_sensor = self.world.spawn_actor(depth_blueprint, depth_transform, self.static_actor)
        # depth_sensor.listen(lambda data: self.process_depth_image(data, "down", "depth"))
        self.sensors.append(depth_sensor)

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
            # rgb_sensor.listen(lambda data, dir=direction: self.process_image(data, dir, "rgb"))
            self.sensors.append(rgb_sensor)

            depth_blueprint = self.world.get_blueprint_library().find('sensor.camera.depth')
            depth_blueprint.set_attribute('image_size_x', str(image_size_x))
            depth_blueprint.set_attribute('image_size_y', str(image_size_y))
            depth_blueprint.set_attribute('fov', str(fov))
            depth_blueprint.set_attribute('sensor_tick', str(capture_intervals))

            depth_transform = carla.Transform(carla.Location(x=0, y=0, z=-1), carla.Rotation(yaw=adjusted_yaw, pitch=pitch_degree))
            depth_sensor = self.world.spawn_actor(depth_blueprint, depth_transform, self.static_actor)
            # depth_sensor.listen(lambda data, dir=direction: self.process_depth_image(data, dir, "depth"))
            self.sensors.append(depth_sensor)

    def process_image(self, image, direction, sensor_type):
        image.save_to_disk(r'D:\CARLA_Latest\WindowsNoEditor\myDemo\_rgb_out\rgb_%s_%s_%06d.png' % (direction, sensor_type, image.frame))

    def process_depth_image(self, image, direction, sensor_type):
        image.convert(carla.ColorConverter.LogarithmicDepth)
        image.save_to_disk(r'D:\CARLA_Latest\WindowsNoEditor\myDemo\_depth_out\depth_%s_%s_%06d.png' % (direction, sensor_type, image.frame))
        depth_map = Image.open(r'D:\CARLA_Latest\WindowsNoEditor\myDemo\_depth_out\depth_%s_%s_%06d.png' % (direction, sensor_type, image.frame)).convert("L")
        depth_map = np.array(depth_map)
        points = depth_to_point_cloud(depth_map)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        o3d.io.write_point_cloud(r'D:\CARLA_Latest\WindowsNoEditor\myDemo\_dot_out\dot_%s_%s_%06d.pcd' % (direction, sensor_type, image.frame), pcd)

    def move(self):
        """
        控制无人机移动，并添加随机扰动。
        只有在移动开关打开时才会移动，每隔指定的tick数进行一次移动。
        """
        # 生成独立的随机扰动
        noise = np.random.normal(0, self.noise_std, size=3)

        # 将随机扰动添加到预定位移向量
        delta_location_with_noise = carla.Location(
            x=self.delta_location.x + noise[0],
            y=self.delta_location.y + noise[1],
            z=self.delta_location.z + noise[2]
        )

        # 计算新的位置
        new_location = self.static_actor.get_location() + delta_location_with_noise

        # 更新UAV的位置
        self.static_actor.set_location(new_location)

    def update(self):
        """
        每次tick调用该方法来检查是否需要移动无人机。
        """
        if self.move_enabled:
            self.tick_counter += 1
            if self.tick_counter >= self.ticks_per_move:
                # 重置tick计数器并移动
                self.tick_counter = 0
                self.move()

    def enable_movement(self, enabled=True):
        """
        控制移动开关。
        :param enabled: True 打开移动，False 关闭移动
        """
        self.move_enabled = enabled

    def set_ticks_per_move(self, ticks):
        """
        设置每次移动的tick数间隔。
        :param ticks: 每隔多少tick移动一次
        """
        self.ticks_per_move = ticks

    def set_delta_location(self, delta_location):
        """
        设置每次移动的位移向量。
        :param delta_location: 需要设置的位移向量
        """
        self.delta_location = delta_location

    def set_noise_std(self, noise_std):
        """
        设置随机扰动的标准差。
        :param noise_std: 随机扰动的标准差
        """
        self.noise_std = noise_std
        
    def destroy(self):
        for sensor in self.sensors:
            sensor.destroy()
        if self.static_actor is not None:
            self.static_actor.destroy()
