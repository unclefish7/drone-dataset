import carla
import numpy as np
import open3d as o3d
import pygame
from PIL import Image


def depth_to_point_cloud(depth_map):
    # h, w = depth_map.shape
    fx = 2892.33
    fy = 2883.18
    cx = 823.205
    cy = 619.071

    h, w = 450, 800
    points = []
    for v in range(h):
        for u in range(w):
            Z = depth_map[v, u]
            X = (u - cx) * Z / fx
            Y = (v - cy) * Z / fy
            points.append([X, Y, Z])
    return np.array(points)

class UAV:
    def __init__(self, world, location, yaw_angle):
        self.world = world
        self.location = location
        self.yaw_angle = yaw_angle
        self.static_actor = None
        self.sensors = []
        self.spawn_uav()
        # pygame.init()
        # self.display = pygame.display.set_mode((800, 600))
        # pygame.display.set_caption("LiDAR Visualization")

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
        depth_sensor.listen(lambda data: self.process_depth_image(data, "down", "depth"))

        self.sensors.append(depth_sensor)

        lidar_blueprint = self.world.get_blueprint_library().find('sensor.lidar.ray_cast')
        lidar_blueprint.set_attribute('range', '100.0')
        lidar_blueprint.set_attribute('rotation_frequency', '10.0')
        lidar_blueprint.set_attribute('points_per_second', '56000')
        lidar_blueprint.set_attribute('sensor_tick', str(capture_intervals))

        lidar_transform = carla.Transform(carla.Location(x=0, y=0, z=-1), carla.Rotation(pitch=-90))  # 根据需要调整位置
        lidar_sensor = self.world.spawn_actor(lidar_blueprint, lidar_transform, self.static_actor)
        lidar_sensor.listen(lambda data: self.process_dot_image(data, "down", "dot"))
        # lidar_sensor.listen(lambda data: self.draw_lidar(self.display,self.process_lidar_data(data)))

        self.sensors.append(lidar_sensor)

        # 创建四个不同方向的传感器
        for direction, yaw in zip(directions, yaw_angles):
            rgb_blueprint = self.world.get_blueprint_library().find('sensor.camera.rgb')
            rgb_blueprint.set_attribute('image_size_x', str(image_size_x))
            rgb_blueprint.set_attribute('image_size_y', str(image_size_y))
            rgb_blueprint.set_attribute('fov', str(fov))
            rgb_blueprint.set_attribute('sensor_tick', str(capture_intervals))

            rgb_transform = carla.Transform(carla.Location(x=0, y=0, z=-1), carla.Rotation(yaw=yaw, pitch=pitch_degree))
            rgb_sensor = self.world.spawn_actor(rgb_blueprint, rgb_transform, self.static_actor)
            rgb_sensor.listen(lambda data, dir=direction: self.process_image(data, dir, "rgb"))
            self.sensors.append(rgb_sensor)

            depth_blueprint = self.world.get_blueprint_library().find('sensor.camera.depth')
            depth_blueprint.set_attribute('image_size_x', str(image_size_x))
            depth_blueprint.set_attribute('image_size_y', str(image_size_y))
            depth_blueprint.set_attribute('fov', str(fov))
            depth_blueprint.set_attribute('sensor_tick', str(capture_intervals))

            depth_transform = carla.Transform(carla.Location(x=0, y=0, z=-1), carla.Rotation(yaw=yaw, pitch=pitch_degree))
            depth_sensor = self.world.spawn_actor(depth_blueprint, depth_transform, self.static_actor)
            depth_sensor.listen(lambda data, dir=direction: self.process_depth_image(data, dir, "depth"))
            self.sensors.append(depth_sensor)

    def process_image(self, image, direction, sensor_type):
        image.save_to_disk(r'D:\CARLA_Latest\WindowsNoEditor\myDemo\_rgb_out\rgb_%s_%s_%06d.png' % (direction, sensor_type, image.frame))

    # def process_depth_image(self, image, direction, sensor_type):
    #     image.convert(carla.ColorConverter.LogarithmicDepth)
    #     image.save_to_disk(r'D:\CARLA_Latest\WindowsNoEditor\myDemo\_depth_out\depth_%s_%s_%06d.png' % (direction, sensor_type, image.frame))

    def process_depth_image(self, image, direction, sensor_type):
        # print(image)
        # print("\n")
        image.convert(carla.ColorConverter.LogarithmicDepth)
        image.save_to_disk(r'D:\CARLA_Latest\WindowsNoEditor\myDemo\_depth_out\depth_%s_%s_%06d.png' % (direction, sensor_type, image.frame))
        # depth_map=Image.open(r'D:\CARLA_Latest\WindowsNoEditor\myDemo\_depth_out\depth_%s_%s_%06d.png' % (direction, sensor_type, image.frame)).convert("L")
        # depth_map=np.array(depth_map)
        # points = depth_to_point_cloud(depth_map)
        # # # print(points)
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(points)
        # o3d.io.write_point_cloud(r'D:\CARLA_Latest\WindowsNoEditor\myDemo\_dot_out\dot_%s_%s_%06d.pcd' % (direction, sensor_type, image.frame), pcd)
        # # image.save_to_disk(r'D:\CARLA_Latest\WindowsNoEditor\myDemo\_depth_out\depth_%s_%s_%06d.png' % (direction, sensor_type, image.frame))

    def process_dot_image(self, image, direction, sensor_type):
        points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
        points = np.reshape(points, (int(points.shape[0] / 4), 4))

        # 保存为PCD格式
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points[:, :3])


        o3d.io.write_point_cloud(r'D:\CARLA_Latest\WindowsNoEditor\myDemo\_dot_out\dot_%s_%s_%06d.pcd' % (direction, sensor_type, image.frame), pcd)


    def move(self, delta_location):
        new_location = self.static_actor.get_location() + delta_location
        self.static_actor.set_location(new_location)
        
    def destroy(self):
        # 销毁所有传感器
        for sensor in self.sensors:
            sensor.destroy()
        # 销毁静态演员
        if self.static_actor is not None:
            self.static_actor.destroy()


    def show_dot(self,image):
        disp_size = [800,450]
        lidar_range = 2.0 * float(200)
        points = np.frombuffer(image.raw_data, dtype=np.dtype('f4'))
        points = np.reshape(points, (int(points.shape[0] / 4), 4))
        lidar_data = np.array(points[:, :2])
        lidar_data *= min(disp_size) / lidar_range
        lidar_data += (0.5 * disp_size[0], 0.5 * disp_size[1])
        lidar_data = np.fabs(lidar_data)  # pylint: disable=E1111
        lidar_data = lidar_data.astype(np.int32)
        lidar_data = np.reshape(lidar_data, (-1, 2))
        lidar_img_size = (disp_size[0], disp_size[1], 3)
        lidar_img = np.zeros((lidar_img_size), dtype=np.uint8)
        lidar_img[tuple(lidar_data.T)] = (255, 255, 255)

        surface = pygame.surfarray.make_surface(lidar_img)
        self.mySurface.blit(surface, (0, 0))  # 更新图像到pygame窗口
        pygame.display.flip()

    def process_lidar_data(self,point_cloud):
        points = np.frombuffer(point_cloud.raw_data, dtype=np.dtype('f4'))
        points = np.reshape(points, (int(points.shape[0] / 4), 4))
        lidar_data = np.array(points[:, :2])
        lidar_data *= 800 / 100.0
        lidar_data += np.array([400, 300])
        return lidar_data

    def draw_lidar(self,display, lidar_data):
        display.fill((0, 0, 0))
        for point in lidar_data:
            pygame.draw.circle(display, (255, 255, 255), (int(point[0]), int(point[1])), 2)
        pygame.display.flip()