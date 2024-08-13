import carla
import numpy as np

class UAV:
    def __init__(self, world, location, yaw_angle):
        self.world = world
        self.location = location
        self.yaw_angle = yaw_angle
        self.static_actor = None
        self.sensors = []
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
        # rgb_sensor.listen(lambda data: self.process_image(data, "down", "rgb"))
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
            rgb_blueprint = self.world.get_blueprint_library().find('sensor.camera.rgb')
            rgb_blueprint.set_attribute('image_size_x', str(image_size_x))
            rgb_blueprint.set_attribute('image_size_y', str(image_size_y))
            rgb_blueprint.set_attribute('fov', str(fov))
            rgb_blueprint.set_attribute('sensor_tick', str(capture_intervals))

            rgb_transform = carla.Transform(carla.Location(x=0, y=0, z=-1), carla.Rotation(yaw=yaw, pitch=pitch_degree))
            rgb_sensor = self.world.spawn_actor(rgb_blueprint, rgb_transform, self.static_actor)
            # rgb_sensor.listen(lambda data, dir=direction: self.process_image(data, dir, "rgb"))
            self.sensors.append(rgb_sensor)

            depth_blueprint = self.world.get_blueprint_library().find('sensor.camera.depth')
            depth_blueprint.set_attribute('image_size_x', str(image_size_x))
            depth_blueprint.set_attribute('image_size_y', str(image_size_y))
            depth_blueprint.set_attribute('fov', str(fov))
            depth_blueprint.set_attribute('sensor_tick', str(capture_intervals))

            depth_transform = carla.Transform(carla.Location(x=0, y=0, z=-1), carla.Rotation(yaw=yaw, pitch=pitch_degree))
            depth_sensor = self.world.spawn_actor(depth_blueprint, depth_transform, self.static_actor)
            # depth_sensor.listen(lambda data, dir=direction: self.process_depth_image(data, dir, "depth"))
            self.sensors.append(depth_sensor)

    def process_image(self, image, direction, sensor_type):
        image.save_to_disk(r'D:\CARLA_Latest\WindowsNoEditor\myDemo\_rgb_out\rgb_%s_%s_%06d.png' % (direction, sensor_type, image.frame))

    def process_depth_image(self, image, direction, sensor_type):
        image.convert(carla.ColorConverter.LogarithmicDepth)
        image.save_to_disk(r'D:\CARLA_Latest\WindowsNoEditor\myDemo\_depth_out\depth_%s_%s_%06d.png' % (direction, sensor_type, image.frame))

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