import carla
import numpy as np

def process_image(image, direction, sensor_type):
    image.save_to_disk(r'D:\CARLA_Latest\WindowsNoEditor\myDemo\_rgb_out\rgb_%s_%s_%06d.png' % (direction, sensor_type, image.frame))

def process_depth_image(image, direction, sensor_type):
    image.convert(carla.ColorConverter.LogarithmicDepth)
    image.save_to_disk(r'D:\CARLA_Latest\WindowsNoEditor\myDemo\_depth_out\depth_%s_%s_%06d.png' % (direction, sensor_type, image.frame))

def depth_to_point_cloud(depth_image):
    # Convert depth image to point cloud
    # Get camera intrinsic parameters
    image_size_x = depth_image.width
    image_size_y = depth_image.height
    fov = float(depth_image.fov)
    focal_length = image_size_x / (2.0 * np.tan(fov * np.pi / 360.0))
    cx = image_size_x / 2.0
    cy = image_size_y / 2.0

    depth_data = np.frombuffer(depth_image.raw_data, dtype=np.dtype("uint8"))
    depth_data = depth_data.reshape((depth_image.height, depth_image.width, 4))
    depth_data = depth_data[:, :, :3]

    points = []
    for v in range(image_size_y):
        for u in range(image_size_x):
            z = depth_data[v, u, 0] / 255.0 * 1000.0
            if z > 0.1:  # Ignore points that are too close
                x = (u - cx) * z / focal_length
                y = (v - cy) * z / focal_length
                points.append([x, y, z])

    return np.array(points)

def spawn_uav_with_sensors(world, location, yaw_angle, static_actor_list, sensor_list):
    image_size_x = 800
    image_size_y = 450
    pitch_degree = -45
    fov = 90
    capture_intervals = 5.0

    directions = ["North", "East", "South", "West"]
    yaw_angles = [0, 90, 180, 270]

    static_blueprint = world.get_blueprint_library().find('static.prop.box01')
    spawn_point = carla.Transform(location, carla.Rotation(yaw=yaw_angle))
    static_actor = world.spawn_actor(static_blueprint, spawn_point)
    static_actor_list.append(static_actor)

    # 创建垂直向下的传感器
    rgb_blueprint = world.get_blueprint_library().find('sensor.camera.rgb')
    rgb_blueprint.set_attribute('image_size_x', str(image_size_x))
    rgb_blueprint.set_attribute('image_size_y', str(image_size_y))
    rgb_blueprint.set_attribute('fov', str(fov))
    rgb_blueprint.set_attribute('sensor_tick', str(capture_intervals))

    rgb_transform = carla.Transform(carla.Location(x=0, y=0, z=-1), carla.Rotation(pitch=-90))
    rgb_sensor = world.spawn_actor(rgb_blueprint, rgb_transform, static_actor)
    rgb_sensor.listen(lambda data: process_image(data, "down", "rgb"))
    sensor_list.append(rgb_sensor)

    depth_blueprint = world.get_blueprint_library().find('sensor.camera.depth')
    depth_blueprint.set_attribute('image_size_x', str(image_size_x))
    depth_blueprint.set_attribute('image_size_y', str(image_size_y))
    depth_blueprint.set_attribute('fov', str(fov))
    depth_blueprint.set_attribute('sensor_tick', str(capture_intervals))

    depth_transform = carla.Transform(carla.Location(x=0, y=0, z=-1), carla.Rotation(pitch=-90))
    depth_sensor = world.spawn_actor(depth_blueprint, depth_transform, static_actor)
    depth_sensor.listen(lambda data: process_depth_image(data, "down", "depth"))
    sensor_list.append(depth_sensor)
    
    # 创建四个不同方向的传感器
    for direction, yaw in zip(directions, yaw_angles):
        rgb_blueprint = world.get_blueprint_library().find('sensor.camera.rgb')
        rgb_blueprint.set_attribute('image_size_x', str(image_size_x))
        rgb_blueprint.set_attribute('image_size_y', str(image_size_y))
        rgb_blueprint.set_attribute('fov', str(fov))
        rgb_blueprint.set_attribute('sensor_tick', str(capture_intervals))

        rgb_transform = carla.Transform(carla.Location(x=0, y=0, z=-1), carla.Rotation(yaw=yaw, pitch=pitch_degree))
        rgb_sensor = world.spawn_actor(rgb_blueprint, rgb_transform, static_actor)
        rgb_sensor.listen(lambda data, dir=direction: process_image(data, dir, "rgb"))
        sensor_list.append(rgb_sensor)

        depth_blueprint = world.get_blueprint_library().find('sensor.camera.depth')
        depth_blueprint.set_attribute('image_size_x', str(image_size_x))
        depth_blueprint.set_attribute('image_size_y', str(image_size_y))
        depth_blueprint.set_attribute('fov', str(fov))
        depth_blueprint.set_attribute('sensor_tick', str(capture_intervals))

        depth_transform = carla.Transform(carla.Location(x=0, y=0, z=-1), carla.Rotation(yaw=yaw, pitch=pitch_degree))
        depth_sensor = world.spawn_actor(depth_blueprint, depth_transform, static_actor)
        depth_sensor.listen(lambda data, dir=direction: process_depth_image(data, dir, "depth"))
        sensor_list.append(depth_sensor)
