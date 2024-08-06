import carla
import random
import math
import time
import numpy as np
actor_list = []

def pure_persuit(wheel_distance, target_location, v_transform):
    L = 2.875
    yaw = v_transform.rotation.yaw * (math.pi / 180)
    x = v_transform.location.x - L / 2 * math.cos(yaw)
    y = v_transform.location.y - L / 2 * math.sin(yaw)
    dx = target_location.x - x
    dy = target_location.y - y
    ld = math.sqrt(dx ** 2 + dy ** 2) - 3
    #print(ld)
    alpha = math.atan2(dy, dx) - yaw
    delta = math.atan(2 * math.sin(alpha) * L / ld) * 180 / math.pi
    steer = delta / 90
    if steer > 1:
        steer = 1
    elif steer < -1:
        steer = -1

    # v_begin = v_transform.location
    # v_end = v_begin + carla.Location(x=math.cos(math.radians(v_transform.rotation.yaw)), 
    #                                  y=math.sin(math.radians(v_transform.rotation.yaw)))
    # v_vec = np.array([v_end.x - v_begin.x, v_end.y - v_begin.y, 0.0])
    # target_vec = np.array([target_location.x - v_begin.x, target_location.y - v_begin.y, 0.0])
    
    # cross_track_error = np.cross(v_vec, target_vec)[-1]
    # yaw_error = math.degrees(math.atan2(cross_track_error, np.dot(v_vec, target_vec)))
    
    # # 简单比例控制
    # steer = yaw_error / 180.0

    return steer


try:
    client = carla.Client('localhost', port = 2000)
    client.set_timeout(5.0)

    world = client.load_world_if_different('Town03')
    if not world:
        world = client.get_world()

    blueprint_library = world.get_blueprint_library()

    map = world.get_map()
    recommended_spawn_points = map.get_spawn_points()

    audi_bp = blueprint_library.find('vehicle.tesla.model3')
    walker = blueprint_library.find('walker.pedestrian.0001')

    for i in range(0, 100):
        sp = random.choice(recommended_spawn_points)
        audi = world.try_spawn_actor(audi_bp, sp)
        actor_list.append(audi)

    while True:
        for actor in actor_list:
            if actor and actor.type_id == "vehicle.tesla.model3":

                # 获取车辆轮胎信息
                wheels = actor.get_physics_control().wheels

                # 前轮和后轮的位置
                front_left_wheel = wheels[0]
                rear_left_wheel = wheels[2]

                # 计算轴距
                wheelbase = front_left_wheel.position.distance(rear_left_wheel.position)

                waypoint01 = map.get_waypoint(actor.get_location(),project_to_road=True, lane_type=(carla.LaneType.Driving | carla.LaneType.Sidewalk))
                v_trans = actor.get_transform()
                waypoints = waypoint01.next(8.0)
                next_waypoint = None
                if waypoints:
                    next_waypoint = waypoints[0]
                if next_waypoint == None:
                    control = carla.VehicleControl(throttle=0, steer=0)
                else:
                    target_loc = next_waypoint.transform.location
                    steer = pure_persuit(wheelbase, target_loc, v_trans)
                    control = carla.VehicleControl(throttle=0.5, steer=steer)
                actor.apply_control(control)
        time.sleep(0.02)



    # for sp in recommended_spawn_points:
    #     audi = world.try_spawn_actor(audi_bp, sp)
    #     actor_list.append(audi)
    #     control = carla.VehicleControl(throttle=1.0, steer=0)
    #     if audi:
    #         audi.apply_control(control)
        
    #     ped = world.try_spawn_actor(walker, sp)
    #     actor_list.append(ped)
    #     man_ctrl = carla.WalkerControl(speed=10)
    #     if ped:
    #         ped.apply_control(man_ctrl)
        

finally:
    for actors in actor_list:
        actors.destroy()

    print("done")

    