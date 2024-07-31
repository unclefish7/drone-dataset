import carla
actor_list = []

try:
    client = carla.Client('localhost', port = 2000)
    client.set_timeout(5.0)
    world = client.get_world()
    actor_list = world.get_actors()
    for actors in actor_list:
        if actors.type_id == "vehicle.tesla.model3":
            actors.destroy()
        if actors.type_id == "walker.pedestrian.0001":
            actors.destroy()

finally:
    print("deleted")


