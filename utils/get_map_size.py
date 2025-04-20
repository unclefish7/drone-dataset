import carla

client = carla.Client('localhost', 2000)
client.set_timeout(10.0)

# 选择你想要的地图
world = client.load_world_if_different('Town04')
if not world:
    world = client.get_world()


map = world.get_map()
spawn_points = map.get_spawn_points()

min_x = min(spawn_points, key=lambda p: p.location.x).location.x
max_x = max(spawn_points, key=lambda p: p.location.x).location.x
min_y = min(spawn_points, key=lambda p: p.location.y).location.y
max_y = max(spawn_points, key=lambda p: p.location.y).location.y

print(f"Map size: width = {max_x - min_x} meters, height = {max_y - min_y} meters")
