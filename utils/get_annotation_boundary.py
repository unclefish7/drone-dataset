import os
import yaml

def compute_z_bounds_from_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    vehicles = data.get('vehicles', {})
    z_mins, z_maxs = [], []

    for vid, vinfo in vehicles.items():
        loc = vinfo.get('location', [0, 0, 0])
        extent = vinfo.get('extent', [0, 0, 0])
        z_min = loc[2] - extent[2]
        z_max = loc[2] + extent[2]
        z_mins.append(z_min)
        z_maxs.append(z_max)

    return z_mins, z_maxs

def traverse_and_compute(root_dir):
    global_z_mins, global_z_maxs = [], []

    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.yaml') or file.endswith('.yml'):
                filepath = os.path.join(root, file)
                try:
                    z_mins, z_maxs = compute_z_bounds_from_file(filepath)
                    global_z_mins.extend(z_mins)
                    global_z_maxs.extend(z_maxs)
                except Exception as e:
                    print(f"Failed to process {filepath}: {e}")

    if global_z_mins and global_z_maxs:
        print(f"Global Z min: {min(global_z_mins):.4f}")
        print(f"Global Z max: {max(global_z_maxs):.4f}")
    else:
        print("No vehicle annotations found.")

if __name__ == '__main__':
    # 将路径换成你的yaml文件夹路径
    root_directory = r'E:\datasets\mydataset\train'
    traverse_and_compute(root_directory)
