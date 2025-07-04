import os
import yaml
import statistics

def compute_z_bounds_from_file(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    vehicles = data.get('vehicles', {})
    bottoms = []
    tops = []

    for vid, vinfo in vehicles.items():
        loc = vinfo.get('location', [0, 0, 0])
        extent = vinfo.get('extent', [0, 0, 0])
        z_min = loc[2] - extent[2]
        z_max = loc[2] + extent[2]
        bottoms.append((z_min, filepath, vid))
        tops.append((z_max, filepath, vid))

    return bottoms, tops

def compute_statistics(z_items, label):
    z_values = [item[0] for item in z_items]
    total_count = len(z_values)

    if total_count == 0:
        print(f"{label}：无数据")
        return

    global_min = min(z_items, key=lambda x: x[0])
    global_max = max(z_items, key=lambda x: x[0])
    z_mean = sum(z_values) / total_count
    z_median = statistics.median(z_values)
    z_variance = statistics.variance(z_values)
    z_stddev = statistics.stdev(z_values)

    print(f"\n===== {label} =====")
    print(f"最小Z值: {global_min[0]:.4f}，文件: {global_min[1]}，车辆ID: {global_min[2]}")
    print(f"最大Z值: {global_max[0]:.4f}，文件: {global_max[1]}，车辆ID: {global_max[2]}")
    print(f"平均Z值 (mean): {z_mean:.4f}")
    print(f"中位数Z值 (median): {z_median:.4f}")
    print(f"方差 (variance): {z_variance:.4f}")
    print(f"标准差 (standard deviation, σ): {z_stddev:.4f}\n")

    print("偏差统计（以均值为中心的 ±σ 范围内统计）:")
    for multiplier in [1, 2, 3]:
        threshold = multiplier * z_stddev
        inside_count = sum(1 for z in z_values if abs(z - z_mean) <= threshold)
        inside_ratio = inside_count / total_count * 100
        outside_count = total_count - inside_count
        outside_ratio = outside_count / total_count * 100

        print(f"  - 在均值 ±{multiplier}σ 范围内的数量: {inside_count} / {total_count} ({inside_ratio:.2f}%)")
        print(f"  - 超出均值 ±{multiplier}σ 范围的数量: {outside_count} / {total_count} ({outside_ratio:.2f}%)")

def traverse_and_compute(root_dir):
    all_bottoms = []
    all_tops = []

    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if file.endswith('.yaml') or file.endswith('.yml'):
                filepath = os.path.join(root, file)
                try:
                    bottoms, tops = compute_z_bounds_from_file(filepath)
                    all_bottoms.extend(bottoms)
                    all_tops.extend(tops)
                except Exception as e:
                    print(f"无法处理文件 {filepath}: {e}")

    compute_statistics(all_bottoms, "车辆底部最低点统计")
    compute_statistics(all_tops, "车辆顶部最高点统计")

if __name__ == '__main__':
    root_directory = r'E:\datasets\mydataset_OPV2V_lowerZ\train'
    traverse_and_compute(root_directory)
