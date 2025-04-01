import os
from collections import defaultdict

# 设置根目录
ROOT_DIR = r"D:\datasets\mydataset"
REQUIRED_EXTENSIONS = [".pcd", ".yaml", "_bev_visibility.png", "_segmentation.png"]
CAMERA_FILES = [f"_camera{i}.png" for i in range(5)]

def check_scene(scene_path):
    missing_report = defaultdict(list)
    for uav_id in ['1', '2', '3', '4']:
        uav_path = os.path.join(scene_path, uav_id)
        if not os.path.isdir(uav_path):
            missing_report['missing_uav_dir'].append(uav_path)
            continue

        # 收集所有帧编号（根据 .pcd 文件名提取）
        frame_ids = set()
        for f in os.listdir(uav_path):
            if f.endswith(".pcd"):
                frame_ids.add(f.replace(".pcd", ""))

        for frame_id in sorted(frame_ids):
            expected_files = [frame_id + ext for ext in REQUIRED_EXTENSIONS]
            expected_files += [frame_id + cam for cam in CAMERA_FILES]

            for ef in expected_files:
                if not os.path.exists(os.path.join(uav_path, ef)):
                    missing_report['missing_files'].append(os.path.join(uav_path, ef))

    return missing_report


def check_all(root_path):
    all_missing = {}
    for split in ['train', 'test', 'validate']:
        split_path = os.path.join(root_path, split)
        if not os.path.exists(split_path):
            print(f"[警告] 缺失 split 文件夹：{split_path}")
            continue

        for scene in os.listdir(split_path):
            scene_path = os.path.join(split_path, scene)
            if not os.path.isdir(scene_path):
                continue

            print(f"正在检查场景：{scene_path}")
            report = check_scene(scene_path)
            if report:
                all_missing[scene_path] = report

    return all_missing


if __name__ == "__main__":
    results = check_all(ROOT_DIR)

    if not results:
        print("✅ 数据集结构完整，无缺失。")
    else:
        print("\n❌ 发现问题：")
        for scene, issues in results.items():
            print(f"\n[场景] {scene}")
            for issue_type, files in issues.items():
                print(f"  - {issue_type}: 共 {len(files)} 个问题")
                for f in files[:5]:  # 只展示前5个
                    print(f"    - {f}")
                if len(files) > 5:
                    print(f"    ...（共 {len(files)} 个）")
