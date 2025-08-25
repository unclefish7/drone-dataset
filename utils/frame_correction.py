import os
import re
import shutil
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

# 配置参数
FRAME_GAP_THRESHOLD = 8
MAX_CORRECTION_RANGE = 2  # 最大帧号差值
SCENE_PATTERN = re.compile(r"^\d{4}_\d{2}_\d{2}_\d{2}_\d{2}_\d{2}$")
DRONE_PATTERN = re.compile(r"^\d+$")
FILE_PATTERN = re.compile(r"^(\d+)(?:_(.+))?\.(yaml|pcd|png)$")

def unify_frames(root_dir: str, dry_run: bool = False) -> List[Dict]:
    """
    场景级帧号统一系统
    
    参数：
    root_dir - 数据集根路径
    dry_run - 试运行模式
    
    返回：
    包含详细修改记录的日志列表
    """
    correction_log = []
    root_path = Path(root_dir)
    
    # 遍历train/test/validate目录
    for dataset_type in ["train", "test", "validate"]:
        dataset_path = root_path / dataset_type
        if not dataset_path.exists():
            continue
        
        # 遍历场景目录
        for scene_path in find_scene_dirs(dataset_path):
            # 获取场景内所有无人机目录
            drone_dirs = find_drone_dirs(scene_path)
            if not drone_dirs:
                continue
            
            # 确定基准无人机（编号最小）
            base_drone = min(drone_dirs, key=lambda x: x[1])
            base_yaml_frames = collect_base_frames(base_drone[0])
            
            # 处理其他无人机目录
            for drone_path, drone_id in drone_dirs:
                
                log_entries = process_drone_dir(
                    drone_path, base_yaml_frames, scene_path.name, drone_id, dry_run
                )
                correction_log.extend(log_entries)
    
    return correction_log

def find_scene_dirs(dataset_path: Path) -> List[Path]:
    """查找所有符合格式的场景目录"""
    scene_dirs = []
    for entry in dataset_path.iterdir():
        if entry.is_dir() and SCENE_PATTERN.match(entry.name):
            scene_dirs.append(entry)
    return scene_dirs

def find_drone_dirs(scene_path: Path) -> List[Tuple[Path, int]]:
    """获取场景内所有无人机目录及其编号"""
    drones = []
    for entry in scene_path.iterdir():
        if entry.is_dir() and DRONE_PATTERN.match(entry.name):
            try:
                drones.append((entry, int(entry.name)))
            except ValueError:
                continue
    return drones

def collect_base_frames(base_drone_path: Path) -> Dict[int, Path]:
    """收集基准无人机的YAML帧映射"""
    base_frames = {}
    for f in base_drone_path.glob("*.yaml"):
        match = FILE_PATTERN.match(f.name)
        if match:
            frame = int(match.group(1))
            base_frames[frame] = f
    return base_frames

def process_drone_dir(drone_path: Path, base_frames: Dict[int, Path],
                     scene_id: str, drone_id: int, dry_run: bool) -> List[Dict]:
    """处理单个无人机目录的修正"""
    log_entries = []
    
    for file_path in drone_path.iterdir():
        if not file_path.is_file():
            continue
        
        match = FILE_PATTERN.match(file_path.name)
        if not match:
            continue
        
        current_frame = int(match.group(1))
        suffix = match.group(2) or ""
        ext = match.group(3)
        
        # 寻找最佳基准帧
        base_frame = find_optimal_base(current_frame, base_frames.keys())
        if base_frame is None:
            continue
        
        # 生成新文件名
        new_name = build_new_filename(base_frame, suffix, ext)
        log_entry = execute_rename(
            file_path, new_name, scene_id, drone_id,
            current_frame, base_frame, dry_run
        )
        
        if log_entry:
            log_entries.append(log_entry)
    
    return log_entries

def find_optimal_base(current: int, base_frames: List[int]) -> Optional[int]:
    """寻找最近的合法基准帧（差值在2以内）"""
    candidates = []
    for bf in base_frames:
        delta = abs(bf - current)
        if delta <= MAX_CORRECTION_RANGE:
            candidates.append((delta, bf))
    
    if not candidates:
        return None
    
    # 优先选择差值最小的，差值相同选择较小的帧号
    candidates.sort()
    return candidates[0][1]

def build_new_filename(base_frame: int, suffix: str, ext: str) -> str:
    """构建符合命名规则的新文件名"""
    if suffix:
        return f"{base_frame}_{suffix}.{ext}"
    return f"{base_frame}.{ext}"

def execute_rename(src: Path, new_name: str, scene_id: str, drone_id: int,
                  original: int, base: int, dry_run: bool) -> Dict:
    """执行重命名操作"""
    dest = src.with_name(new_name)
    if src.name == new_name:
        return None
    
    log_entry = {
        "dataset_type": src.parts[-4],  # train/test/validate
        "scene": scene_id,
        "drone_id": drone_id,
        "original_path": str(src),
        "new_path": str(dest),
        "original_frame": original,
        "base_frame": base,
        "action": "dry_run" if dry_run else "renamed",
        "backup": None
    }
    
    if dry_run:
        return log_entry
    
    # 处理文件冲突
    if dest.exists():
        timestamp = int(datetime.now().timestamp())
        backup_name = f"{dest.stem}_conflict_{timestamp}{dest.suffix}"
        backup_path = dest.with_name(backup_name)
        shutil.move(str(dest), str(backup_path))
        log_entry["backup"] = str(backup_path)
    
    shutil.move(str(src), str(dest))
    return log_entry

if __name__ == "__main__":
    # 获取脚本所在目录
    script_dir = Path(__file__).parent
    
    # 示例执行（试运行模式）
    # log = unify_frames("mydataset", dry_run=True)
    
    # 实际执行
    log = unify_frames(r"E:\datasets\mydataset_OPV2V_h50_all_data", dry_run=False)
    
    # 生成CSV格式日志
    csv_log = "DatasetType,Scene,DroneID,OriginalFrame,BaseFrame,OriginalPath,NewPath\n"
    for entry in log:
        csv_log += f"{entry['dataset_type']},{entry['scene']},{entry['drone_id']},"
        csv_log += f"{entry['original_frame']},{entry['base_frame']},"
        csv_log += f"{entry['original_path']},{entry['new_path']}\n"
    
    # 将日志输出到脚本所在目录
    log_path = script_dir / "frame_unification_log.csv"
    with open(log_path, "w") as f:
        f.write(csv_log)
    
    print(f"处理完成，日志已保存至：{log_path}")
