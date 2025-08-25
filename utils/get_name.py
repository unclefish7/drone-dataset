import os
import json

# 获取脚本所在的绝对路径
# script_dir = os.path.dirname(os.path.abspath(__file__))
script_dir = r"E:\datasets\mydataset_OPV2V_h50_all_data\train"
# 获取所有子目录（排除文件）
folders = [
    f for f in os.listdir(script_dir)
    if os.path.isdir(os.path.join(script_dir, f))
]

# 构建数据结构
output_data = {
    folder: {
        "1": "m1",
        "2": "m1", 
        "3": "m1",
        "4": "m1"
    }
    for folder in folders
}

# 输出到脚本所在目录
output_path = os.path.join(script_dir, "output.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(output_data, f, indent=4, ensure_ascii=False)

print(f"生成完成 -> {output_path}")
