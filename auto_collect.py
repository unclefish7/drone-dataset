import os
import subprocess

def run_scripts_with_args(directory, scripts_with_args):
    for script, args in scripts_with_args:
        filepath = os.path.join(directory, script)
        if os.path.isfile(filepath) and script.endswith(".py"):
            subprocess.run(["python", filepath] + args)

if __name__ == "__main__":
    current_directory = os.path.dirname(os.path.abspath(__file__))
    script_name = "auto_drive_with_sensors.py"
    scripts_with_args = [
        # (script_name, ["--repetitions", "10", "--town", "Town03", "--random_seed", "0"]),
        (script_name, ["--repetitions", "1", "--town", "Town03_2", "--random_seed", "0"]),
        # (script_name, ["--repetitions", "5", "--town", "Town03_3", "--random_seed", "0"]),
        # (script_name, ["--repetitions", "5", "--town", "Town03_4", "--random_seed", "0"]),
        # (script_name, ["--repetitions", "10", "--town", "Town05", "--random_seed", "0"]),
    ]
    run_scripts_with_args(current_directory, scripts_with_args)