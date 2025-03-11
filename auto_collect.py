import os
import subprocess

def run_scripts_with_args(directory, scripts_with_args):
    for script, args in scripts_with_args:
        filepath = os.path.join(directory, script)
        if os.path.isfile(filepath) and script.endswith(".py"):
            subprocess.run(["python", filepath] + args)

if __name__ == "__main__":
    current_directory = os.path.dirname(os.path.abspath(__file__))
    scripts_with_args = [
        # ("auto_drive_with_sensors.py", ["--repetitions", "1", "--town", "Town03"]),
        ("auto_drive_with_sensors.py", ["--repetitions", "1", "--town", "Town05"]),
    ]
    run_scripts_with_args(current_directory, scripts_with_args)