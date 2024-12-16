import subprocess
import os,sys
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)
# Add the parent directory of 'models' to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# Specify the path to the arm_demo.py script
script_path = project_root+"/experiments/arm_demo.py"

# Run the script for 1000 iterations
for i in range(1, 100001):
    print(f"Running iteration {i}...")
    result = subprocess.run(["python3", script_path, str(i)], capture_output=True, text=True)
    
    # Log output and errors, if any
    if result.returncode == 0:
        print(f"Iteration {i} completed successfully.")
    else:
        print(f"Error in iteration {i}:")
        print(result.stderr)
