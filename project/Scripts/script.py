import subprocess

# Define the parameters
available_bandwidth = 50
crosstraffic_values = [12.5, 25, 37.5]
num_runs = 100

# Run the command for each crosstraffic value, 100 times each
for crosstraffic in crosstraffic_values:
    print(f"cross traffic = {crosstraffic}")
    for test_num in range(num_runs):
        print(f"test no. : {test_num}")
        subprocess.run(["mn", "-c"], check=True)
        command = ["python", "mini.py", str(available_bandwidth), str(crosstraffic)]
        try:
            result = subprocess.run(command, check=True, capture_output=True, text=True)
            print(f"Command executed: {' '.join(command)}")
            print("Output:", result.stdout)
        except subprocess.CalledProcessError as e:
            print(f"Error executing command {' '.join(command)}: {e}")
            print("Error Output:", e.stderr)
