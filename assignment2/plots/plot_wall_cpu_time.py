import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

filename = "jacobi_j_v1.dat"
path = "/home/max/Documents/DTU/HighPerformanceComputing/hpc_assignments/assignment2/plots/"
file = path+filename

column_names = ["j","alloc_time", "init_time", "compute_time", "N3_times_Niter","Niter","threads","wall_time","cpu_time"]

df = pd.read_csv(file,header=None,names=column_names)
df = df.drop(columns=df.columns[0])

df = df[df['N3_times_Niter']==10240000000]


# Create a single figure with 1 row and 2 columns
fig, axes = plt.subplots(1, 2, figsize=(8, 4))  # Adjust figsize as needed

# First subplot: Wall Time
axes[0].plot(df['threads'], df['wall_time'], "o-")
axes[0].set_xlabel("Number of Threads")
axes[0].set_ylabel("Wall Time [s]")
axes[0].set_title("Wall Time vs Threads")
axes[0].grid(True)

# Second subplot: CPU Time
axes[1].plot(df['threads'], df['cpu_time'], "o-")
axes[1].set_xlabel("Number of Threads")
axes[1].set_ylabel("CPU Time [s]")
axes[1].set_title("CPU Time vs Threads")
axes[1].grid(True)

# Adjust layout to prevent overlap
plt.tight_layout()

# Display the combined plots
plt.show()