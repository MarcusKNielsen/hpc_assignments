import os
import pandas as pd
import matplotlib.pyplot as plt

# Globals
L1 = 32
L2 = 256
L3 = 30 * 1024

# Load files
folder_path = "/home/max/Documents/DTU/HighPerformanceComputing/hpc_assignments/assignment1/results"
files = os.listdir(folder_path)

files = [file for file in files if file.lower().startswith('r')]

titles = {"results_Ofast_fno_unroll.dat": "-Ofast -fno_unroll-loops",
          "results_-O3_-funroll-all-loops.dat": "-O3 -funroll-all-loops",
          "results_-O.dat": "-O",
          "results_O3_fno_unroll.dat": "-O3 -fno_unroll-loops",
          "results_O3_funroll.dat": "-O3 -funroll-loops",
          "results_-Ofast_-funroll-all-loops.dat": "-Ofast -funroll-all-loops"
          }

set_of_files = {'results_no_opt_divide_m.dat',
        'results_no_opt_divide_k.dat',
        'results_no_opt_divide_n.dat',
        "results_smaller_m.dat",
        "results_no_opt.dat",
        'results_blk.dat'}

# Create a figure with subplots
fig, axes = plt.subplots(3, 2, figsize=(6, 7))
axes = axes.flatten()  # Flatten to easily index

# Plot stuff
plot_index = 0
handles, labels = None, None  # Initialize handles and labels for the shared legend

for i in range(len(files)):
    if files[i] not in set_of_files:

        file_name = folder_path + "/" + files[i]
        data = pd.read_csv(file_name, delim_whitespace=True, header=None)
        data = data.drop(columns=[2])
        data.columns = ['memory', 'performance', 'version']
        versions = data['version'].unique()

        ax = axes[plot_index]  # Select the current subplot

        ax.vlines([L1, L2, L3], 0, 5000, linestyle="--", color="red", label="Cache Levels")
        for v in range(len(versions)):
            if versions[v] != 'matmult_lib':
                x = data[data['version'] == versions[v]]['memory']
                y = data[data['version'] == versions[v]]['performance']
                ax.semilogx(x, y, ".-", label=versions[v][-3:])

        if handles is None and labels is None:
            # Capture the handles and labels from the first subplot
            handles, labels = ax.get_legend_handles_labels()

        ax.set_yticks([0, 1500, 3000, 4500])  # Set specific ticks on the y-axis
        ax.set_xlabel("Memory [KiB]")
        ax.set_ylabel("Compute [Mflop/s]")
        ax.set_title(titles[files[i]])
        plot_index += 1

# Add shared legend
fig.legend(
    handles, labels,
    loc='lower center',  # Position the legend below the plot
    bbox_to_anchor=(0.5, -0.001),  # Adjust the vertical position of the legend
    ncol=4
)

# Adjust layout
plt.tight_layout(rect=[0, 0.1, 1, 1])
plt.show()

#%%

titles = {"results_Ofast_fno_unroll.dat": "-Ofast -fno_unroll-loops",
          "results_-O3_-funroll-all-loops.dat": "-O3 -funroll-all-loops",
          "results_-O.dat": "-O",
          "results_O3_fno_unroll.dat": "-O3 -fno_unroll-loops",
          "results_O3_funroll.dat": "-O3 -funroll-loops",
          "results_-Ofast_-funroll-all-loops.dat": "-Ofast -funroll-all-loops",
          "results_no_opt.dat": "No optimization"
          }

# Create a figure with subplots
plt.figure(figsize=(7, 5.7))
plt.vlines([L1, L2, L3], 0, 8500, linestyle="--", color="red", label="Cache Levels")

for i in range(len(files)):
    if files[i] not in set_of_files:
        file_name = folder_path + "/" + files[i]
        data = pd.read_csv(file_name, delim_whitespace=True, header=None)
        data = data.drop(columns=[2])
        data.columns = ['memory', 'performance', 'version']
        versions = data['version'].unique()
    
    
        
        for v in range(len(versions)):
            if versions[v] == 'matmult_mkn':
                x = data[data['version'] == versions[v]]['memory']
                y = data[data['version'] == versions[v]]['performance']
                plt.semilogx(x, y, ".-", label=titles[files[i]])
        
        plt.xlabel("Memory [KiB]")
        plt.ylabel("Compute [Mflop/s]")
        plt.title("Best Version vs. Library (mkn vs. cblas)")

x = data[data['version'] == 'matmult_lib']['memory']
y = data[data['version'] == 'matmult_lib']['performance']
plt.semilogx(x, y, ".-", label="dgemm (BLAS)")

plt.legend()
# Adjust layout
plt.tight_layout()
plt.show()

#%%

# Create a figure with subplots
plt.figure(figsize=(7, 5.7))
plt.vlines([L1, L2, L3], 180, 350, linestyle="--", color="red", label="Cache Levels")


file_name = folder_path + "/" + "results_no_opt.dat"
data = pd.read_csv(file_name, delim_whitespace=True, header=None)
data = data.drop(columns=[2])
data.columns = ['memory', 'performance', 'version']
versions = data['version'].unique()


for v in range(len(versions)):
    if versions[v] != 'matmult_lib':
        x = data[data['version'] == versions[v]]['memory']
        y = data[data['version'] == versions[v]]['performance']
        plt.semilogx(x, y, ".-", label=versions[v][-3:])

plt.xlabel("Memory [KiB]")
plt.ylabel("Compute [Mflop/s]")
plt.title("Performance With No Optimization")

plt.legend()
plt.tight_layout()
plt.show()

#%%

# Create a figure with subplots
fig, axes = plt.subplots(1, 2, figsize=(8, 4))  # 1 row, 2 columns

# Load and preprocess the data
file_name = folder_path + "/" + "results_no_opt.dat"
data = pd.read_csv(file_name, delim_whitespace=True, header=None)
data = data.drop(columns=[2])
data.columns = ['memory', 'performance', 'version']
versions = data['version'].unique()

# Shared x-axis and y-axis labels
x_label = "Memory [KiB]"
y_label = "Compute [Mflop/s]"
title1 = "Performance: matmult_lib"
title2 = "Performance: matmult_nat"

# Plot for 'matmult_lib'
axes[0].vlines([L1, L2, L3], 0, 8500, linestyle="--", color="red", label="Cache Levels")
x = data[data['version'] == 'matmult_lib']['memory']
y = data[data['version'] == 'matmult_lib']['performance']
axes[0].semilogx(x, y, "o-", label='matmult_lib')
axes[0].set_xlabel(x_label)
axes[0].set_ylabel(y_label)
axes[0].set_title(title1)
axes[0].legend()

# Plot for 'matmult_nat'
axes[1].vlines([L1, L2, L3], 150, 400, linestyle="--", color="red", label="Cache Levels")
x = data[data['version'] == 'matmult_nat']['memory']
y = data[data['version'] == 'matmult_nat']['performance']
axes[1].semilogx(x, y, "o-", label='matmult_nat')
axes[1].set_xlabel(x_label)
axes[1].set_title(title2)
axes[1].legend()

# Adjust layout
plt.tight_layout()
plt.show()

#%%

# "matmult_mnk"
# "matmult_nkm"

file_name = folder_path + "/" + 'results_no_opt_divide_m.dat'
data = pd.read_csv(file_name, delim_whitespace=True, header=None)
data = data.drop(columns=[2])
data.columns = ['memory', 'performance', 'version']

plt.figure(figsize=(6,4))

plt.title(r"Performance with $n=k=10m$")

plt.vlines([L1, L2, L3], 100, 350, linestyle="--", color="red", label="Cache Levels")


x = data[data['version'] == "matmult_mnk"]['memory']
y = data[data['version'] == "matmult_mnk"]['performance']
plt.semilogx(x, y, "o-", label="mnk")

x = data[data['version'] == "matmult_nkm"]['memory']
y = data[data['version'] == "matmult_nkm"]['performance']
plt.semilogx(x, y, "o-", label="nkm")

plt.xlabel("Memory [KiB]")
plt.ylabel("Compute [Mflop/s]")
plt.legend(loc="lower right")        

plt.tight_layout()
plt.show()





