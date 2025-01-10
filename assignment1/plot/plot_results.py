import pandas as pd
import matplotlib.pyplot as plt

# Specify the file name
#file_name = '/home/max/Documents/DTU/HighPerformanceComputing/hpc_assignments/assignment1/results.dat'
file_name = '/home/max/Documents/DTU/HighPerformanceComputing/results_O3_funroll.txt'

# Read the .dat file using pandas
data = pd.read_csv(file_name, delim_whitespace=True, header=None)

# Remove column 2 (index 2 as Python is zero-indexed)
data = data.drop(columns=[2])

# Add column names: memory, performance, version
data.columns = ['memory', 'performance', 'version']

# Optional: Print or save the modified DataFrame to confirm changes
print(data)

# Get unique values in the 'version' column
versions = data['version'].unique()

# Print the unique values
print(versions)

#%%

L1 = 32
L2 = 56
L3 = 30*1024

plt.figure()
plt.vlines([L1, L2, L1+L2+L3], 180, data['performance'].max(), linestyle="--", color="red", label="Cache Levels")
for i in range(len(versions)):
    if versions[i] != 'matmult_lib':
        x = data[data['version'] == versions[i]]['memory']
        y = data[data['version'] == versions[i]]['performance']
        plt.loglog(x,y,".-",label=versions[i])

plt.xlabel("Memory")
plt.ylabel("Performance")
plt.legend()
plt.show()





