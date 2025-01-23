import pandas as pd
import matplotlib.pyplot as plt

path = "/home/max/Documents/DTU/HighPerformanceComputing/rawData_Assignment3/"
filename = "matmul_ex2_graph1_data"
file = path + filename

# Define column names
columns = ['memory', 'flops', 'hashtag', 'name1', 'name2', 'size', 'teams_threads']

# Read the file with proper whitespace handling
data = pd.read_csv(
    file, 
    delim_whitespace=True, 
    header=None, 
    names=columns
)

# Split the 'teams_threads' column into 'teams' and 'threads'
data[['teams', 'threads']] = data['teams_threads'].str.split(',', expand=True)

# Drop the original 'teams_threads' column
data = data.drop(columns=['teams_threads'])

data = data.drop(columns=['hashtag', 'name1'])

version = data['name2'].unique()

plt.figure()
for i in range(len(version)):
    
    sub_data = data[data['name2'] == version[i]]
    plt.plot(sub_data['memory'],sub_data['flops'],".-",label=f"{version[i]}")


plt.xlabel("Memory [KiB]")
plt.ylabel(r"Performance [Mflop/s]")
plt.legend()
plt.tight_layout()

plt.savefig(filename)

plt.show()

