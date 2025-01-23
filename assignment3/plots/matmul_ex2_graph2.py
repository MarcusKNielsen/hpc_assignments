import pandas as pd
import matplotlib.pyplot as plt

path = "/home/max/Documents/DTU/HighPerformanceComputing/rawData_Assignment3/"
filename = "matmul_ex2_graph2_data"
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

tt_unique = data[['teams', 'threads']].drop_duplicates()


plt.figure()
for j in range(len(tt_unique)):
    tt_combination = tt_unique.loc[j]
    desired_team = tt_combination['teams']
    desired_thread = tt_combination['threads']
    sub_data = data[(data['teams'] == desired_team) & (data['threads'] == desired_thread)]
    plt.plot(sub_data['memory'],sub_data['flops'],".-",label=f"teams={desired_team}, threads={desired_thread}")

plt.xlabel("Memory [KiB]")
plt.ylabel(r"Performance [Mflop/s]")
plt.legend()
plt.tight_layout()

plt.savefig(filename)

plt.show()





