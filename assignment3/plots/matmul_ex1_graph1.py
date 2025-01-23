import pandas as pd
import matplotlib.pyplot as plt

path = "/home/max/Documents/DTU/HighPerformanceComputing/rawData_Assignment3/"
filename = "matmul_ex1_graph1_data"
file = path + filename

# Define column names
columns = ['memory', 'flops', 'hashtag', 'name1', 'name2', 'size', 'threads']

# Read the file with proper whitespace handling
data = pd.read_csv(file, 
                   delim_whitespace=True, 
                   header=None, 
                   names=columns)

data = data.drop(columns=['hashtag', 'name1', 'name2'])

threads = data['threads'].unique()

plt.figure()
for i in range(len(threads)):
    
    sub_data = data[data['threads'] == threads[i]]
    plt.plot(sub_data['size'],sub_data['flops'],".-",label=f"threads = {threads[i]}")

plt.xlabel("Matrix Size")
plt.ylabel(r"Performance [Mflop/s]")
plt.legend()
plt.tight_layout()

plt.savefig(filename)

plt.show()
    