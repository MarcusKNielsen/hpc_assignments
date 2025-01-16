import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl

filename = "poisson_j_serial.dat"
path = "/home/max/Documents/DTU/HighPerformanceComputing/hpc_assignments/assignment2/"
file = path+filename

column_names = ["j","alloc_time", "init_time", "compute_time", "N3_times_Niter","Niter"]

df = pd.read_csv(file,header=None,names=column_names)

df = df.drop(columns=df.columns[0])

df['N'] = (df["N3_times_Niter"] / df["Niter"])**(1/3)
df['memory'] = (df["N3_times_Niter"] / df["Niter"]) * 8 / 1024

# Create a single figure with 1 row and 2 columns
fig, axes = plt.subplots(1, 2, figsize=(8, 4))  # Adjust figsize as needed

# First subplot: Performance plot
perf = df["Niter"] / df["compute_time"]
axes[0].loglog(df['N'], perf, "o-")
axes[0].set_xlabel("Matrix Dimension: N")
axes[0].set_ylabel(r"Performance $\left[\frac{\text{cell update}}{second} \right]$")
axes[0].set_title("Performance Plot (loglog)")

# Add specific ticks to the x-axis
axes[0].set_xticks([15, 100])  # Specify the ticks you want
axes[0].get_xaxis().set_tick_params(which='both', direction='in')  # Optional: Adjust tick appearance
axes[0].grid(visible=True, which='both', linestyle='--', linewidth=0.5)  # Optional: Add grid for clarity

# Second subplot: Number of Iterations plot
axes[1].plot(df['N'], df["Niter"], "o-")
axes[1].set_xlabel("Matrix Dimension: N")
axes[1].set_ylabel("Number of Iterations")
axes[1].set_title("Number of Iterations Until Convergence")
axes[1].grid(visible=True, which='both', linestyle='--', linewidth=0.5) 

# Adjust layout to prevent overlap
plt.tight_layout()

# Save the figure to a file (e.g., PNG or PDF)
output_path = "performance_and_convergence_jacobi.png"
plt.savefig(output_path, dpi=100)

# Display the combined plots
plt.show()

#%%

L1 = 32
L2 = 56
L3 = 30*1024

L1 = (L1 * 1024 / 8)**(1/3)
L2 = (L2 * 1024 / 8)**(1/3)
L3 = (L3 * 1024 / 8)**(1/3)

# Load the VTK file
filename = "poisson_gs_serial.dat"
path = "/home/max/Documents/DTU/HighPerformanceComputing/hpc_assignments/assignment2/"
file_gs = path+filename

column_names = ["gs","alloc_time", "init_time", "compute_time", "N3_times_Niter","Niter"]

df_gs = pd.read_csv(file_gs,header=None,names=column_names)

df_gs = df_gs.drop(columns=df_gs.columns[0])

df_gs['N'] = (df_gs["N3_times_Niter"] / df_gs["Niter"])**(1/3)
df_gs['memory'] = (df_gs["N3_times_Niter"] / df_gs["Niter"]) * 8 / 1024

# Create a single figure with 1 row and 2 columns
fig, axes = plt.subplots(1, 2, figsize=(8, 4))  # Adjust figsize as needed

# First subplot: Performance plot
perf = df["Niter"] / df["compute_time"]
axes[0].loglog(df['N'], perf, "o-",label="Jacobi")
perf_gs = df_gs["Niter"] / df_gs["compute_time"]
axes[0].loglog(df_gs['N'], perf_gs, "o-",label="Gauss-Seidel")
axes[0].set_xlabel("Matrix Dimension: N")
axes[0].set_ylabel(r"Performance $\left[\dfrac{\text{Iterations}}{second} \right]$")
axes[0].set_title("Performance Plot (loglog)")
axes[0].plot()
#axes[0].vlines([L1, L2, L3], 10, 10**6, linestyle="--", color="red", label="Cache Levels")
axes[0].loglog(df['N'],10**7.7*df['N']**(-3),"--",label=r"$\mathcal{O}(N^{-3})$")
axes[0].legend(fontsize=11)

# Manually set x-axis ticks and labels
manual_ticks = [10,20,40,60,100,140]
manual_labels = [f"{tick:.0f}" for tick in manual_ticks]  # Format as integers
axes[0].set_xticks(manual_ticks)
axes[0].set_xticklabels(manual_labels, rotation=45)  # Rotate labels for clarity
axes[0].grid(visible=True, which='both', linestyle='--', linewidth=0.5) 

# Second subplot: Number of Iterations plot
axes[1].plot(df['N'], df["Niter"], "o-",label="Jacobi")
axes[1].plot(df_gs['N'], df_gs["Niter"], "o-",label="Gauss-Seidel")
axes[1].plot(df['N'], 0.15*df['N']**2 * np.log(df['N']), "--",label=r"$\mathcal{O}(N^2 \log(N))$")
axes[1].set_xlabel("Matrix Dimension: N")
axes[1].set_ylabel("Number of Iterations")
axes[1].set_title("Number of Iterations Until Convergence")
axes[1].legend(fontsize=11)
axes[1].grid(visible=True, which='both', linestyle='--', linewidth=0.5) 

# Adjust layout to prevent overlap
plt.tight_layout()

# Save the figure to a file (e.g., PNG or PDF)
output_path = "performance_and_convergence_gauss_seidel.png"
plt.savefig(output_path, dpi=100)

# Display the combined plots
plt.show()


