import os

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

plt.style.use('tableau-colorblind10')

os.chdir(os.path.join(Path(__file__).parent, "..", "results"))
df = pd.read_csv("results_blk.dat", delimiter=' ',
                 names=["Memory [KiB]", "Compute [Mflop/s]", "Name", "Block Size"],
                 header=None)

matmult_blk_df = df[df["Name"] == "matmult_blk"]

fig = plt.figure()
for block_size in matmult_blk_df['Block Size'].unique():
    subset = matmult_blk_df[matmult_blk_df['Block Size'] == block_size]
    plt.semilogx(subset['Memory [KiB]'], subset['Compute [Mflop/s]'], label=f"{block_size}", marker='o')
plt.axvline(x=32, color='r', linestyle="dashed", label='L1d')
plt.axvline(x=256, color='r', linestyle="dashed", label='L2')
plt.axvline(x=30 * 1024, color='r', linestyle="dashed", label='L3')
plt.xlabel('Memory [KiB]')
plt.ylabel('Compute [Mflop/s]')
plt.legend()
plt.xlim([matmult_blk_df["Memory [KiB]"].min(), matmult_blk_df["Memory [KiB]"].max()])
plt.ylim([0, 1.05 * matmult_blk_df["Compute [Mflop/s]"].max()])
fig.tight_layout()
plt.show()

matmult_goodblock_size_df = df[df["Block Size"].isin([0, 100])]

fig = plt.figure()
for name in matmult_goodblock_size_df['Name'].unique():
    subset = matmult_goodblock_size_df[matmult_goodblock_size_df['Name'] == name]
    plt.semilogx(subset['Memory [KiB]'], subset['Compute [Mflop/s]'],
                 label=name if name != "matmult_blk" else "matmult_blk (bs = 100)", marker='o')
plt.axvline(x=32, color='r', linestyle="dashed", label='L1d')
plt.axvline(x=256, color='r', linestyle="dashed", label='L2')
plt.axvline(x=30 * 1024, color='r', linestyle="dashed", label='L3')
plt.xlabel('Memory [KiB]')
plt.ylabel('Compute [Mflop/s]')
plt.legend()
plt.xlim([matmult_goodblock_size_df["Memory [KiB]"].min(), matmult_goodblock_size_df["Memory [KiB]"].max()])
plt.ylim([0, 1.05 * matmult_goodblock_size_df["Compute [Mflop/s]"].max()])
fig.tight_layout()
plt.show()
