import matplotlib.pyplot as plt
import re
import os

file_path = 'results/gprofng_blk_results.dat'

# Function to parse the .dat file and aggregate L1 and L2 cache statistics by BLOCKSIZE
def parse_cache_data_by_blocksize(file_path):
    blocksize_cache_data = {}
    current_blocksize = None

    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            # Check for a new BLOCKSIZE case
            blocksize_match = re.match(r"^BLKSIZE = (\w+)", line)
            if blocksize_match:
                current_blocksize = blocksize_match.group(1)
                blocksize_cache_data[current_blocksize] = {
                    'L1 Hits': 0,
                    'L1 Misses': 0,
                    'L2 Hits': 0,
                    'L2 Misses': 0
                }
                continue

            # Match lines with relevant data (e.g., skip headers and blank lines)
            if current_blocksize:
                match = re.match(
                    r"^\s*([\d\.]+)\s+[\d\.]+\s+[\d\.]+\s+[\d\.]+\s+(\d+)\s+[\d\.]+\s+(\d+)\s+[\d\.]+\s+(\d+)\s+[\d\.]+\s+(\d+)\s+[\d\.]+\s+(.+)$",
                    line
                )
                if match:
                    blocksize_cache_data[current_blocksize]['L1 Hits'] += int(match.group(2))
                    blocksize_cache_data[current_blocksize]['L1 Misses'] += int(match.group(3))
                    blocksize_cache_data[current_blocksize]['L2 Hits'] += int(match.group(4))
                    blocksize_cache_data[current_blocksize]['L2 Misses'] += int(match.group(5))

    return blocksize_cache_data

def plot_percentage_stats(blocksize_cache_data, cache_type):
    blocksizes = list(blocksize_cache_data.keys())
    if cache_type == 'L1':
        hits = [blocksize_cache_data[blocksize]['L1 Hits'] for blocksize in blocksizes]
        misses = [blocksize_cache_data[blocksize]['L1 Misses'] for blocksize in blocksizes]
    elif cache_type == 'L2':
        hits = [blocksize_cache_data[blocksize]['L2 Hits'] for blocksize in blocksizes]
        misses = [blocksize_cache_data[blocksize]['L2 Misses'] for blocksize in blocksizes]

    total_counts = [hits[i] + misses[i] for i in range(len(blocksizes))]
    hit_percentages = [(hits[i] / total_counts[i]) * 100 if total_counts[i] > 0 else 0 for i in range(len(blocksizes))]
    miss_percentages = [(misses[i] / total_counts[i]) * 100 if total_counts[i] > 0 else 0 for i in range(len(blocksizes))]

    x = range(len(blocksizes))

    # Plot for hits
    plt.figure(figsize=(6, 6))
    plt.plot(x, hit_percentages, marker='o',color='blue', label=f'{cache_type} Hits')
    plt.ylabel('Percentage (%)')
    plt.xlabel('BlockSize')
    plt.title(f'{cache_type} Hits')
    plt.xticks(x, blocksizes, rotation=45, ha='right')
    plt.ylim(0, 100)
    plt.tight_layout()
    
    save_dir = "figures"
    plot_filename = os.path.join(save_dir, f"{cache_type}_hits_blk.png")
    plt.savefig(plot_filename)
    print(f"Plot saved as {plot_filename}")
    plt.close()

    # Plot for misses
    plt.figure(figsize=(6, 6))
    plt.plot(x, miss_percentages, marker='o',color='red', label=f'{cache_type} Misses')
    plt.ylabel('Percentage (%)')
    plt.xlabel('BlockSize')
    plt.title(f'{cache_type} Misses')
    plt.xticks(x, blocksizes, rotation=45, ha='right')
    plt.ylim(0, 100)
    plt.tight_layout()
    
    plot_filename = os.path.join(save_dir, f"{cache_type}_misses_blk.png")
    plt.savefig(plot_filename)
    print(f"Plot saved as {plot_filename}")
    plt.close()

if __name__ == "__main__":
    blocksize_cache_data = parse_cache_data_by_blocksize(file_path)

    plot_percentage_stats(blocksize_cache_data, 'L1')
    plot_percentage_stats(blocksize_cache_data, 'L2')
