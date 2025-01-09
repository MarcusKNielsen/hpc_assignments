import matplotlib.pyplot as plt
import re
import os

file_paths = ['results/gprofng_results_no_opt.dat',
              'results/gprofng_results_-O3-funroll-all-loops.dat']

# Function to parse the .dat file and aggregate L1 and L2 cache statistics by PERM case
def parse_cache_data_by_perm(file_path):
    perm_cache_data = {}
    current_perm = None

    with open(file_path, 'r') as file:
        lines = file.readlines()
        for line in lines:
            # Check for a new PERM case
            perm_match = re.match(r"^PERM = (\w+)", line)
            if perm_match:
                current_perm = perm_match.group(1)
                perm_cache_data[current_perm] = {
                    'L1 Hits': 0,
                    'L1 Misses': 0,
                    'L2 Hits': 0,
                    'L2 Misses': 0
                }
                continue

            # Match lines with relevant data (e.g., skip headers and blank lines)
            if current_perm:
                match = re.match(
                    r"^\s*([\d\.]+)\s+[\d\.]+\s+[\d\.]+\s+[\d\.]+\s+(\d+)\s+[\d\.]+\s+(\d+)\s+[\d\.]+\s+(\d+)\s+[\d\.]+\s+(\d+)\s+[\d\.]+\s+(.+)$",
                    line
                )
                if match:
                    perm_cache_data[current_perm]['L1 Hits'] += int(match.group(2))
                    perm_cache_data[current_perm]['L1 Misses'] += int(match.group(3))
                    perm_cache_data[current_perm]['L2 Hits'] += int(match.group(4))
                    perm_cache_data[current_perm]['L2 Misses'] += int(match.group(5))

    return perm_cache_data

def get_title_from_filepath(file_path):
    base_name = os.path.basename(file_path)
    name_without_ext = base_name.replace('.dat', '').split('_', 2)[-1]
    return name_without_ext.replace("_", " ")

def plot_cache_stats(perm_cache_data, title):
    perms = list(perm_cache_data.keys())
    l1_hits = [perm_cache_data[perm]['L1 Hits'] for perm in perms]
    l1_misses = [perm_cache_data[perm]['L1 Misses'] for perm in perms]
    l2_hits = [perm_cache_data[perm]['L2 Hits'] for perm in perms]
    l2_misses = [perm_cache_data[perm]['L2 Misses'] for perm in perms]

    x = range(len(perms))
    bar_width = 0.1

    plt.figure(figsize=(6, 6))
    plt.bar(x, l1_hits, width=bar_width, label='L1 Hits', color='blue')
    plt.bar([i + bar_width for i in x], l1_misses, width=bar_width, label='L1 Misses', color='orange')
    plt.bar([i + 2 * bar_width for i in x], l2_hits, width=bar_width, label='L2 Hits', color='green')
    plt.bar([i + 3 * bar_width for i in x], l2_misses, width=bar_width, label='L2 Misses', color='red')

    plt.ylabel('Cache Counts')
    plt.title(f'{title}')
    plt.xticks([i + 1.5 * bar_width for i in x], perms, rotation=45, ha='right')
    plt.yscale('log') 
    plt.legend()
    plt.tight_layout()

    save_dir = "figures"
    plot_filename = os.path.join(save_dir, f"plot_{title}.png")
    plt.savefig(plot_filename)
    print(f"Plot saved as {plot_filename}")
    plt.close()

if __name__ == "__main__":
    for file_path in file_paths:
        perm_cache_data = parse_cache_data_by_perm(file_path)

        plot_title = get_title_from_filepath(file_path)
        plot_cache_stats(perm_cache_data, plot_title)
