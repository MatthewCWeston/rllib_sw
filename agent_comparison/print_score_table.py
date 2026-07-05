import numpy as np
from tabulate import tabulate

def print_score_table(scores):
  sorted_keys, v = sorted([k for k in scores.keys()]), list(scores.values())
  min_elo, max_elo = np.min(v)-1e-6, np.max(v)
  gap = max_elo - min_elo
  table_data = [[k, f'\033[38;2;{int(255*min(1, 2*(1-(scores[k]-min_elo)/gap)))};{int(255*min(1, 2*(scores[k]-min_elo)/gap))};0m{scores[k]}\033[0m'] for k in sorted_keys]
  floatfmt = ".0f" if max_elo > 1000 else ".3f"
  print(tabulate(table_data, headers=["Checkpoint", "Rating"], tablefmt='rounded_outline', floatfmt=floatfmt))