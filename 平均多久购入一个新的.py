import os
import pandas as pd
import numpy as np
from glob import glob
import matplotlib.pyplot as plt

# --- Configuration ---
SNAPSHOT_DIR = r""
project_name = 'Pudgy Penguins'

# --- Load and Merge All Snapshots ---
snapshot_files = sorted(glob(os.path.join(SNAPSHOT_DIR, "*.csv")))
snapshot_dates = [os.path.basename(f).split("_")[-1].replace(".csv", "") for f in snapshot_files]

holder_dfs = []
for file in snapshot_files:
    date = os.path.basename(file).split("_")[-1].replace(".csv", "")
    df = pd.read_csv(file)
    df['date'] = date
    holder_dfs.append(df[['holder', 'date']])
all_holders = pd.concat(holder_dfs, ignore_index=True)

# --- Ensure Date Format and Sort ---
all_holders['date'] = pd.to_datetime(all_holders['date'])
all_holders = all_holders.sort_values(['holder', 'date'])

# --- Remove Duplicate Purchases on the Same Day for the Same Holder ---
unique_purchases = all_holders.drop_duplicates(['holder', 'date'])

# --- Calculate Average Interval for Each Address ---
def compute_avg_interval(group):
    if len(group) < 2:
        return np.nan
    diffs = group['date'].sort_values().diff().dt.days.dropna()
    return diffs.mean()

avg_intervals = unique_purchases.groupby('holder').apply(compute_avg_interval)
avg_intervals = avg_intervals.dropna()

# --- Overall Statistics ---
overall_avg_interval = avg_intervals.mean()
summary = avg_intervals.describe()

print(f"Overall average interval between new NFT purchases: {overall_avg_interval:.2f} days")
print("Summary statistics for average purchase interval per address:")
print(summary)

# --- Optional: Save Results ---
avg_intervals.to_csv(os.path.join(SNAPSHOT_DIR, f"{project_name}_avg_purchase_interval_per_address.csv"))

# --- Optional: Visualize Distribution ---
plt.figure(figsize=(10,6))
plt.hist(avg_intervals, bins=30, color='skyblue', edgecolor='black')
plt.xlabel('Average Days Between Purchases')
plt.ylabel('Number of Addresses')
plt.title('Distribution of Average Purchase Interval per Address')
plt.tight_layout()
plt.savefig(os.path.join(SNAPSHOT_DIR, f"{project_name}_avg_purchase_interval_hist.png"), dpi=300)
plt.show()
