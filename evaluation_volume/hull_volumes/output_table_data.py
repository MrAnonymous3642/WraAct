"""This is for organizing the data of volume to output a table for the paper."""

import numpy as np
import time

time_start = time.perf_counter()


all_data = {
    2: {3: {}, 4: {}},
    3: {3: {}, 4: {}},
    4: {3: {}, 4: {}},
}

print("[INFO] Collecting data for 2d, 3d, and 4d polytopes with 3^n constraints...")

for dim in [2, 3, 4]:  # dimension
    for n in [3]:  # constraints number indicator
        for method in [
            "single_sigmoid",
            "prima_sigmoid",
            "our_sigmoid",
            "single_tanh",
            "prima_tanh",
            "our_tanh",
            "single_maxpool",
            "prima_maxpool",
            "our_maxpool_dlp",
        ]:
            file_path = f"./polytopes_{dim}d_{n}_{method}_volume.txt"
            with open(file_path, "r") as f:
                lines = f.readlines()

            lines = [line.replace("\n", "").split("\t") for line in lines]
            data = np.asarray([[float(item) for item in line] for line in lines])

            cal_time = data[:, 0]
            constraints_num = data[:, 1]
            volume = data[:, -2] / data[:, -1]

            data = np.stack([cal_time, volume, constraints_num], axis=1)
            data_mean = np.mean(data, axis=0)
            data_std = np.std(data, axis=0)
            all_data[dim][n][method] = [data_mean.tolist(), data_std.tolist()]

print(
    f"[INFO] Data collection completed in "
    f"{time.perf_counter() - time_start:.2f} seconds."
)
print(f"[INFO] Data saved to polytopes_data_dict.txt")

print(f"[INFO] Outputting the table...")
print()

n = 3
methods = [
    "prima_sigmoid",
    "our_sigmoid",
    "prima_tanh",
    "our_tanh",
    "prima_maxpool",
    "our_maxpool_dlp",
]
# 0: time, 1: volume, 2: constraints number
print(" & ".join(["Dimension"] + methods) + " \\\\")
for item_idx, item_name in enumerate(
    ["Runtime (s)", "Estimated Volume", "Number of Constraints"]
):
    print(f"{item_name} \\\\")
    for dim in {2, 3, 4}:
        row_str = f"{dim} & "
        for method in methods:
            mean_data = all_data[dim][n][method][0][item_idx]
            std_data = all_data[dim][n][method][1][item_idx]
            if item_idx in {0, 1}:
                row_str += f"{mean_data:.6f} & $\\pm$ & ({std_data:.6f}) & "
            else:
                row_str += f"{mean_data:>8.2f} & $\\pm$ & {f"({std_data:.2f})":<8} & "
        row_str = row_str[:-3] + " \\\\"
        print(row_str)

print()
print(f"[INFO] Done in {time.perf_counter() - time_start:.2f} seconds")
