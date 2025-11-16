#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.signal import butter, filtfilt
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from tqdm import tqdm

# ========== 参数设置 ==========
csv1 = "/home/chun/DASNet/analysis/merged_output_missT.csv"
csv2 = "/home/chun/DASNet/analysis/merged_output.csv"
processed_csv = "/home/chun/DASNet/temp_eval/miss_T.csv"
label_target = 6
output_dir = "/nfs2/group/chun/img_missT/comparision/"
max_workers = 8  # 并行进程数，可根据机器核心数调整

os.makedirs(output_dir, exist_ok=True)

# ========== 滤波与归一化函数 ==========
def bandpass_filter(data, fs, lowcut=2.0, highcut=10.0, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="band")
    return filtfilt(b, a, data, axis=0)

def normalize(data):
    data = data - np.mean(data, axis=0, keepdims=True)
    data = data / (np.std(data, axis=0, keepdims=True) + 1e-6)
    return np.clip(data, -3, 3)

# ========== 读取数据 ==========
df1 = pd.read_csv(csv1)
df2 = pd.read_csv(csv2)
processed = pd.read_csv(processed_csv)

df1["source_file"] = df1["source_file"].str.replace(".h5.csv", ".h5", regex=False)
df2["source_file"] = df2["source_file"].str.replace(".h5.csv", ".h5", regex=False)
valid_files = set(processed["File Name"])

df1 = df1[(df1["label"] == label_target) & (df1["source_file"].isin(valid_files))]
df2 = df2[(df2["label"] == label_target) & (df2["source_file"].isin(valid_files))]
print(f"File1 保留 {len(df1)} 条, File2 保留 {len(df2)} 条")

file_map = dict(zip(processed["File Name"], processed["File Path"]))

def extract_time(s):
    try:
        t_str = s.split("_FS200Hz_")[1].replace(".h5", "")
        t_str = t_str.replace("T", " ").replace("Z", "")
        return datetime.strptime(t_str, "%Y-%m-%d %H%M%S")
    except Exception:
        return None

df1["timestamp"] = df1["source_file"].apply(extract_time)
df2["timestamp"] = df2["source_file"].apply(extract_time)

# ========== 单文件绘图函数 ==========
def process_one_file(source, df1_sub, df2_sub, file_path, output_dir, fs=200):
    import matplotlib
    matplotlib.use("Agg")  # 后端防止并行绘图冲突

    try:
        with h5py.File(file_path, "r") as f:
            raw_data = f["Acquisition/Raw[0]/RawData"][:, 7400:10245]
        data = bandpass_filter(raw_data, fs)
        data = normalize(data)
        data = data.T  # (channels, time)

        fig, ax = plt.subplots(figsize=(10, 6))
        im = ax.imshow(
            data,
            aspect="auto",
            extent=[0, 12000, 0, 2845],
            cmap="seismic",
            vmin=-3,
            vmax=3,
            origin="lower"
        )
        plt.colorbar(im, ax=ax, label="Normalized amplitude")

        # file1 红色矩形
        for _, row in df1_sub.iterrows():
            rect = patches.Rectangle(
                (row["x_min"] * 2, row["y_min"] * 2),
                (row["x_max"] - row["x_min"]) * 2,
                (row["y_max"] - row["y_min"]) * 2,
                linewidth=1.2,
                edgecolor="r",
                facecolor="none",
                alpha=0.8
            )
            ax.add_patch(rect)
            ax.text(row["x_min"] * 2, row["y_max"] * 2 + 20,
                    f"{row['score']:.2f}", color="r", fontsize=7)

        # file2 蓝色矩形
        for _, row in df2_sub.iterrows():
            rect = patches.Rectangle(
                (row["x_min"] * 2, row["y_min"] * 2),
                (row["x_max"] - row["x_min"]) * 2,
                (row["y_max"] - row["y_min"]) * 2,
                linewidth=1.2,
                edgecolor="b",
                facecolor="none",
                alpha=0.8
            )
            ax.add_patch(rect)
            ax.text(row["x_min"] * 2, row["y_max"] * 2 + 20,
                    f"{row['score']:.2f}", color="b", fontsize=7)

        ax.set_xlim(0, 12000)
        ax.set_ylim(0, 2845)
        ax.set_xlabel("Time (samples)")
        ax.set_ylabel("Channel")
        ax.set_title(source)
        plt.tight_layout()

        save_path = os.path.join(output_dir, f"{source.replace('.h5', '')}_compare.png")
        plt.savefig(save_path, dpi=200)
        plt.close()
        return f"[OK] {source}"

    except Exception as e:
        return f"[ERROR] {source}: {e}"

# ========== 并行处理 ==========
fs = 200
tasks = []
sources = sorted(set(df1["source_file"]).union(df2["source_file"]))

with ProcessPoolExecutor(max_workers=max_workers) as executor:
    for source in sources:
        if source not in file_map:
            continue
        file_path = file_map[source]
        if not os.path.exists(file_path):
            continue
        df1_sub = df1[df1["source_file"] == source]
        df2_sub = df2[df2["source_file"] == source]
        tasks.append(executor.submit(process_one_file, source, df1_sub, df2_sub, file_path, output_dir, fs))

    for future in tqdm(as_completed(tasks), total=len(tasks), desc="绘图中"):
        print(future.result())

print(f"[✓] 所有并行绘图完成，结果已保存到 {output_dir}")
