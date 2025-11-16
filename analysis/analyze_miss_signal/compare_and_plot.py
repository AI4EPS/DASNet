#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.signal import butter, filtfilt
from tqdm import tqdm
from datetime import datetime

# ========== 参数设置 ==========
csv1 = "/home/chun/DASNet/analysis/merged_output_missT.csv"
csv2 = "/home/chun/DASNet/analysis/merged_output.csv"   # 第二个结果文件
processed_csv = "/home/chun/DASNet/temp_eval/miss_T.csv"  # File Path,File Name
label_target = 6
output_dir = "/nfs2/group/chun/img_missT/comparision/"

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

# ========== 筛选 ==========
df1 = df1[(df1["label"] == label_target) & (df1["source_file"].isin(valid_files))]
df2 = df2[(df2["label"] == label_target) & (df2["source_file"].isin(valid_files))]

print(f"File1 保留 {len(df1)} 条, File2 保留 {len(df2)} 条")

# ========== 建立 File Name → Path 映射 ==========
file_map = dict(zip(processed["File Name"], processed["File Path"]))

# ========== 提取时间戳 ==========
def extract_time(s):
    try:
        t_str = s.split("_FS200Hz_")[1].replace(".h5", "")
        t_str = t_str.replace("T", " ").replace("Z", "")
        return datetime.strptime(t_str, "%Y-%m-%d %H%M%S")
    except Exception:
        return None

df1["timestamp"] = df1["source_file"].apply(extract_time)
df2["timestamp"] = df2["source_file"].apply(extract_time)

# ========== 绘制每个文件 ==========
fs = 200  # 采样率
for source in tqdm(sorted(set(df1["source_file"]).union(df2["source_file"]))):
    if source not in file_map:
        print(f"[跳过] {source} 未在 processed_csv 中找到路径")
        continue

    file_path = file_map[source]
    if not os.path.exists(file_path):
        print(f"[缺失] 文件不存在: {file_path}")
        continue

    try:
        # 读取底图数据
        with h5py.File(file_path, "r") as f:
            raw_data = f["Acquisition/Raw[0]/RawData"][:, 7400:10245]

        # 滤波与归一化
        data = bandpass_filter(raw_data, fs)
        data = normalize(data)

        # 转置以便 X=时间, Y=通道
        data = data.T  # shape: (channels, time)

        # 绘图
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

        # 绘制矩形框（注意 x, y * 2）
        for _, row in df1[df1["source_file"] == source].iterrows():
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

        for _, row in df2[df2["source_file"] == source].iterrows():
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
    except Exception as e:
        print(f"[错误] {source}: {e}")

print(f"[✓] 所有结果已保存到 {output_dir}")
