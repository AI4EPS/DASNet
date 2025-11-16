#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
单文件绘制脚本
读取一个 detections CSV（含 x_min,y_min,x_max,y_max,score,label,source_file），
只针对指定的 source_file 绘制所有框（可选按 label 过滤），不做对比。

用法示例：
python draw_single_file.py \
  --detections_csv /home/chun/DASNet/analysis/merged_output.csv \
  --processed_csv /home/chun/DASNet/temp_eval/miss_T.csv \
  --source_file MBARI_F8_STU_GL20m_OCP5m_FS200Hz_2023-10-27T174910Z.h5 \
  --label 6 \
  --h5-dataset "Acquisition/Raw[0]/RawData" \
  --time-slice 7400 10245 \
  --fs 200 \
  --outdir /nfs2/group/chun/img_single/
"""

import os
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from scipy.signal import butter, filtfilt
from argparse import ArgumentParser


def bandpass_filter(data, fs, lowcut=2.0, highcut=10.0, order=4):
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="band")
    return filtfilt(b, a, data, axis=0)


def normalize(data):
    data = data - np.mean(data, axis=0, keepdims=True)
    data = data / (np.std(data, axis=0, keepdims=True) + 1e-6)
    return np.clip(data, -3, 3)


def main():
    ap = ArgumentParser()
    ap.add_argument("--detections_csv", required=True, help="包含检测结果的CSV（单份）")
    ap.add_argument("--processed_csv", required=True, help="File Path / File Name 映射CSV")
    ap.add_argument("--source_file", required=True, help="只绘制该 source_file 下的所有检测")
    ap.add_argument("--label", type=int, default=None, help="可选：只保留该label")
    ap.add_argument("--fs", type=int, default=200, help="采样率（Hz）")
    ap.add_argument("--h5-dataset", default="Acquisition/Raw[0]/RawData", help="H5数据集路径")
    ap.add_argument("--time-slice", nargs=2, type=int, default=[7400, 10245], help="时间切片 [start, end)")
    ap.add_argument("--scale-x", type=float, default=2.0, help="bbox X 轴缩放（与训练/导出的坐标系匹配）")
    ap.add_argument("--scale-y", type=float, default=2.0, help="bbox Y 轴缩放（与训练/导出的坐标系匹配）")
    ap.add_argument("--lowcut", type=float, default=2.0, help="带通低频 (Hz)")
    ap.add_argument("--highcut", type=float, default=10.0, help="带通高频 (Hz)")
    ap.add_argument("--order", type=int, default=4, help="带通阶数")
    ap.add_argument("--outdir", required=True, help="输出图片目录")
    ap.add_argument("--cmap", default="seismic", help="imshow 颜色图")
    ap.add_argument("--vlim", nargs=2, type=float, default=[-3, 3], help="imshow vmin vmax")
    ap.add_argument("--dpi", type=int, default=200, help="保存DPI")
    ap.add_argument("--show-score", action="store_true", help="在框上方标注score")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # 读取映射
    df_map = pd.read_csv(args.processed_csv)
    file_map = dict(zip(df_map["File Name"], df_map["File Path"]))

    # 读取检测结果
    det = pd.read_csv(args.detections_csv)

    # 统一成 .h5（如果原表是 .h5.csv）
    if "source_file" in det.columns:
        det["source_file"] = det["source_file"].str.replace(".h5.csv", ".h5", regex=False)
    else:
        raise ValueError("detections_csv 缺少 'source_file' 列")

    # 过滤到目标文件
    src = args.source_file
    if src not in file_map:
        raise FileNotFoundError(f"{src} 未在 processed_csv 中找到路径映射")

    det_file = det[det["source_file"] == src]
    if args.label is not None and "label" in det_file.columns:
        det_file = det_file[det_file["label"] == args.label]

    if det_file.empty:
        raise ValueError(f"{src} 在给定条件下没有任何检测（label={args.label}）。")

    h5_path = file_map[src]
    if not os.path.exists(h5_path):
        raise FileNotFoundError(f"H5 文件不存在：{h5_path}")

    # 读取数据
    t0, t1 = args.time_slice
    with h5py.File(h5_path, "r") as f:
        raw = f[args.h5_dataset][:, t0:t1]  # shape: (time, channels)

    # 滤波 + 归一化
    data = bandpass_filter(raw, args.fs, args.lowcut, args.highcut, args.order)
    data = normalize(data)
    data = data.T  # (channels, time)

    # 画图
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(
        data,
        aspect="auto",
        extent=[0, 12000, 0, data.shape[0]],  # 与原代码保持一致的时间/通道范围
        cmap=args.cmap,
        vmin=args.vlim[0],
        vmax=args.vlim[1],
        origin="lower"
    )
    plt.colorbar(im, ax=ax, label="Normalized amplitude")

    # 绘制该文件下所有框
    sx, sy = args.scale_x, args.scale_y
    required_cols = {"x_min", "y_min", "x_max", "y_max"}
    if not required_cols.issubset(set(det_file.columns)):
        raise ValueError(f"detections_csv 缺少必要列：{required_cols}")

    for _, row in det_file.iterrows():
        x0 = row["x_min"] * sx
        y0 = row["y_min"] * sy
        w = (row["x_max"] - row["x_min"]) * sx
        h = (row["y_max"] - row["y_min"]) * sy

        rect = patches.Rectangle(
            (x0, y0),
            w,
            h,
            linewidth=1.2,
            edgecolor="r",
            facecolor="none",
            alpha=0.9
        )
        ax.add_patch(rect)

        if args.show_score and "score" in det_file.columns:
            ax.text(x0, y0 + h + 12, f"{row['score']:.2f}", color="r", fontsize=7)

    ax.set_xlim(0, 12000)
    ax.set_ylim(0, data.shape[0])
    ax.set_xlabel("Time (samples)")
    ax.set_ylabel("Channel")
    title_label = f"  (label={args.label})" if args.label is not None else ""
    ax.set_title(f"{src}{title_label}")
    plt.tight_layout()

    out_path = os.path.join(args.outdir, f"{os.path.splitext(src)[0]}_single.png")
    plt.savefig(out_path, dpi=args.dpi)
    plt.close()

    print(f"[✓] 已保存：{out_path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# """
# 单/多文件绘制脚本
# - 当提供 --source_file 时：只绘制该文件下的所有检测
# - 当不提供 --source_file 时：遍历 detections_csv 中的所有 source_file 逐个绘制
# """

# import os
# import h5py
# import numpy as np
# import pandas as pd
# import matplotlib.pyplot as plt
# import matplotlib.patches as patches
# from scipy.signal import butter, filtfilt
# from argparse import ArgumentParser
# from tqdm import tqdm


# def bandpass_filter(data, fs, lowcut=2.0, highcut=10.0, order=4):
#     nyq = 0.5 * fs
#     b, a = butter(order, [lowcut / nyq, highcut / nyq], btype="band")
#     return filtfilt(b, a, data, axis=0)


# def normalize(data):
#     data = data - np.mean(data, axis=0, keepdims=True)
#     data = data / (np.std(data, axis=0, keepdims=True) + 1e-6)
#     return np.clip(data, -3, 3)


# def plot_one(src, det, file_map, args):
#     """对单个 source_file 进行读取、绘图与保存。"""
#     det_file = det[det["source_file"] == src]
#     if args.label is not None and "label" in det_file.columns:
#         det_file = det_file[det_file["label"] == args.label]

#     if det_file.empty:
#         print(f"[跳过] {src}: 在给定条件下没有任何检测（label={args.label}）。")
#         return

#     if src not in file_map:
#         print(f"[跳过] {src}: 未在 processed_csv 中找到路径映射")
#         return
#     h5_path = file_map[src]
#     if not os.path.exists(h5_path):
#         print(f"[缺失] H5 文件不存在：{h5_path}")
#         return

#     # 读取数据
#     t0, t1 = args.time_slice
#     with h5py.File(h5_path, "r") as f:
#         raw = f[args.h5_dataset][:, t0:t1]  # shape: (time, channels)

#     # 滤波 + 归一化
#     data = bandpass_filter(raw, args.fs, args.lowcut, args.highcut, args.order)
#     data = normalize(data)
#     data = data.T  # (channels, time)

#     # 画图
#     fig, ax = plt.subplots(figsize=(10, 6))
#     im = ax.imshow(
#         data,
#         aspect="auto",
#         extent=[0, 12000, 0, data.shape[0]],  # 与已有标注坐标系保持一致
#         cmap=args.cmap,
#         vmin=args.vlim[0],
#         vmax=args.vlim[1],
#         origin="lower"
#     )
#     plt.colorbar(im, ax=ax, label="Normalized amplitude")

#     # 绘制该文件下所有框
#     sx, sy = args.scale_x, args.scale_y
#     required_cols = {"x_min", "y_min", "x_max", "y_max"}
#     if not required_cols.issubset(set(det_file.columns)):
#         print(f"[跳过] {src}: detections_csv 缺少必要列：{required_cols}")
#         plt.close()
#         return

#     for _, row in det_file.iterrows():
#         x0 = row["x_min"] * sx
#         y0 = row["y_min"] * sy
#         w = (row["x_max"] - row["x_min"]) * sx
#         h = (row["y_max"] - row["y_min"]) * sy

#         rect = patches.Rectangle(
#             (x0, y0),
#             w,
#             h,
#             linewidth=1.2,
#             edgecolor="r",
#             facecolor="none",
#             alpha=0.9
#         )
#         ax.add_patch(rect)

#         if args.show_score and "score" in det_file.columns:
#             ax.text(x0, y0 + h + 12, f"{row['score']:.2f}", color="r", fontsize=7)

#     ax.set_xlim(0, 12000)
#     ax.set_ylim(0, data.shape[0])
#     ax.set_xlabel("Time (samples)")
#     ax.set_ylabel("Channel")
#     title_label = f"  (label={args.label})" if args.label is not None else ""
#     ax.set_title(f"{src}{title_label}")
#     plt.tight_layout()

#     out_name = f"{os.path.splitext(src)[0]}_single.png"
#     out_path = os.path.join(args.outdir, out_name)
#     plt.savefig(out_path, dpi=args.dpi)
#     plt.close()
#     print(f"[✓] 已保存：{out_path}")


# def main():
#     ap = ArgumentParser()
#     ap.add_argument("--detections_csv", required=True, help="包含检测结果的CSV（单份）")
#     ap.add_argument("--processed_csv", required=True, help="File Path / File Name 映射CSV")
#     ap.add_argument("--source_file", default=None, help="只绘制该 source_file；若不提供则绘制所有")
#     ap.add_argument("--label", type=int, default=None, help="可选：只保留该label")
#     ap.add_argument("--fs", type=int, default=200, help="采样率（Hz）")
#     ap.add_argument("--h5-dataset", default="Acquisition/Raw[0]/RawData", help="H5数据集路径")
#     ap.add_argument("--time-slice", nargs=2, type=int, default=[7400, 10245], help="时间切片 [start, end)")
#     ap.add_argument("--scale-x", type=float, default=2.0, help="bbox X 轴缩放（与训练/导出的坐标系匹配）")
#     ap.add_argument("--scale-y", type=float, default=2.0, help="bbox Y 轴缩放（与训练/导出的坐标系匹配）")
#     ap.add_argument("--lowcut", type=float, default=2.0, help="带通低频 (Hz)")
#     ap.add_argument("--highcut", type=float, default=10.0, help="带通高频 (Hz)")
#     ap.add_argument("--order", type=int, default=4, help="带通阶数")
#     ap.add_argument("--outdir", required=True, help="输出图片目录")
#     ap.add_argument("--cmap", default="seismic", help="imshow 颜色图")
#     ap.add_argument("--vlim", nargs=2, type=float, default=[-3, 3], help="imshow vmin vmax")
#     ap.add_argument("--dpi", type=int, default=100, help="保存DPI")
#     ap.add_argument("--show-score", action="store_true", help="在框上方标注score")
#     args = ap.parse_args()

#     os.makedirs(args.outdir, exist_ok=True)

#     # 读取映射
#     df_map = pd.read_csv(args.processed_csv)
#     file_map = dict(zip(df_map["File Name"], df_map["File Path"]))

#     # 读取检测结果
#     det = pd.read_csv(args.detections_csv)
#     if "source_file" not in det.columns:
#         raise ValueError("detections_csv 缺少 'source_file' 列")
#     det["source_file"] = det["source_file"].str.replace(".h5.csv", ".h5", regex=False)

#     # 选择目标文件集合
#     if args.source_file:
#         sources = [args.source_file]
#     else:
#         # 去重后排序；仅保留有路径映射且本地存在的文件
#         unique_sources = sorted(det["source_file"].unique().tolist())
#         sources = []
#         for s in unique_sources:
#             p = file_map.get(s)
#             if p and os.path.exists(p):
#                 sources.append(s)
#             else:
#                 # 安静跳过缺失映射/文件
#                 if not p:
#                     print(f"[跳过] {s}: 无路径映射")
#                 else:
#                     print(f"[缺失] {s}: 文件不存在 -> {p}")

#     if not sources:
#         raise ValueError("没有可绘制的 source_file。")

#     # 逐个绘制
#     for src in tqdm(sources):
#         plot_one(src, det, file_map, args)


# if __name__ == "__main__":
#     main()
