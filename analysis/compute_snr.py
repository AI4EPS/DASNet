#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import sys
import ast
import math
import argparse
import shutil
from glob import glob
from typing import Dict, Tuple, List

import h5py
import numpy as np
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser(
        description="批量为检测CSV计算SNR（到时点后1000点/前1000点 RMS 比值），并回写到原文件（保留.bak备份）。"
    )
    p.add_argument("--detections_dir", required=True,
                   help="目录，内含多份检测结果 CSV（每行含 peak_points 与 source_file 字段）")
    p.add_argument("--mapping_csv", required=True,
                   help="两列映射CSV，表头为 FileName,FilePath，用于将 source_file 映射到真实HDF5路径")
    p.add_argument("--pre_points", type=int, default=1000,
                   help="到时点前窗口点数（噪声），默认1000")
    p.add_argument("--post_points", type=int, default=1000,
                   help="到时点后窗口点数（信号），默认1000")
    p.add_argument("--encoding", default="utf-8",
                   help="读取/写回CSV的编码，默认utf-8")
    return p.parse_args()


def load_name_to_path(mapping_csv: str) -> Dict[str, str]:
    mdf = pd.read_csv(mapping_csv)
    if "File Name" not in mdf.columns or "File Path" not in mdf.columns:
        raise ValueError("mapping_csv 必须包含列: File Name, File Path")
    out = {}
    for _, r in mdf.iterrows():
        fn = str(r["File Name"]).strip()
        fp = str(r["File Path"]).strip()
        if fn:
            out[fn] = fp
    return out


def safe_literal_list(s: str) -> List[List[float]]:
    """
    解析 peak_points 的字符串，返回 [[x, y], ...]
    """
    if isinstance(s, list):
        return s
    if not isinstance(s, str):
        return []
    s = s.strip()
    if not s:
        return []
    try:
        val = ast.literal_eval(s)
        if isinstance(val, list):
            return val
        return []
    except Exception:
        return []


def transform_xy(x: float, y: float) -> Tuple[int, int]:
    """
    变换为真实点坐标：[2x, 2y + 7400]，取整
    """
    tx = int(round(2 * float(x)))
    ty = int(round(2 * float(y) + 7400))
    return tx, ty


def rms(a: np.ndarray) -> float:
    if a.size == 0:
        return np.nan
    return float(np.sqrt(np.mean(np.square(a.astype(np.float64)))))


def snr_from_timeseries(ts: np.ndarray, ix: int, pre: int, post: int) -> Tuple[float, float]:
    """
    在单通道时间序列 ts 上，以 ix 为到时点：
    - 噪声窗口: [ix-pre, ix)
    - 信号窗口: [ix, ix+post)
    返回 (snr_ratio, snr_db)。边界不足则返回 (nan, nan)
    """
    start_n = ix - pre
    end_n = ix
    start_s = ix
    end_s = ix + post
    if start_n < 0 or end_s > ts.shape[0]:
        return (np.nan, np.nan)

    noise = ts[start_n:end_n]
    signal = ts[start_s:end_s]
    r_noise = rms(noise)
    r_signal = rms(signal)
    if r_noise == 0 or np.isnan(r_noise) or np.isnan(r_signal):
        return (np.nan, np.nan)
    ratio = r_signal / r_noise
    db = 20.0 * math.log10(ratio) if ratio > 0 else np.nan
    return (ratio, db)


def get_layout(ds_shape: Tuple[int, int]) -> Tuple[int, int, bool]:
    """
    返回 (time_len, chan_len, time_first)
    - 若 shape[0] >= shape[1]，视为 (time, channel)
    - 否则视为 (channel, time)
    """
    if len(ds_shape) != 2:
        raise ValueError(f"RawData 不是二维数组，实际形状: {ds_shape}")
    if ds_shape[0] >= ds_shape[1]:
        return ds_shape[0], ds_shape[1], True
    else:
        return ds_shape[1], ds_shape[0], False


def read_channel_slice(h5f: h5py.File, chan_idx: int, t_start: int, t_end: int) -> np.ndarray:
    """
    只读取某通道在 [t_start:t_end) 的切片，返回 1D ndarray
    自动识别 (time, channel) 与 (channel, time)
    """
    ds = h5f["Acquisition/Raw[0]/RawData"]
    time_len, chan_len, time_first = get_layout(ds.shape)

    if chan_idx < 0 or chan_idx >= chan_len:
        return np.array([], dtype=np.float32)
    if t_start < 0 or t_end > time_len or t_start >= t_end:
        return np.array([], dtype=np.float32)

    if time_first:
        # (time, channel)
        # 注意 h5py 切片: 先时间，再通道
        arr = ds[t_start:t_end, chan_idx]
    else:
        # (channel, time)
        arr = ds[chan_idx, t_start:t_end]
    return np.array(arr, dtype=np.float32)


def compute_row_snr(
    row: pd.Series,
    name2path: Dict[str, str],
    pre_pts: int,
    post_pts: int,
    h5_cache: Dict[str, h5py.File]
) -> Tuple[float, float, float, float, int]:
    """
    对一条记录（一个实例）：
    - 解析 peak_points -> 变换坐标 -> 对每个通道/到时点计算 SNR
    - 汇总得到 mean/median/max 以及 dB median
    - 返回 (snr_mean, snr_median, snr_max, snr_db_median, count)
    """
    # 1) 解析 peak_points
    pts = safe_literal_list(row.get("peak_points", ""))
    if not pts:
        return (np.nan, np.nan, np.nan, np.nan, 0)

    # 2) source_file -> FileName（去掉末尾 .csv）
    src = str(row.get("source_file", "")).strip()
    if not src:
        return (np.nan, np.nan, np.nan, np.nan, 0)
    if src.lower().endswith(".csv"):
        file_name = src[:-4]
    else:
        file_name = src

    # 3) 找到 H5 路径
    h5_path = name2path.get(file_name)
    if not h5_path or not os.path.exists(h5_path):
        return (np.nan, np.nan, np.nan, np.nan, 0)

    # 4) 打开/复用 h5 文件
    h5f = h5_cache.get(h5_path)
    if h5f is None or not h5f.id.valid:
        if h5f is not None:
            try:
                h5f.close()
            except Exception:
                pass
        h5f = h5py.File(h5_path, "r")
        h5_cache[h5_path] = h5f

    # 5) 获取时间长度用于边界检查
    ds = h5f["Acquisition/Raw[0]/RawData"]
    time_len, chan_len, _ = get_layout(ds.shape)

    # 6) 遍历到时点
    snr_list = []
    db_list = []
    seen = set()
    for p in pts:
        if not (isinstance(p, (list, tuple)) and len(p) == 2):
            continue
        tx, ty = transform_xy(p[0], p[1])
        # 去重（相同通道/时间）
        key = (tx, ty)
        if key in seen:
            continue
        seen.add(key)

        # 边界检查
        if tx - pre_pts < 0 or tx + post_pts > time_len:
            continue
        if ty < 0 or ty >= chan_len:
            continue

        # 只读取需要的窗口
        ts = read_channel_slice(h5f, ty, tx - pre_pts, tx + post_pts)
        if ts.size != pre_pts + post_pts:
            continue

        ratio, db = snr_from_timeseries(ts, pre_pts, pre_pts, post_pts)
        if not np.isnan(ratio):
            snr_list.append(ratio)
        if not np.isnan(db):
            db_list.append(db)

    if len(snr_list) == 0:
        return (np.nan, np.nan, np.nan, np.nan, 0)

    snr_arr = np.array(snr_list, dtype=np.float64)
    db_arr = np.array(db_list, dtype=np.float64) if len(db_list) else np.array([], dtype=np.float64)
    snr_mean = float(np.mean(snr_arr))
    snr_median = float(np.median(snr_arr))
    snr_max = float(np.max(snr_arr))
    snr_db_median = float(np.median(db_arr)) if db_arr.size > 0 else np.nan
    return (snr_mean, snr_median, snr_max, snr_db_median, int(len(snr_arr)))


def process_csv_file(csv_path: str,
                     name2path: Dict[str, str],
                     pre_pts: int,
                     post_pts: int,
                     encoding: str):
    print(f"[INFO] Processing: {csv_path}")
    df = pd.read_csv(csv_path, encoding=encoding)

    # 需要的列检查
    required_cols = {"peak_points", "source_file"}
    missing = [c for c in required_cols if c not in df.columns]
    if missing:
        print(f"[WARN] 缺少列 {missing}，跳过：{csv_path}")
        return

    # H5 打开缓存（同一CSV里复用）
    h5_cache: Dict[str, h5py.File] = {}

    # 逐行计算
    out_cols = {
        "snr_mean": [],
        "snr_median": [],
        "snr_max": [],
        "snr_db_median": [],
        "snr_count": [],
    }
    for _, row in df.iterrows():
        s_mean, s_med, s_max, s_dbm, s_cnt = compute_row_snr(
            row, name2path, pre_pts, post_pts, h5_cache
        )
        out_cols["snr_mean"].append(s_mean)
        out_cols["snr_median"].append(s_med)
        out_cols["snr_max"].append(s_max)
        out_cols["snr_db_median"].append(s_dbm)
        out_cols["snr_count"].append(s_cnt)

    # 关闭本CSV所用到的 H5
    for f in h5_cache.values():
        try:
            f.close()
        except Exception:
            pass

    # 追加列并写回（做备份）
    for k, v in out_cols.items():
        df[k] = v

    bak_path = csv_path + ".bak"
    if not os.path.exists(bak_path):
        shutil.copy2(csv_path, bak_path)
    df.to_csv(csv_path, index=False, encoding=encoding)
    print(f"[OK] 写回完成，已生成备份：{bak_path}")


def main():
    args = parse_args()
    name2path = load_name_to_path(args.mapping_csv)

    # 收集 detections_dir 下的 CSV
    csv_files = sorted(
        [p for p in glob(os.path.join(args.detections_dir, "*.csv")) if os.path.isfile(p)]
    )
    if not csv_files:
        print(f"[WARN] 在 {args.detections_dir} 下未找到CSV")
        return

    for csv_path in csv_files:
        try:
            process_csv_file(csv_path, name2path, args.pre_points, args.post_points, args.encoding)
        except Exception as e:
            print(f"[ERROR] 处理 {csv_path} 出错：{e}")

    print("[DONE] 全部完成。")


if __name__ == "__main__":
    main()

#python compute_snr.py --detections_dir /path/to/detections_csvs --mapping_csv /path/to/FileName_FilePath_map.csv --pre_points 1000 --post_points 1000
