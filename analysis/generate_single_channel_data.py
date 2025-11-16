import h5py
import os, logging
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, spectrogram, welch
from scipy.signal import medfilt as mf
from scipy.ndimage import median_filter
import pandas as pd
import argparse
from PIL import Image

import numpy as np

import time
from tqdm import tqdm

detect_name_path = '/home/chun/SeaFOAM_chun/tools/data_name_2025_04_08.csv'
filt_name = 'MBARI_F8_STU_GL20m_OCP5m_FS200Hz_2024-05-08T133622Z.h5'
save_path = 'single_channel'
Channel_idx = 8800

detect_files = pd.read_csv(detect_name_path)
filtered_detect_files = detect_files[detect_files['File Name'].str.startswith(filt_name)]

if not os.path.exists(save_path):
    os.makedirs(save_path)

for ifile in tqdm(range(len(filtered_detect_files))):
    start_time = time.time()
    save_name = filtered_detect_files.iloc[ifile]['File Name']

    try:
        with h5py.File(filtered_detect_files.iloc[ifile]['File Path'], 'r') as file:
            raw_data = file['Acquisition/Raw[0]/RawData'][:]
            raw_data_time = file['Acquisition/Raw[0]/RawDataTime'][:]
    except Exception as e:
        logging.error(f"Failed to read file {filtered_detect_files.iloc[ifile]['File Path']}: {e}")
        continue

    das_time = raw_data_time/1e6
    dt = das_time[1]-das_time[0]

    p = save_path
    p = os.path.join(p, save_name+f'_ch{Channel_idx}')

    data = signal = np.sum(raw_data[:, Channel_idx-5:Channel_idx+5], axis=1)
    np.save(p + '.npy', data)