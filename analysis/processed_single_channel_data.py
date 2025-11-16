import os
import numpy as np
from glob import glob
from tqdm import tqdm
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt, spectrogram, welch

# ===== 配置路径 =====
folder = '/home/chun/DASNet/analysis/single_channel'  # 替换为你的npy所在文件夹
output_file = 'merged.npy'       # 输出文件名

# ===== 加载 & 处理所有 .npy 文件 =====
all_files = sorted(glob(os.path.join(folder, '*.npy')))

processed_data_list = []

for file_path in tqdm(all_files, desc="Processing"):
    try:
        data = np.load(file_path)

        freq = 2
        sos = butter(4, freq*2*1/200, output='sos', btype='high')
        data = sosfiltfilt(sos, data)

        # Clip：只保留绝对值在95%分位数以下的数据
        clip_val = np.percentile(np.abs(data), 95)
        data = np.clip(data, -clip_val, clip_val)
        data = (data + clip_val) / (2 * clip_val)

        processed_data_list.append(data)
    except Exception as e:
        print(f"Failed to process {file_path}: {e}")

# ===== 拼接 & 保存 =====
merged = np.concatenate(processed_data_list)
np.save(output_file, merged)
print(f"Merged data saved to: {output_file}")

nperseg = 500
noverlap = 475
fs = 200
signal = merged
pad = nperseg // 2
signal_padded = np.pad(signal, (pad, pad), mode='constant')
print(signal_padded.shape)

frequencies, times, Sxx = spectrogram(signal_padded, fs, nperseg=nperseg, noverlap=noverlap)
Sxx_log = 10 * np.log10(Sxx)

######### only save freq > x Hz
freq_idx = frequencies > 0.1
frequencies = frequencies[freq_idx]
Sxx_log = Sxx_log[freq_idx, :]
#########
plt.figure(figsize=(16, 4))
print(f'vmin: {np.percentile(Sxx_log, 70)}, vmax: {np.percentile(Sxx_log, 99.9)}')
plt.pcolormesh(times-pad/fs, frequencies, Sxx_log, shading='gouraud', cmap='plasma', vmin=np.percentile(Sxx_log, 50), rasterized=True)
plt.ylabel('Frequency (Hz)', fontsize=20)
plt.xlabel('Time (s)', fontsize=20)
# plt.colorbar(label='PSD (dB)')
# plt.yscale('log')
plt.ylim(frequencies[1], 100)#frequencies[1], frequencies[-1]
plt.xlim(0, 60)
plt.xticks(fontsize=20)
plt.yticks(fontsize=20)
# plt.title('Spectrogram')
plt.savefig(
    os.path.join('spe.png'),
    bbox_inches='tight',
    # dpi=300,
    pad_inches=0
)
