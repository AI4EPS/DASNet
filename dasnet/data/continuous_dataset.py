import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
import fsspec
from scipy.signal import butter, sosfiltfilt
from torchvision.transforms import functional as F

def normalize(data, pclip=90):
    clipVal = np.percentile(np.absolute(data), pclip)
    norm_data = np.clip(data, -clipVal, clipVal)
    norm_data = (norm_data + clipVal) / (2 * clipVal) * 255
    return norm_data

class DASNet__ContinuousDataset(Dataset):
    def __init__(self, hdf5_files, key_path, target_size=(948, 4000), pred_object='default', channel_range=[7400, 10245]):
        self.hdf5_files = hdf5_files
        self.fs = None
        self.keypath = key_path
        self.target_size = target_size
        self.pred_object = pred_object
        self.channel_start, self.channel_end = channel_range

    def __len__(self):
        return len(self.hdf5_files)

    def __getitem__(self, idx):
        if self.fs == None:
            if self.keypath is not None:
                self.fs = fsspec.filesystem("gcs", token=self.keypath)
            else:
                self.fs = fsspec.filesystem("gcs")
        
        if self.pred_object == 'default':
            file_name = self.hdf5_files[idx]
            try:
                with self.fs.open(file_name, "rb") as f:
                    with h5py.File(f, "r") as hf:
                        strain = hf['Acquisition/Raw[0]/RawData'][:, self.channel_start:self.channel_end]
                        das_time = hf['Acquisition/Raw[0]/RawDataTime'][:] / 1e6
                        dt = das_time[1] - das_time[0]
            except Exception as e:
                raise RuntimeError(f"Failed to read file {file_name}: {e}")
            
        elif self.pred_object == 'OOI':
            file_name = self.hdf5_files.iloc[idx]['File Name']
            try:
                with h5py.File(self.hdf5_files.iloc[idx]['File Path'], 'r') as file:
                    strain = file['Acquisition/Raw[0]/RawData'][:, :]
                    das_time = file['Acquisition/Raw[0]/RawDataTime'][:] / 1e6
                    dt = das_time[1] - das_time[0]
            except Exception as e:
                raise RuntimeError(f"Failed to read file {file_name}: {e}")
            
            num_points = 47500
            new_num_points = int(47500 * 2 / 5.1)

            x_old = np.linspace(0, 95000, num_points)
            x_new = np.linspace(0, 95000, new_num_points)

            nearest_indices = np.round(np.interp(x_new, x_old, np.arange(num_points))).astype(int)
            nearest_indices = np.clip(nearest_indices, 0, num_points - 1)

            strain = strain[nearest_indices, :]
            strain = strain[7154:9999,:].T
            
        else:
            file_name = self.hdf5_files.iloc[idx]['File Name']
            try:
                with h5py.File(self.hdf5_files.iloc[idx]['File Path'], 'r') as file:
                    strain = file['Acquisition/Raw[0]/RawData'][:, self.channel_start:self.channel_end]
                    das_time = file['Acquisition/Raw[0]/RawDataTime'][:] / 1e6
                    dt = das_time[1] - das_time[0]
            except Exception as e:
                raise RuntimeError(f"Failed to read file {file_name}: {e}")

        # make sure the matrix shape is 12000x2845
        channel_numbers = self.channel_end - self.channel_start
        target_shape = (12000, channel_numbers)
        if strain.shape[0] < target_shape[0]:
            padding = np.zeros((target_shape[0] - strain.shape[0], strain.shape[1]), dtype=strain.dtype)
            strain = np.vstack([strain, padding])
        elif strain.shape[0] > target_shape[0]:
            strain = strain[:target_shape[0], :]

        strain_rate = np.diff(strain, axis=0)
        strain_rate = np.vstack([np.zeros((1, strain_rate.shape[1]), dtype=strain_rate.dtype), strain_rate])

        freq1, freq2, freq3 = 2, 30, 10
        sos1 = butter(4, [freq1 * 2 * dt, freq2 * 2 * dt], output='sos', btype='bandpass')
        sos2 = butter(4, freq3 * 2 * dt, output='sos', btype='high')

        strain_filt1 = sosfiltfilt(sos1, strain, axis=0)
        strain_filt2 = sosfiltfilt(sos2, strain, axis=0)

        rgb_input = np.zeros((12000, channel_numbers, 3), dtype=np.float32)
        rgb_input[:, :, 0] = normalize(strain_rate)
        rgb_input[:, :, 1] = normalize(strain_filt1)
        rgb_input[:, :, 2] = normalize(strain_filt2)

        image = torch.from_numpy(rgb_input).permute(2, 1, 0)

        # resize image
        target_height, target_width = self.target_size
        image = F.resize(image, (target_height, target_width))

        return image, file_name