import torch
from torch.utils.data import Dataset
import wfdb
import numpy as np
import h5py

class ECGBagInstanceDataset(Dataset):
    def __init__(self, meta, leads, seg_sec):
        self.meta = meta
        self.leads = leads
        self.seg_sec = seg_sec

        self.idx_map = []
        for bag_idx, (rec_path, starts, fs) in enumerate(meta):
            for s in starts:
                self.idx_map.append((bag_idx, s, fs))

    def __len__(self):
        return len(self.idx_map)

    def __getitem__(self, idx):
        bag_idx, start, fs = self.idx_map[idx]
        rec_ref, _, _ = self.meta[bag_idx]
        
        with h5py.File(rec_ref, 'r') as f:
            sig = f['sig'][:]    
        seg_len = int(fs*self.seg_sec)
        window = sig[start:start+seg_len]   # [seg_len, C]
        return torch.tensor(window.flatten(), dtype=torch.float32)