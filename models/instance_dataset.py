import torch
from torch.utils.data import Dataset
import wfdb
import numpy as np
import h5py

import h5py
import torch
from torch.utils.data import Dataset, get_worker_info

class ECGBagInstanceDataset(Dataset):
    def __init__(self, meta, leads, seg_sec):
        """
        meta   : List of (h5_path, starts, fs)
        leads  : List/tuple of channel indices, e.g. [0,1,2]
        seg_sec: Segment length in seconds
        """
        self.meta    = meta
        self.leads   = tuple(leads)
        self.seg_sec = seg_sec

        # (bag_idx, start_sample, fs) 를 미리 매핑
        self.idx_map = [
            (bag_idx, start, fs)
            for bag_idx, (_, starts, fs) in enumerate(meta)
            for start in starts
        ]

        # 워커별로 한 번만 열어 둘 파일 핸들 저장소
        self._files = None

    def __len__(self):
        return len(self.idx_map)

    def _init_worker_files(self):
        # 각 워커 프로세스가 첫 사용 시에만 호출됩니다.
        self._files = []
        for rec_path, _, _ in self.meta:
            f = h5py.File(
                rec_path, 'r',
                libver='latest',    # SWMR 모드 활성화
                swmr=True
            )
            self._files.append(f)

    def __getitem__(self, idx):
        # 워커 프로세스별로 파일 핸들 초기화
        if self._files is None:
            self._init_worker_files()

        bag_idx, start, fs = self.idx_map[idx]
        f = self._files[bag_idx]

        # 필요한 구간 + 채널만 슬라이스 (전체 읽기 방지)
        seg_len = int(fs * self.seg_sec)
        window = f['ecg'][start : start + seg_len, self.leads]  # shape: [seg_len, len(leads)]

        # flatten 해서 1D 벡터로 반환
        return torch.tensor(window.flatten(), dtype=torch.float32)
