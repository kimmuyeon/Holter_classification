import numpy as np 
import os
import wfdb
from fastdtw import fastdtw
from scipy.spatial.distance import euclidean
import h5py

def load_record(record_path, leads):
    rec = wfdb.rdrecord(record_path)
    sig = rec.p_signal[:, leads]
    print(f"Loaded {record_path}: total_samples={sig.shape[0]}, channels={sig.shape[1]}") # 너무 짧은(빈) 데이터 확인용
    return sig, int(rec.fs)

def segment_signal(sig, fs, seg_sec):
    seg_len = int(fs * seg_sec)
    if sig.shape[0] < seg_len:
        # 너무 짧아서 세그먼트로 쪼갤 수 없으면 None 반환
        return None
    n_win = sig.shape[0] // seg_len
    windows = [sig[i*seg_len:(i+1)*seg_len] for i in range(n_win)]
    return np.stack(windows)

def dtw_distance(a, b):
    # a, b: 1D or flattened arrays
    dist, _ = fastdtw(a, b, dist=euclidean)
    return dist

def filter_by_dtw(windows, template, threshold):
    selected = []
    for w in windows:
        if dtw_distance(w.flatten(), template.flatten()) <= threshold:
            selected.append(w)
    return np.array(selected)

def preprocess_raw_signals(rec_dir, npy_dir, leads):
    os.makedirs(npy_dir, exist_ok=True)
    for fname in os.listdir(rec_dir):
        if not fname.endswith('.hea'):
            continue
        record_name = fname[:-4]
        npz_path = os.path.join(npy_dir, record_name + '.npz')
        if os.path.exists(npz_path):
            continue  # 이미 전처리되어 있으면 건너뛰기

        # WFDB에서 로드
        rec = wfdb.rdrecord(os.path.join(rec_dir, record_name))
        sig = rec.p_signal[:, leads]  # [total_samples, len(leads)]
        fs  = int(rec.fs)

        # 압축된 npz로 저장
        np.savez_compressed(npz_path, sig=sig, fs=fs)
        print(f"Saved preprocessed record: {npz_path}")


def load_h5(record_h5_path):
    """.h5 파일에서 sig, fs를 읽어 반환"""
    with h5py.File(record_h5_path, 'r') as f:
        sig = f['sig'][:]       # [total_samples, channels]
        fs  = int(f['fs'][()])  # 파일에 fs를 저장해 두셨다면
    return sig, fs


def preprocess_to_h5(
    rec_dir: str,
    h5_dir: str,
    leads: list[int]
):
    os.makedirs(h5_dir, exist_ok=True)

    for fname in os.listdir(rec_dir):
        if not fname.endswith('.hea'):
            continue
        rec_name = fname[:-4]
        h5_path = os.path.join(h5_dir, rec_name + '.h5')
        if os.path.exists(h5_path):
            # 이미 처리된 파일 건너뛰기
            continue

        # WFDB 로드
        rec = wfdb.rdrecord(os.path.join(rec_dir, rec_name))
        sig = rec.p_signal[:, leads].astype('float32')  # (T, C)
        fs  = int(rec.fs)

        # HDF5로 저장
        with h5py.File(h5_path, 'w') as f:
            f.create_dataset('sig', data=sig, compression='gzip')
            f.create_dataset('fs',  data=fs)
        print(f"[H5] Saved {h5_path}: shape={sig.shape}, fs={fs}")


# 샘플 데이터로 시도해보기
'''
if __name__ == '__main__':
    preprocess_to_h5(
        rec_dir = r'C:\Users\andus\Desktop\my_work\BOAZ 23기\ADV\boaz_sample2\projects\data\originals',
        h5_dir = r'C:\Users\andus\Desktop\my_work\BOAZ 23기\ADV\boaz_sample2\projects\data\denoised',
        leads = [0,1,2]
    )
'''