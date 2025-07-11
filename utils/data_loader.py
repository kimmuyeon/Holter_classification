# utils/data_loader.py

import os, re, random, numpy as np, pandas as pd
from .preprocess import load_record, segment_signal, load_h5
from tqdm import tqdm
import h5py
from sklearn.model_selection import train_test_split

def load_data(
    data_dir,             # e.g. './data'
    leads,
    seg_sec,
    h5_dir,
    labels_csv="psvt_labeling.csv",
    id_col="person_id",
    label_col="is_psvt",
    test_ratio=0.2,
    random_seed=42
):
    # 1) 레이블 읽기 (data_dir/labels.csv)
    label_path = os.path.join(data_dir, labels_csv)
    df = pd.read_csv(label_path, usecols=[id_col, label_col])
    df[id_col] = df[id_col].astype(str)
    df_map = df.set_index(id_col)[label_col].to_dict()

    # 2) 레코드(.hea) 파일 리스트 (data_dir/originals/*.hea)
    # rec_dir = os.path.join(data_dir, "originals")
    rec_dir = data_dir
    all_heas = [f for f in os.listdir(rec_dir) if f.endswith(".hea")]

    # 3) 이름 끝 숫자 추출 & 레이블 매핑
    records, labels = [], []
    for hea in tqdm(all_heas, desc="Mapping records to labels"):
        rec_name = hea[:-4]  # 확장자 제거
        m = re.search(r"(\d+)$", rec_name)
        if not m: 
            continue
        rec_id = m.group(1)
        if rec_id in df_map:
            rec_path = os.path.join(rec_dir, rec_name)
            records.append((rec_path, rec_id))
            labels.append(int(df_map[rec_id]))

    h5_files = os.listdir(h5_dir)
    valid = []
    for rec_tuple, lab in zip(records, labels):
        rec_path, rec_id = rec_tuple
        rec_id = str(rec_id)
        pattern = re.compile(rf".*{re.escape(rec_id)}.*\.h5$")
        matches = [f for f in h5_files if pattern.match(f)]
        if not matches:
            print(f"skipping (no h5 match for ID): {rec_id}")
            continue
        h5_path = os.path.join(h5_dir, matches[0])
        valid.append((rec_path, lab))
    if not valid:
        raise FileNotFoundError(f"No records with H5 files found in {h5_dir}")
    recs, labs = zip(*valid)

    recs_train, recs_test, labs_train, labs_test = train_test_split(
        recs, labs,
        test_size=test_ratio,
        random_state=random_seed,
        stratify=labs
    )

    # 세그먼트 생성
    def build_meta(rec_list, label_list, mode):
        metas, labs = [], []
        for rec_name, lab in tqdm(zip(rec_list, label_list), desc=f"Building {mode} meta", total=len(rec_list)):
            try:
                sig, fs = sig, fs = load_h5(h5_path)
            except Exception:
                continue
            starts = []
            segs = segment_signal(sig, fs, seg_sec)
            if segs is None:
                continue
            seg_len = int(fs*seg_sec)
            n_win = sig.shape[0] // seg_len
            if n_win < 1:
                print(f"Skipping {rec_name}, too short (total samples: {sig.shape[0]})")
                continue
            starts = [i*seg_len for i in range(n_win)]
            rec_id = rec_path.split(os.sep)[-1]
            rec_id = str(rec_id)
            pattern = re.compile(rf".*{re.escape(rec_id)}.*\.h5$")
            matches = [f for f in h5_files if pattern.match(f)]
            if not matches:
                continue
            h5_path = os.path.join(h5_dir, matches[0])
            metas.append((h5_path, starts, fs))
            labs.append(lab)
        return metas, labs

    train_meta, train_labels = build_meta(recs_train, labs_train, "train")
    test_meta,  test_labels  = build_meta(recs_test, labs_test, "test")

    
    return list(train_meta), list(train_labels), test_meta, test_labels


