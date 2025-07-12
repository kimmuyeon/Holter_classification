import torch, numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from utils.data_loader import load_data
from utils.preprocess    import dtw_distance
from models.attention_ae import AttentionAE
from models.mil_model    import SimpleMIL
from models.instance_dataset import ECGBagInstanceDataset
import os, pickle
import wfdb
import glob
from tqdm import tqdm
from utils.preprocess import preprocess_raw_signals
import h5py
from collections import Counter

# ── 설정값 ──
DATA_DIR      = r'C:\Users\andus\Desktop\my_work\BOAZ 23기\ADV\boaz_sample2\projects\data'
H5_DIR        = os.path.join(DATA_DIR, 'denoised')
LEADS         = [0,1,2]
SEG_SEC       = 10
BATCH_AE      = 64
BATCH_MIL     = 1
LR_AE, LR_MIL = 1e-4, 1e-4
fs = 125
seg_len  = int(fs * SEG_SEC)
EPOCH_AE      = 20
EPOCH_MIL     = 20
ATTN_TH       = 0.5     # attention threshold
DTW_TH        = 1e3     # DTW 거리 threshold (필요 시)
device        = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
test_ratio    = 0.2
random_seed   = 42

os.makedirs('checkpoints', exist_ok=True)
cache_path = 'checkpoints/data_cache.pkl'

if os.path.exists(cache_path):
    print(f"Loading dataset from cache: {cache_path}")
    with open(cache_path, 'rb') as f:
        train_meta, train_labels, test_meta, test_labels = pickle.load(f)
else:
    train_meta, train_labels, test_meta, test_labels = load_data(
        DATA_DIR, LEADS, SEG_SEC, H5_DIR,
        test_ratio=test_ratio, random_seed=random_seed
    )
    with open(cache_path, 'wb') as f:
        pickle.dump((train_meta, train_labels, test_meta, test_labels), f)
    print("Saved train/test meta to cache")

print(device)


filtered_meta, filtered_labels = [], []
for meta, lbl in zip(train_meta, train_labels):
    rec_path, _, _ = meta
    rec_id = os.path.basename(rec_path)
    print(rec_id)
    
    h5_path = os.path.join(H5_DIR, f"{rec_id}")  # _denoised.h5 붙여주기 (연구실 서버에서는)
    
    if not os.path.exists(h5_path):
        matches = glob.glob(os.path.join(H5_DIR, f"{rec_id}*.h5"))
        if matches:
            h5_path = matches[0]
        else:
            print(f"Skipping {rec_id}")
            continue
    filtered_meta.append((h5_path, meta[1], meta[2]))
    filtered_labels.append(lbl)

train_meta, train_labels = filtered_meta, filtered_labels
print('[AE] Train set label distribution')


for lbl, cnt in Counter(train_labels).items():
    print(f"label={lbl}: {cnt} samples")
print()

# ── 2) AE 학습 ──
ae = AttentionAE(input_dim=seg_len*len(LEADS), latent_dim=8).to(device)
opt_ae = torch.optim.Adam(ae.parameters(), lr=LR_AE)
crit_ae= nn.MSELoss()

# 인스턴스 단위 DataLoader
print("train length:", len(train_meta))

ae_ds = ECGBagInstanceDataset(train_meta, LEADS, SEG_SEC)
loader_ae = DataLoader(ae_ds, batch_size=BATCH_AE, shuffle=True)

for batch in tqdm(loader_ae, desc="Loading AE training data"):
    pass

for ep in range(EPOCH_AE):
    ae.train()
    tot=0
    for x in loader_ae:
        x = x.to(device)
        recon, _ = ae(x)
        loss = crit_ae(recon, x)
        opt_ae.zero_grad(); loss.backward(); opt_ae.step()
        tot += loss.item()
    print(f"[AE] Epoch {ep+1}/{EPOCH_AE}  Loss: {tot/len(loader_ae):.4f}")

ae_path = 'checkpoints/ae.pth'
torch.save(ae.state_dict(), ae_path)
print(f"Saved AE model to {ae_path}")

# ── 3) 대표 인스턴스 선택 함수 ──
def select_bag(ae_model, meta):
    rec_path, starts, fs = meta
    # 인스턴스에 대해서 attention score 계산
    insts = []

    for s in starts:
        rec = wfdb.rdrecord(rec_path)
        sig = rec.p_signal[:, LEADS]
        insts.append(sig[s: s+ int(fs * SEG_SEC)]) # [L, C]

    X = torch.tensor(
        np.stack(insts).reshape(len(insts), -1),
        dtype=torch.float32
    ).to(device)

    ae.eval()
    with torch.no_grad():
        _, scores = ae_model(X)           # [N,1]
    mask = (scores.squeeze() >= ATTN_TH).cpu().numpy()
    sel  = bag[mask]
    # DTW 필터링 (옵션)
    # template = bag[0]
    # sel = np.array([w for w in sel if dtw_distance(w.flatten(),template.flatten())<=DTW_TH])
    if len(sel)==0:
        idx = scores.squeeze().argmax().item()
        sel = np.array(insts)[idx:idx+1]
    return sel

# ── 4) MIL 학습 준비 ──
sel_train = [select_bag(ae, b) for b in train_meta]
sel_test  = [select_bag(ae, b) for b in test_meta]

class BagDataset(Dataset):
    def __init__(self, bags, labels):
        self.bags, self.labels = bags, labels
    def __len__(self): return len(self.bags)
    def __getitem__(self,i):
        b = torch.tensor(self.bags[i].reshape(len(self.bags[i]), -1),
                         dtype=torch.float32)
        return b, torch.tensor(self.labels[i], dtype=torch.float32)

loader_mil_train = DataLoader(BagDataset(sel_train, train_labels),
                              batch_size=BATCH_MIL, shuffle=True)
loader_mil_test  = DataLoader(BagDataset(sel_test,  test_labels),
                              batch_size=1, shuffle=False)

# ── 5) MIL 학습 ──
mil = SimpleMIL(input_dim=seg_len*len(LEADS)).to(device)
opt_mil = torch.optim.Adam(mil.parameters(), lr=LR_MIL)
crit_mil= nn.BCELoss()

for ep in range(EPOCH_MIL):
    mil.train(); tot=0
    for bag, y in loader_mil_train:
        bag, y = bag.squeeze(0).to(device), y.to(device)
        pred = mil(bag)
        loss = crit_mil(pred, y) # y를 unsqueeze 해야할까
        opt_mil.zero_grad(); loss.backward(); opt_mil.step()
        tot += loss.item()
    print(f"[MIL] Epoch {ep+1}/{EPOCH_MIL}  Loss: {tot/len(loader_mil_train):.4f}")

# MIL 모델 저장
mil_path = 'checkpoints/mil.pth'
torch.save(mil.state_dict(), mil_path)
print(f"Saved MIL model to {mil_path}")

# ── 6) 평가 ──
mil.eval()
correct=0
for bag, y in loader_mil_test:
    bag = bag.squeeze(0).to(device)
    p   = (mil(bag).item() >= 0.5)
    correct += (p == bool(y.item()))
acc = correct / len(loader_mil_test)
print(f"Test Accuracy: {acc:.4f}")

# AE 로드
ae2 = AttentionAE(input_dim=seg_len * len(LEADS), latent_dim=8).to(device)
ae2.load_state_dict(torch.load(ae_path))
ae2.eval()

# MIL 로드
mil2 = SimpleMIL(input_dim=seg_len * len(LEADS)).to(device)
mil2.load_state_dict(torch.load(mil_path))
mil2.eval()

bag0 = test_bags[0]
sel0 = select_bag(ae2, bag0)
with torch.no_grad():
    bag_tensor = torch.tensor(sel0.reshape(len(sel0), -1),
                              dtype=torch.float32).to(device)
    prob = mil2(bag_tensor).item()
print(f"Test patient #0 predicted positive probability: {prob:.4f}")