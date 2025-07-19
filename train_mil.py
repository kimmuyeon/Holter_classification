import os
import random
import time
import h5py
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score, f1_score
 
# =============================================================================
# 1) 파일(레코드) 단위로 세그먼트 샘플링
# =============================================================================
def get_record_segments(directory, max_files=300, samples_per_hour=10, label=0):
    """
    directory: H5 파일들이 저장된 경로
    max_files: 최대 사용할 파일 수
    samples_per_hour: 각 파일에서 시간대별로 샘플링할 세그먼트 개수
    label: 해당 레코드의 레이블 (예: PSVT=1, NON-PSVT=0)
    각 파일에 대해 /segments/ 아래 seg_XXXX에서 숫자를 추출해
      hour = (숫자 // 360) 으로 계산한 후,
      각 시간대별로 최대 samples_per_hour 개의 seg_key를 선택.
    반환: 리스트의 각 원소는 (file_path, label, seg_keys)
           여기서 seg_keys는 해당 파일에서 샘플링된 seg 키들의 리스트.
    """
    records = []
    file_paths = [os.path.join(directory, f) for f in os.listdir(directory) if f.endswith('.h5')]
    file_paths = file_paths[:max_files]
    for fp in file_paths:
        try:
            with h5py.File(fp, 'r') as f:
                seg_keys = list(f['segments'].keys())
        except Exception as e:
            print(f"Error opening {fp}: {e}")
            continue
        hour_dict = {}
        for key in seg_keys:
            try:
                index_num = int(key.split('_')[1])
            except Exception:
                continue
            hour = index_num // 360
            hour_dict.setdefault(hour, []).append(key)
        selected_keys = []
        for hour, keys in hour_dict.items():
            if len(keys) > samples_per_hour:
                selected = random.sample(keys, samples_per_hour)
            else:
                selected = keys
            selected_keys.extend(selected)
        if selected_keys:
            records.append((fp, label, selected_keys))
    return records
 
# =============================================================================
# 2) 파일(record) 단위로 데이터를 분할
# =============================================================================
class RecordSegmentDataset(Dataset):
    def __init__(self, records):
        """
        records: list of tuples (file_path, label, seg_keys)
        """
        self.indices = []  # 각 원소: (file_path, seg_key, label)
        for record in records:
            file_path, label, seg_keys = record
            for key in seg_keys:
                self.indices.append((file_path, key, label))
 
    def __len__(self):
        return len(self.indices)
 
    def __getitem__(self, idx):
        file_path, seg_key, label = self.indices[idx]
        try:
            with h5py.File(file_path, 'r') as f:
                # 10초 세그먼트는 '/segments/<seg_key>/signal'에 저장됨.
                segment = f['segments'][seg_key]['signal'][()]
        except Exception as e:
            print(f"Error loading {file_path}, seg={seg_key}: {e}")
            segment = None
        if segment is None:
            dummy = torch.zeros((3, 1250), dtype=torch.float32)
            return dummy, torch.tensor(label, dtype=torch.long), file_path
        # 가정: 원본 signal shape은 (1250, 3)
        seg_tensor = torch.tensor(segment, dtype=torch.float32)
        # 1D CNN 입력에 맞게 (채널, 길이)로 변경 → (3, 1250)
        seg_tensor = seg_tensor.permute(1, 0)
        # record 식별자(file_path)도 함께 반환
        return seg_tensor, torch.tensor(label, dtype=torch.long), file_path
 
# =============================================================================
# 3) CNN+LSTM 모델 (입력: (batch, 3, 1250))
# =============================================================================
class CNNLSTM(nn.Module):
    def __init__(self, num_classes=2, lstm_hidden_size=128, lstm_num_layers=1):
        super(CNNLSTM, self).__init__()
        # CNN 블록: 6개의 convolution layer
        self.conv1 = nn.Conv1d(in_channels=3, out_channels=32, kernel_size=15, stride=1, padding=7)
        self.bn1   = nn.BatchNorm1d(32)
        self.pool1 = nn.MaxPool1d(kernel_size=2)  # 1250 -> 625
 
        self.conv2 = nn.Conv1d(32, 64, kernel_size=11, stride=1, padding=5)
        self.bn2   = nn.BatchNorm1d(64)
        self.pool2 = nn.MaxPool1d(kernel_size=2)  # 625 -> 312
 
        self.conv3 = nn.Conv1d(64, 64, kernel_size=11, stride=1, padding=5)
        self.bn3   = nn.BatchNorm1d(64)
        self.pool3 = nn.MaxPool1d(kernel_size=2)  # 312 -> 156
 
        self.conv4 = nn.Conv1d(64, 128, kernel_size=7, stride=1, padding=3)
        self.bn4   = nn.BatchNorm1d(128)
        self.pool4 = nn.MaxPool1d(kernel_size=2)  # 156 -> 78
 
        self.conv5 = nn.Conv1d(128, 256, kernel_size=7, stride=1, padding=3)
        self.bn5   = nn.BatchNorm1d(256)
        self.pool5 = nn.MaxPool1d(kernel_size=2)  # 78 -> 39
 
        self.conv6 = nn.Conv1d(256, 256, kernel_size=5, stride=1, padding=2)
        self.bn6   = nn.BatchNorm1d(256)
        self.pool6 = nn.MaxPool1d(kernel_size=2)  # 39 -> 19
 
        self.relu = nn.ReLU()
 
        # LSTM 블록: CNN 출력은 (batch, 256, 19) → LSTM 입력: (batch, 19, 256)
        self.lstm = nn.LSTM(input_size=256, hidden_size=lstm_hidden_size, num_layers=lstm_num_layers, batch_first=True)
 
        # 최종 분류를 위한 Fully Connected layer
        self.fc = nn.Linear(lstm_hidden_size, num_classes)
 
    def forward(self, x):
        # x: (batch, 3, 1250)
        x = self.conv1(x)  # (batch, 32, 1250)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.pool1(x)  # (batch, 32, 625)
 
        x = self.conv2(x)  # (batch, 64, 625)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.pool2(x)  # (batch, 64, 312)
 
        x = self.conv3(x)  # (batch, 64, 312)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.pool3(x)  # (batch, 64, 156)
 
        x = self.conv4(x)  # (batch, 128, 156)
        x = self.bn4(x)
        x = self.relu(x)
        x = self.pool4(x)  # (batch, 128, 78)
 
        x = self.conv5(x)  # (batch, 256, 78)
        x = self.bn5(x)
        x = self.relu(x)
        x = self.pool5(x)  # (batch, 256, 39)
 
        x = self.conv6(x)  # (batch, 256, 39)
        x = self.bn6(x)
        x = self.relu(x)
        x = self.pool6(x)  # (batch, 256, 19)
 
        # LSTM 입력에 맞게 차원 변환: (batch, seq_len, feature_dim)
        x = x.permute(0, 2, 1)  # (batch, 19, 256)
 
        lstm_out, (hn, cn) = self.lstm(x)
        # 마지막 타임스텝의 출력 사용
        last_out = lstm_out[:, -1, :]  # (batch, lstm_hidden_size)
        logits = self.fc(last_out)
        return logits
 
# =============================================================================
# 4) 학습/검증 함수 (tqdm 진행바 포함) + loss 기록 및 그래프 출력
# =============================================================================
def train_and_validate(model, train_loader, val_loader, device, epochs=5):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    model.to(device)
    # 에폭별 손실 및 정확도 기록
    train_loss_list, val_loss_list = [], []
    train_acc_list, val_acc_list = [], []
 
    for epoch in range(epochs):
        # -- Training --
        model.train()
        total_loss = 0
        correct = 0
        total = 0
        train_pbar = tqdm(train_loader, desc=f"[Epoch {epoch+1}/{epochs}][Train]", ncols=100)
        for batch in train_pbar:
            if len(batch) == 3:
                signals, labels, _ = batch
            else:
                signals, labels = batch
            signals = signals.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(signals)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * signals.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += signals.size(0)
            batch_loss = loss.item()
            batch_acc = (predicted == labels).float().mean().item()
            train_pbar.set_postfix({"loss": f"{batch_loss:.4f}", "acc": f"{batch_acc:.4f}"})
        epoch_train_loss = total_loss / total
        epoch_train_acc = correct / total
        train_loss_list.append(epoch_train_loss)
        train_acc_list.append(epoch_train_acc)
 
        # -- Validation --
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        val_pbar = tqdm(val_loader, desc=f"[Epoch {epoch+1}/{epochs}][Valid]", ncols=100)
        with torch.no_grad():
            for batch in val_pbar:
                if len(batch) == 3:
                    signals, labels, _ = batch
                else:
                    signals, labels = batch
                signals = signals.to(device)
                labels = labels.to(device)
                outputs = model(signals)
                loss = criterion(outputs, labels)
                val_loss += loss.item() * signals.size(0)
                _, predicted = torch.max(outputs, 1)
                val_correct += (predicted == labels).sum().item()
                val_total += signals.size(0)
        epoch_val_loss = val_loss / val_total
        epoch_val_acc = val_correct / val_total
        val_loss_list.append(epoch_val_loss)
        val_acc_list.append(epoch_val_acc)
 
        print(f"Epoch [{epoch+1}/{epochs}] Train Loss: {epoch_train_loss:.4f} | Train Acc: {epoch_train_acc:.4f} | " +
              f"Val Loss: {epoch_val_loss:.4f} | Val Acc: {epoch_val_acc:.4f}")
 
    # 학습 및 검증 손실/정확도 그래프 출력
    epochs_range = range(1, epochs+1)
    plt.figure(figsize=(10,4))
    plt.subplot(1,2,1)
    plt.plot(epochs_range, train_loss_list, label='Train Loss')
    plt.plot(epochs_range, val_loss_list, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss vs. Epoch')
    plt.legend()
 
    plt.subplot(1,2,2)
    plt.plot(epochs_range, train_acc_list, label='Train Acc')
    plt.plot(epochs_range, val_acc_list, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. Epoch')
    plt.legend()
    plt.tight_layout()
    plt.show()
 
    return train_loss_list, val_loss_list, train_acc_list, val_acc_list
 
# =============================================================================
# 5) 테스트 함수: record 단위로 segment 예측 결과를 모아 평가 (confusion matrix plot & metric 출력)
# =============================================================================
def test_model(model, test_loader, device):
    """
    배치마다 record id를 수집하고,
    각 segment의 예측 결과(확률)를 record별로 축적한 후,
    record 단위 평균 확률을 기준으로 최종 예측을 수행.
    """
    criterion = nn.CrossEntropyLoss()
    model.eval()
    model.to(device)
    segment_total_loss = 0
    segment_count = 0
 
    # record 단위로 예측 결과와 확률, label 저장
    record_preds = {}
    record_probs = {}  # 각 segment의 class 1 확률
    record_labels = {}
 
    test_pbar = tqdm(test_loader, desc="[Test]", ncols=100)
    with torch.no_grad():
        for signals, labels, record_ids in test_pbar:
            signals = signals.to(device)
            labels = labels.to(device)
            outputs = model(signals)
            loss = criterion(outputs, labels)
            segment_total_loss += loss.item() * signals.size(0)
            segment_count += signals.size(0)
 
            _, predicted = torch.max(outputs, 1)
            probs = nn.functional.softmax(outputs, dim=1)[:, 1]  # class 1 확률
 
            # 배치 내 각 샘플에 대해 record 단위로 저장
            for i in range(len(record_ids)):
                rec_id = record_ids[i]
                if rec_id not in record_preds:
                    record_preds[rec_id] = []
                    record_probs[rec_id] = []
                    record_labels[rec_id] = labels[i].item()
                record_preds[rec_id].append(predicted[i].item())
                record_probs[rec_id].append(probs[i].item())
 
    test_loss = segment_total_loss / segment_count
 
    # record 단위 최종 예측 결정 (평균 확률 기준, threshold 0.5)
    final_preds = []
    final_labels = []
    final_probs = []
    for rec_id in record_preds:
        avg_prob = sum(record_probs[rec_id]) / len(record_probs[rec_id])
        final_pred = 1 if avg_prob >= 0.5 else 0
        final_preds.append(final_pred)
        final_labels.append(record_labels[rec_id])
        final_probs.append(avg_prob)
 
    # Confusion Matrix (record 단위) 계산 및 출력
    cm = confusion_matrix(final_labels, final_preds)
    print("Confusion Matrix (Record-wise):")
    print(cm)
 
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=[0, 1])
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Record-wise Confusion Matrix")
    plt.show()
 
    # ROC Curve와 AUROC (record 단위)
    auroc = roc_auc_score(final_labels, final_probs)
    fpr, tpr, thresholds = roc_curve(final_labels, final_probs)
    plt.figure()
    plt.plot(fpr, tpr, label=f"ROC curve (AUROC = {auroc:.4f})")
    plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Record-wise ROC Curve")
    plt.legend()
    plt.show()
 
    # 추가 metric: Accuracy, Precision, Recall, F1 Score, Specificity
    accuracy = accuracy_score(final_labels, final_preds)
    precision = precision_score(final_labels, final_preds)
    recall = recall_score(final_labels, final_preds)
    f1 = f1_score(final_labels, final_preds)
    tn, fp, fn, tp = cm.ravel()  # binary confusion matrix
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
 
    print("Record-wise Evaluation Metrics:")
    print(f"Accuracy    : {accuracy:.4f}")
    print(f"Precision   : {precision:.4f}")
    print(f"Recall      : {recall:.4f}")
    print(f"F1 Score    : {f1:.4f}")
    print(f"AUROC       : {auroc:.4f}")
    print(f"Specificity : {specificity:.4f}")
 
    return test_loss, accuracy
 
# =============================================================================
# 6) 메인 실행: 파일 단위 (record-wise) 분리 → Dataset 생성 → 모델 학습 및 테스트
# =============================================================================
if __name__ == "__main__":
    # GPU 0 사용 (가능한 경우)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("[INFO] Using device:", device)
    random.seed(42)
    torch.manual_seed(42)
 
    # PSVT와 NON-PSVT 데이터 경로 설정 (필요에 따라 경로 수정)
    psvt_dir = "/home/coder/workspace/data/10s_segment_finlal/psvt"
    not_psvt_dir = "/home/coder/workspace/data/10s_segment_finlal/not_psvt"
 
    # 각 디렉토리에서 record 단위로 세그먼트 샘플링 (최대 300개 파일, 시간대별 10개 샘플)
    psvt_records = get_record_segments(psvt_dir, max_files=300, samples_per_hour=10, label=1)
    not_psvt_records = get_record_segments(not_psvt_dir, max_files=300, samples_per_hour=10, label=0)
 
    # 두 record 리스트 합쳐서 섞은 뒤 train/val/test 분할 (예: 70:15:15)
    all_records = psvt_records + not_psvt_records
    random.shuffle(all_records)
    total_records = len(all_records)
    train_count = int(0.70 * total_records)
    val_count = int(0.15 * total_records)
    test_count = total_records - train_count - val_count
 
    train_records = all_records[:train_count]
    val_records   = all_records[train_count: train_count + val_count]
    test_records  = all_records[train_count + val_count:]
    print(f"[INFO] Records - Train: {len(train_records)}, Val: {len(val_records)}, Test: {len(test_records)}")
 
    # Dataset 및 DataLoader 생성
    train_dataset = RecordSegmentDataset(train_records)
    val_dataset = RecordSegmentDataset(val_records)
    test_dataset = RecordSegmentDataset(test_records)
    print(f"[INFO] Segments - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
 
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
 
    # 모델 생성: CNN+LSTM 사용
    model = CNNLSTM(num_classes=2, lstm_hidden_size=128, lstm_num_layers=1)
    train_and_validate(model, train_loader, val_loader, device, epochs=5)
 
    # 최종 테스트 및 그래프 출력 (record-wise 평가)
    test_loss, test_acc = test_model(model, test_loader, device)
    print(f"\n[Test Result] Loss: {test_loss:.4f} | Record-wise Accuracy: {test_acc:.4f}")