import os
import random
import time
import h5py
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, roc_auc_score, roc_curve, accuracy_score, precision_score, recall_score, f1_score
import numpy as np

def get_record_embeddings(directory, label=0):
    """
    directory 내의 모든 h5 파일을 하나의 bag으로 만듦
    """
    records = []
    file_paths = [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.endswith('.h5')
    ]

    for fp in file_paths:
        try:
            with h5py.File(fp, 'r') as f:
                length = f['standardized_latent_z'].shape[0]
                if length > 0:
                    records.append((fp, label))
        except Exception as e:
            print(f"Error opening {fp}: {e}")
            continue

    return records

class MILEmbeddingDataset(Dataset):
    def __init__(self, records):
        """
        records: [(file_path, label), ...]
        """
        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, idx):
        file_path, label = self.records[idx]
        try:
            with h5py.File(file_path, 'r') as f:
                emb = f['standardized_latent_z'][()]  # (num_instances, 1, 4)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            emb = np.zeros((1,1,4), dtype=np.float32)

        emb_tensor = torch.tensor(emb, dtype=torch.float32)
        label_tensor = torch.tensor(label, dtype=torch.long)

        return emb_tensor, label_tensor, file_path

# 어텐션 풀링 기반 MIL 모델
class AttentionMIL(nn.Module):
    def __init__(self, input_dim=4, hidden_dim=128, num_classes=2):
        super(AttentionMIL, self).__init__()

        # Instance-level 임베딩 네트워크
        self.instance_embedding = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )

        # Attention 모듈
        self.attention_layer = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

        # 최종 bag-level 분류기
        self.classifier = nn.Linear(hidden_dim, num_classes)

    def forward(self, x):
        """
        x: (batch_size, num_instances, 1, 4)
        """
        batch_size, num_instances, _, input_dim = x.size()
        
        # 인스턴스 차원 간소화 (batch, num_instances, 4)
        x = x.view(batch_size, num_instances, input_dim)
        
        # Instance-level 임베딩 (batch_size, num_instances, hidden_dim)
        H = self.instance_embedding(x)
        
        # Attention 계산 (batch_size, num_instances, 1)
        A = self.attention_layer(H)  # attention logits
        A = torch.softmax(A, dim=1)  # 각 인스턴스에 대한 attention weight
        
        # Bag-level 표현 계산 (batch_size, hidden_dim)
        M = torch.sum(A * H, dim=1)

        # Bag-level logits 계산 (batch_size, num_classes)
        logits = self.classifier(M)
        
        return logits, A.squeeze(-1)  # attention weights도 반환하여 해석 가능하게 함
 
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
            logits, attention_weights = model(signals)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * signals.size(0)
            _, predicted = torch.max(logits, 1)
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
                logits, attention_weights = model(signals)
                loss = criterion(logits, labels)
                val_loss += loss.item() * signals.size(0)
                _, predicted = torch.max(logits, 1)
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
 
def test_model(model, test_loader, device):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    model.to(device)
    

    final_preds, final_labels, final_probs = [], [], []

    with torch.no_grad():
        for signals, labels, record_ids in tqdm(test_loader, desc="[Test]"):
            signals = signals.to(device)
            labels = labels.to(device)
            logits, _ = model(signals)
            probs = F.softmax(logits, dim=1)[:, 1].cpu().numpy()
            preds = (probs >= 0.5).astype(int)
            final_preds.extend(preds)
            final_labels.extend(labels.cpu().numpy())
            final_probs.extend(probs)

 
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
 
if __name__ == "__main__":
    # GPU
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("[INFO] Using device:", device)
    random.seed(42)
    torch.manual_seed(42)
 
    # PSVT와 NON-PSVT 데이터 경로
    psvt_dir = "경로 수정"
    not_psvt_dir = "경로 수정"
 
    # 각 디렉토리에서 record 단위로 임베딩 추출
    psvt_records = get_record_embeddings(psvt_dir, label=1)
    not_psvt_records = get_record_embeddings(not_psvt_dir, label=0)
 
    # 두 record 리스트 합쳐서 섞은 뒤 train/val/test 분할
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
    train_dataset = MILEmbeddingDataset(train_records)
    val_dataset = MILEmbeddingDataset(val_records)
    test_dataset = MILEmbeddingDataset(test_records)
    print(f"[INFO] Segments - Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
 
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
 
    # 모델 생성: Attention 기반 MIL 모델
    model = AttentionMIL(input_dim=4, hidden_dim=128, num_classes=2)
    train_and_validate(model, train_loader, val_loader, device, epochs=5)
 
    # 최종 테스트 및 그래프 출력
    test_loss, test_acc = test_model(model, test_loader, device)
    print(f"\n[Test Result] Loss: {test_loss:.4f} | Record-wise Accuracy: {test_acc:.4f}")