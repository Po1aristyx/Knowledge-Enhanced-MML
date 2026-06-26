import csv
import nibabel as nib
import matplotlib.pyplot as plt
import random
import torch
import os
import numpy as np
import json
from model1 import CNN_3D,NiiDataset,MultiModalTransformer,NeuralNet,TransEModel,KGMultiModalTransformer
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import nibabel as nib
import shutil
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
import math
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import pandas as pd

path_existence = []
data_prodromal=[]
data_swedd=[]
data_control=[]
data_PD = []
count_control = 0
count_PD = 0
count_swedd = 0
count_prodromal = 0
with open('PD1.csv', mode='r', newline='', encoding='utf-8') as file:
    csv_reader = csv.reader(file)
    next(csv_reader)  
    for row in csv_reader:
        path = 'E:/code/PPMI/PD/' + row[1]
        exists = os.path.exists(path)
        path_existence.append((path, exists))
        if exists:
            count_PD=count_PD+1
            data_PD.append(row)
            
with open('control.csv', mode='r', newline='', encoding='utf-8') as file:
    csv_reader = csv.reader(file)
    next(csv_reader) 
    for row in csv_reader:
        path = 'E:/code/PPMI/Control/' + row[1]
        exists = os.path.exists(path)
        path_existence.append((path, exists))
        if exists:
            count_control=count_control+1
            data_control.append(row)
            
with open('swedd.csv', mode='r', newline='', encoding='utf-8') as file:
    csv_reader = csv.reader(file)
    next(csv_reader) 
    for row in csv_reader:
        path = 'E:/code/PPMI/SWEDD/' + row[1]
        exists = os.path.exists(path)
        path_existence.append((path, exists))
        if exists:
            count_swedd=count_swedd+1
            data_swedd.append(row)

with open('prodromal.csv', mode='r', newline='', encoding='utf-8') as file:
    csv_reader = csv.reader(file)
    next(csv_reader) 
    for row in csv_reader:
        path = 'E:/code/PPMI/Prodromal/' + row[1]
        exists = os.path.exists(path)
        path_existence.append((path, exists))
        if exists:
            count_prodromal=count_prodromal+1
            data_prodromal.append(row)
print(count_PD) 
print(count_control) 
print(count_swedd) 
print(count_prodromal)

import numpy as np
import math
import torch

replace_dict = {
    'F': '0', 'M':'1', 'Normal':'0', 'Slight':'1', 'Mild':'2', 'Moderate':'3', 'Severe':'4',
    'T1-anatomical':'1', 'Processed':'1', 'BL':'0', 'V04':'1', 'V06':'2', 'V08':'4', 'V10':'5',
    'NiFTI':'1', '': '0', 'No':'0', 'Yes':'1', 'Stage 1':'1', 'Stage 2':'2', 'Stage 3':'3', 'Stage 4':'4',
    'On':'1', 'Off':'0', 'Stage 0':'0'
}

ehr_cols = [4, 5, 82, 83]
phe_cols = list(range(12, 45)) + [78, 79, 80, 84]

def preprocess_data(data, replace_dict, target_cols):
    processed_data = []
    for row in data:
        extracted_row = [row[idx] for idx in target_cols]
        extracted_row = [replace_dict.get(item, item) for item in extracted_row]
        
        numerical_row = []
        for item in extracted_row:
            try:
                val = float(item)
                if math.isnan(val):
                    val = 0.0
            except ValueError:
                val = 0.0 
            numerical_row.append(val)
            
        processed_data.append(numerical_row)
    return np.array(processed_data)

# PHE模态
PD_array = preprocess_data(data_PD, replace_dict, phe_cols)
control_array = preprocess_data(data_control, replace_dict, phe_cols)
swedd_array = preprocess_data(data_swedd, replace_dict, phe_cols)
prodromal_array = preprocess_data(data_prodromal, replace_dict, phe_cols)

print('PD_array shape:', PD_array.shape)
print('control_array shape:', control_array.shape)
print('swedd_array shape:', swedd_array.shape)
print('prodromal_array shape:', prodromal_array.shape)

pd_ehr_data = preprocess_data(data_PD, replace_dict, ehr_cols)
control_ehr_data = preprocess_data(data_control, replace_dict, ehr_cols)
swedd_ehr_data = preprocess_data(data_swedd, replace_dict, ehr_cols)
prodromal_ehr_data = preprocess_data(data_prodromal, replace_dict, ehr_cols)

pd_EHR = torch.from_numpy(pd_ehr_data).float()
control_EHR = torch.from_numpy(control_ehr_data).float()
swedd_EHR = torch.from_numpy(swedd_ehr_data).float()
prodromal_EHR = torch.from_numpy(prodromal_ehr_data).float()

print('pd_EHR shape:', pd_EHR.shape)
print('control_EHR shape:', control_EHR.shape)
print('swedd_EHR shape:', swedd_EHR.shape)
print('prodromal_EHR shape:', prodromal_EHR.shape)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def center_crop_3d(tensor, size):
    depth, height, width = tensor.shape
    target_depth, target_height, target_width = size

    start_depth = (depth - target_depth) // 2
    start_height = (height - target_height) // 2
    start_width = (width - target_width) // 2

    end_depth = start_depth + target_depth
    end_height = start_height + target_height
    end_width = start_width + target_width

    return tensor[start_depth:end_depth, start_height:end_height, start_width:end_width]

import os
import torch
from torch.utils.data import DataLoader, Dataset
import nibabel as nib
import gc
from tqdm import tqdm

# 你的 NiiDataset，务必用这版
class NiiDataset(Dataset):
    def __init__(self, file_list_or_folder):
        if isinstance(file_list_or_folder, list):
            self.file_list = file_list_or_folder
        elif isinstance(file_list_or_folder, str):
            folder_path = file_list_or_folder
            self.file_list = [os.path.join(folder_path, filename)
                              for filename in os.listdir(folder_path)
                              if filename.endswith('.nii') or filename.endswith('.nii.gz')]
        else:
            raise ValueError("参数应为list或str")
    def __len__(self):
        return len(self.file_list)
    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        img = nib.load(file_path)
        img_data = img.get_fdata()
        img_tensor = torch.from_numpy(img_data).float()
        cropped_tensor = center_crop_3d(img_tensor, (64, 64, 64))   # 你的裁剪函数
        input_tensor = cropped_tensor.unsqueeze(0)
        return input_tensor

# 1. 严格用csv顺序生成影像路径
pd_img_paths = [os.path.join('E:/code/PPMI/PD', row[1]) for row in data_PD if os.path.exists(os.path.join('E:/code/PPMI/PD', row[1]))]
control_img_paths = [os.path.join('E:/code/PPMI/Control', row[1]) for row in data_control if os.path.exists(os.path.join('E:/code/PPMI/Control', row[1]))]
swedd_img_paths = [os.path.join('E:/code/PPMI/SWEDD', row[1]) for row in data_swedd if os.path.exists(os.path.join('E:/code/PPMI/SWEDD', row[1]))]
prodromal_img_paths = [os.path.join('E:/code/PPMI/Prodromal', row[1]) for row in data_prodromal if os.path.exists(os.path.join('E:/code/PPMI/Prodromal', row[1]))]

print(f'PD影像数量: {len(pd_img_paths)}')
print(f'Control影像数量: {len(control_img_paths)}')
print(f'SWEDD影像数量: {len(swedd_img_paths)}')
print(f'Prodromal影像数量: {len(prodromal_img_paths)}')

# 清理缓存
torch.cuda.empty_cache()
gc.collect()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 将模型输出维度修改为 64
nii = CNN_3D(num_class=64)
nii = nii.to(device)
nii.eval()

# 封装特征提取函数
def extract_mri_features_from_list(img_paths, batch_size=16, desc="Extracting"):
    dataset = NiiDataset(img_paths)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    all_outputs = []
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(tqdm(dataloader, desc=desc)):
            batch_data = batch_data.to(device)
            output = nii(batch_data)
            all_outputs.append(output.detach().cpu())
            
    del dataset, dataloader
    torch.cuda.empty_cache()
    gc.collect()
            
    return torch.cat(all_outputs, dim=0)

# 2. 用于影像特征提取
# 提取并将结果赋值给对应的变量，打印内容与你原逻辑保持一致
PD_output = extract_mri_features_from_list(pd_img_paths, batch_size=16, desc="PD Features")
print('PD nii shape--->', PD_output.shape)

control_output = extract_mri_features_from_list(control_img_paths, batch_size=16, desc="Control Features")
print('control nii shape--->', control_output.shape)

prodromal_output = extract_mri_features_from_list(prodromal_img_paths, batch_size=16, desc="Prodromal Features")
print('prodromal nii shape--->', prodromal_output.shape)

swedd_output = extract_mri_features_from_list(swedd_img_paths, batch_size=16, desc="SWEDD Features")
print('swedd nii shape--->', swedd_output.shape)

import numpy as np
import torch
import pandas as pd
import json

# 读取PPMI数据集对应的图谱特征文件
ENTITY_EMB_NPY = "PPMI-DementiaHKG-Embeddings.npy"
ENTITY2ID_FILE = "PPMI-DementiaHKG-Entity2ID.json"
PATIENT_QUERY_DICT = "PPMI-DementiaHKG-Entities.json"

with open(ENTITY2ID_FILE, 'r', encoding='utf-8') as f:
    entity2id = json.load(f)

with open(PATIENT_QUERY_DICT, 'r', encoding='utf-8') as f:
    patient_entities = json.load(f)

emb = torch.from_numpy(np.load(ENTITY_EMB_NPY)).float()
embed_dim = emb.shape[1] 

# 核心修改：适配PPMI图谱构建时的最大序列长度
max_seq_len = 35 

def get_patient_embedding_seq(subject_id):
    subject_id = str(subject_id).strip()
    # 兼容部分读取包含浮点后缀的异常情况
    try:
        subject_id = str(int(float(subject_id)))
    except ValueError:
        pass
        
    patient_data = patient_entities.get(subject_id, {})
    active_nodes = patient_data.get("input_nodes", [])
    
    vecs = []
    pad_masks = []
    
    for i in range(max_seq_len):
        if i < len(active_nodes):
            node = active_nodes[i]
            if node != "[PAD]" and node in entity2id:
                vecs.append(emb[entity2id[node]])
                pad_masks.append(False)
            else:
                vecs.append(torch.zeros(embed_dim))
                pad_masks.append(True)
        else:
            vecs.append(torch.zeros(embed_dim))
            pad_masks.append(True)
            
    return torch.stack(vecs), torch.tensor(pad_masks, dtype=torch.bool)

def process_and_verify_seq(df, cohort_name):
    seq_tensors = []
    mask_tensors = []
    valid_count = 0
    
    for _, r in df.iterrows():
        # 兼容不同列名获取患者ID
        if 'Subject' in r and pd.notna(r['Subject']):
            pid = r['Subject']
        elif 'PATNO' in r and pd.notna(r['PATNO']):
            pid = r['PATNO']
        else:
            pid = r.iloc[0] 
            
        seq_tensor, mask_tensor = get_patient_embedding_seq(pid)
        seq_tensors.append(seq_tensor)
        mask_tensors.append(mask_tensor)
        
        if not torch.all(mask_tensor):
            valid_count += 1
            
    stacked_seq = torch.stack(seq_tensors)
    stacked_mask = torch.stack(mask_tensors)
    
    print(f"[{cohort_name}] 总样本数: {len(df)} | 成功注入KG序列的样本数: {valid_count}")
    return stacked_seq, stacked_mask

pd_df        = pd.read_csv('PD1.csv')
control_df   = pd.read_csv('control.csv')
prodromal_df = pd.read_csv('prodromal.csv')
swedd_df     = pd.read_csv('swedd.csv')

print("开始提取PPMI知识序列与注意力掩码...")
pd_seq, pd_mask               = process_and_verify_seq(pd_df, "PD")
control_seq, control_mask     = process_and_verify_seq(control_df, "Control")
prodromal_seq, prodromal_mask = process_and_verify_seq(prodromal_df, "Prodromal")
swedd_seq, swedd_mask         = process_and_verify_seq(swedd_df, "SWEDD")

print("最终生成的知识序列 Shapes:", pd_seq.shape, control_seq.shape, prodromal_seq.shape, swedd_seq.shape)
print("最终生成的掩码张量 Shapes:", pd_mask.shape, control_mask.shape, prodromal_mask.shape, swedd_mask.shape)

# ------- 重新拼接特征 + 构造 DataLoader ------- #
# 将原先的 array 转换为 tensor，同时去掉旧的 pd_transe 均值特征
pd_tensor = torch.from_numpy(PD_array).float()
control_tensor = torch.from_numpy(control_array).float()
prodromal_tensor = torch.from_numpy(prodromal_array).float()
swedd_tensor = torch.from_numpy(swedd_array).float()

X_pd = torch.cat([pd_EHR, PD_output.cpu(), pd_tensor], dim=1)
X_control = torch.cat([control_EHR, control_output.cpu(), control_tensor], dim=1)
X_swedd = torch.cat([swedd_EHR, swedd_output.cpu(), swedd_tensor], dim=1)
X_prodromal = torch.cat([prodromal_EHR, prodromal_output.cpu(), prodromal_tensor], dim=1)

y_pd = torch.zeros(len(X_pd))          
y_control = torch.ones(len(X_control)) 
y_swedd = torch.full((len(X_swedd),), 2)
y_prodromal = torch.full((len(X_prodromal),), 3)

# 合并所有受试者，严格保持原代码的切片逻辑 (PD取前120)
features = torch.cat([X_pd[:120], X_control, X_prodromal, X_swedd], dim=0).float()
kg_seq_all = torch.cat([pd_seq[:120], control_seq, prodromal_seq, swedd_seq], dim=0).float()
kg_mask_all = torch.cat([pd_mask[:120], control_mask, prodromal_mask, swedd_mask], dim=0).bool()
y = torch.cat([y_pd[:120], y_control, y_prodromal, y_swedd], dim=0).long()

from sklearn.model_selection import train_test_split

# 第一次划分测试集
X_train, X_test, y_train, y_test, kg_seq_train, kg_seq_test, kg_mask_train, kg_mask_test = train_test_split(
    features.numpy(), y.numpy(), kg_seq_all.numpy(), kg_mask_all.numpy(),
    test_size=0.20, stratify=y.numpy(), random_state=32
)

# 第二次划分验证集
X_train, X_val, y_train, y_val, kg_seq_train, kg_seq_val, kg_mask_train, kg_mask_val = train_test_split(
    X_train, y_train, kg_seq_train, kg_mask_train,
    test_size=0.20, stratify=y_train, random_state=30
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

X_train_tensor = torch.FloatTensor(X_train).to(device)
y_train_tensor = torch.LongTensor(y_train).to(device)
kg_seq_train_tensor = torch.FloatTensor(kg_seq_train).to(device)
kg_mask_train_tensor = torch.BoolTensor(kg_mask_train).to(device)

X_val_tensor = torch.FloatTensor(X_val).to(device)
y_val_tensor = torch.LongTensor(y_val).to(device)
kg_seq_val_tensor = torch.FloatTensor(kg_seq_val).to(device)
kg_mask_val_tensor = torch.BoolTensor(kg_mask_val).to(device)

X_test_tensor = torch.FloatTensor(X_test).to(device)
y_test_tensor = torch.LongTensor(y_test).to(device)
kg_seq_test_tensor = torch.FloatTensor(kg_seq_test).to(device)
kg_mask_test_tensor = torch.BoolTensor(kg_mask_test).to(device)

from torch.utils.data import TensorDataset, DataLoader

batch_size = 32
train_dataset = TensorDataset(X_train_tensor, kg_seq_train_tensor, kg_mask_train_tensor, y_train_tensor)
val_dataset   = TensorDataset(X_val_tensor, kg_seq_val_tensor, kg_mask_val_tensor, y_val_tensor)
test_dataset  = TensorDataset(X_test_tensor, kg_seq_test_tensor, kg_mask_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader   = DataLoader(val_dataset,   batch_size=batch_size)
test_loader  = DataLoader(test_dataset,  batch_size=batch_size)
print("DataLoaders ready:", len(train_loader), len(val_loader), len(test_loader))

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    all_probs = []
    all_labels = []
    all_preds = []
    
    # 修改参数解析，增加 kg_seq 和 kg_mask
    for inputs, kg_seq, kg_mask, labels in loader:
        inputs, kg_seq, kg_mask, labels = inputs.to(device), kg_seq.to(device), kg_mask.to(device), labels.to(device)
        labels = labels.long()
        
        optimizer.zero_grad()
        # 传入新的模型定义所需参数
        outputs = model(inputs, kg_seq, kg_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()

        probs = torch.softmax(outputs, dim=1).detach().cpu().numpy()
        preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds)

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    train_auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')
    train_f1 = f1_score(all_labels, all_preds, average='macro')
    train_recall = recall_score(all_labels, all_preds, average='macro')
    train_precision = precision_score(all_labels, all_preds, average='macro')

    avg_loss = total_loss / len(loader)
    return avg_loss, train_auc, train_f1, train_recall, train_precision

def validate_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    all_probs = []
    all_labels = []
    all_preds = []
    with torch.no_grad():
        # 同步修改参数解析
        for inputs, kg_seq, kg_mask, labels in loader:
            inputs, kg_seq, kg_mask, labels = inputs.to(device), kg_seq.to(device), kg_mask.to(device), labels.to(device)
            labels = labels.long()
            
            outputs = model(inputs, kg_seq, kg_mask)
            loss = criterion(outputs, labels)
            
            total_loss += loss.item()
            probs = torch.softmax(outputs, dim=1).detach().cpu().numpy()
            preds = torch.argmax(outputs, dim=1).detach().cpu().numpy()
            all_probs.extend(probs)
            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds)
    
    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)
    all_preds = np.array(all_preds)

    val_auc = roc_auc_score(all_labels, all_probs, multi_class='ovo')
    val_f1 = f1_score(all_labels, all_preds, average='macro')
    val_recall = recall_score(all_labels, all_preds, average='macro')
    val_precision = precision_score(all_labels, all_preds, average='macro')
    
    avg_loss = total_loss / len(loader)
    return avg_loss, val_auc, val_f1, val_recall, val_precision

embed_dim        = 128   # 适配你新模型设计的嵌入维度
transe_embed_dim = 128  # 知识图谱嵌入维度保持不变
num_epochs       = 200
batch_size       = 32
learning_rate    = 1e-5
weight_decay     = 1e-3
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 实例化新模型，显式传入37维的bio_dim以及最大序列长度35
model = KGMultiModalTransformer(
    ehr_dim=4, 
    img_dim=64, 
    bio_dim=37, 
    embed_dim=embed_dim,
    num_heads=2, 
    transe_embed_dim=transe_embed_dim, 
    max_seq_len=35
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

train_losses = []
train_aucs = []
train_f1s = []
train_recalls = []
train_precisions = []
test_losses = []
test_aucs = []
test_f1s = []
test_recalls = []
test_precisions = []

for epoch in range(num_epochs):
    train_loss, train_auc, train_f1, train_recall, train_precision = train_epoch(model, train_loader, optimizer, criterion, device)
    test_loss, test_auc, test_f1, test_recall, test_precision = validate_epoch(model, test_loader, criterion, device)
    
    train_losses.append(train_loss)
    train_aucs.append(train_auc)
    train_f1s.append(train_f1)
    train_recalls.append(train_recall)
    train_precisions.append(train_precision)
    
    test_losses.append(test_loss)
    test_aucs.append(test_auc)
    test_f1s.append(test_f1)
    test_recalls.append(test_recall)
    test_precisions.append(test_precision)
    
    print(f"Epoch {epoch + 1}/{num_epochs}, "
          f"Train Loss: {train_loss:.4f}, Train AUC: {train_auc:.4f}  "   
          f"test Loss: {test_loss:.4f}, test AUC: {test_auc:.4f}")

model.eval()
all_probs = []
all_labels = []
with torch.no_grad():
    # 同步修改参数接收
    for inputs, kg_seq, kg_mask, labels in val_loader:
        inputs, kg_seq, kg_mask, labels = inputs.to(device), kg_seq.to(device), kg_mask.to(device), labels.to(device)
        # 传入序列和掩码进行前向传播
        outputs = model(inputs, kg_seq, kg_mask)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()
        all_probs.extend(probs)
        all_labels.extend(labels.cpu().numpy())

preds = np.argmax(all_probs, axis=1)
accuracy = accuracy_score(all_labels, preds)
precision = precision_score(all_labels, preds, average='macro')
recall = recall_score(all_labels, preds, average='macro')
f1 = f1_score(all_labels, preds, average='macro')
auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')

print("\n=== Final Validation Metrics ===")
print(f"Accuracy:  {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall:    {recall:.4f}")
print(f"F1 Score:  {f1:.4f}")
print(f"AUC-ROC:   {auc:.4f}")

plt.figure(figsize=(10, 5))
plt.plot(range(len(train_aucs)), train_aucs, label="Train AUC", color="blue")
plt.plot(range(len(test_aucs)), test_aucs, label="Test AUC", color="red")
plt.title("PPMI AUC")
plt.xlabel("Epoch")
plt.ylabel("AUC")
plt.legend()
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 5))
plt.plot(range(len(train_losses)), train_losses, label="Train Loss", color="blue")
plt.plot(range(len(test_losses)), test_losses, label="Test Loss", color="red")
plt.title("PPMI Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)
plt.show()

import torch
from thop import profile
# 如果提示找不到 thop，请先执行 !pip install thop

# 确保在相同的设备上运行
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ==========================================
# 1. 影像提取模块 (3D-CNN) 测算
# ==========================================
# 实例化你的影像提取模型
model_cnn = CNN_3D(num_class=64).to(device)

# 构建符合 NiiDataset 输出维度的虚拟输入: (Batch_size, Channels, D, H, W)
dummy_input_img = torch.randn(1, 1, 64, 64, 64).to(device)

# verbose=False 用于关闭 thop 默认的逐层打印输出
flops_cnn, params_cnn = profile(model_cnn, inputs=(dummy_input_img, ), verbose=False)


# ==========================================
# 2. 下游多模态分类模块 (Transformer) 测算
# ==========================================
# 根据你 .ipynb 中真实的维度进行实例化
model_transformer = KGMultiModalTransformer(
    ehr_dim=4, 
    img_dim=64, 
    bio_dim=37, 
    embed_dim=128, 
    num_heads=2, 
    transe_embed_dim=128,
    max_seq_len=35
).to(device)

# 构建符合数据特征的虚拟输入
# x 的总维度 = ehr(4) + img(64) + bio(37) = 105
dummy_x = torch.randn(1, 105).to(device)
dummy_kg_seq = torch.randn(1, 35, 128).to(device)
# 你的模型 forward 里有 (kg_mask == 0).bool()，所以这里用 int 类型的 0 来模拟
dummy_kg_mask = torch.zeros(1, 35, dtype=torch.int64).to(device)

flops_trans, params_trans = profile(model_transformer, inputs=(dummy_x, dummy_kg_seq, dummy_kg_mask), verbose=False)


# ==========================================
# 3. 打印统计结果
# ==========================================
print("=" * 45)
print("【影像特征提取模块 (3D-CNN)】")
print(f"参数量 (Params): {params_cnn / 1e6:.4f} M")
print(f"算力损耗 (FLOPs): {flops_cnn / 1e9:.4f} G")
print("-" * 45)

print("【下游多模态分类模块 (KGMultiModalTransformer)】")
print(f"参数量 (Params): {params_trans / 1e6:.4f} M")
print(f"算力损耗 (FLOPs): {flops_trans / 1e9:.4f} G")
print("-" * 45)

print("【总计】")
print(f"总参数量: {(params_cnn + params_trans) / 1e6:.4f} M")
print(f"总算力损耗: {(flops_cnn + flops_trans) / 1e9:.4f} G")
print("=" * 45)



