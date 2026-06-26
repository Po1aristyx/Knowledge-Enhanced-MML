import torch
import torch.nn as nn
import os
import nibabel as nib
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
import math
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score


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


class NiiDataset(Dataset):
    def __init__(self, folder_path_or_list):
        if isinstance(folder_path_or_list, list):
            self.file_list = folder_path_or_list
        else:
            self.folder_path = folder_path_or_list
            self.file_list = [os.path.join(folder_path_or_list, filename) for filename in os.listdir(folder_path_or_list) if
                              filename.endswith('.nii') or filename.endswith('.nii.gz')]

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        img = nib.load(file_path)
        img_data = img.get_fdata()
        img_tensor = torch.from_numpy(img_data).float()
        cropped_tensor = center_crop_3d(img_tensor, (64, 64, 64)) #1 128 128 128
        # 1通道
        input_tensor = cropped_tensor.unsqueeze(0)
        return input_tensor


class CNN_3D(nn.Module):
    def __init__(self, num_class=1):  # num_class
        super().__init__()
        self.features = nn.Sequential(
            # 1 128 128 128 3 1 2
            nn.Conv3d(1, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2), #1 64 64 64
            nn.Conv3d(16, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool3d(kernel_size=2, stride=2),
        )
        self.classifier = nn.Sequential(
            # 1 32 32 32
            nn.Linear(32 * 16 * 16 * 16, 64),  #32 24
            nn.ReLU(),
            nn.Linear(64, num_class)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# 定义神经网络模型
class NeuralNet(nn.Module):
    def __init__(self,embedding, num_classes=3):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(embedding, 32),
            nn.ReLU(),    # 添加非线性激活函数
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, num_classes)  # 1wei
        )
    
    def forward(self, x):
        return self.layers(x)

# 4.11双模态 模型定义
class DualTransformer(nn.Module):
    def __init__(self, embed_dim=32, num_heads=4, num_layers=2, num_classes=3):
        super().__init__()
        # 模态编码器 (输出形状: [batch, embed_dim])
        self.ehr_encoder = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.ReLU()
        )
        self.img_encoder = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.ReLU()
        )
        
        # 模态 A (ehr) 的 Q、K、V 线性层
        self.ehr_q = nn.Linear(embed_dim, embed_dim)
        self.ehr_k = nn.Linear(embed_dim, embed_dim)
        self.ehr_v = nn.Linear(embed_dim, embed_dim)
        
        # 模态 B (img) 的 Q、K、V 线性层
        self.img_q = nn.Linear(embed_dim, embed_dim)
        self.img_k = nn.Linear(embed_dim, embed_dim)
        self.img_v = nn.Linear(embed_dim, embed_dim)
        
        # 交叉注意力机制
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # 将 ab_combined 的维度从 2 * embed_dim 调整为 embed_dim
        self.ab_proj = nn.Linear(2 * embed_dim, embed_dim)
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, 64),  # 输入维度改为 2 * embed_dim
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        # 输入分解
        ehr = x[:, 0].unsqueeze(1)  # [batch, 1]
        img = x[:, 1].unsqueeze(1)
        
        # 模态编码
        ehr_feat = self.ehr_encoder(ehr).unsqueeze(1)  # [batch, 1, embed_dim]
        img_feat = self.img_encoder(img).unsqueeze(1)
        
        # 模态 A (ehr) 的 Q、K、V
        Qa = self.ehr_q(ehr_feat)  # [batch, 1, embed_dim]
        Ka = self.ehr_k(ehr_feat)
        Va = self.ehr_v(ehr_feat)
        
        # 模态 B (img) 的 Q、K、V
        Qb = self.img_q(img_feat)  # [batch, 1, embed_dim]
        Kb = self.img_k(img_feat)
        Vb = self.img_v(img_feat)
        
        # 交叉注意力机制
        # Qa 与 Kb、Vb 进行注意力计算
        ehr_img_attn, _ = self.cross_attention(
            query=Qa,  # [batch, 1, embed_dim]
            key=Kb,    # [batch, 1, embed_dim]
            value=Vb   # [batch, 1, embed_dim]
        )
        
        # Qb 与 Ka、Va 进行注意力计算
        img_ehr_attn, _ = self.cross_attention(
            query=Qb,  # [batch, 1, embed_dim]
            key=Ka,    # [batch, 1, embed_dim]
            value=Va   # [batch, 1, embed_dim]
        )
        
        # 拼接交叉注意力结果
        ab_combined = torch.cat([ehr_img_attn.squeeze(1), img_ehr_attn.squeeze(1)], dim=1)  # [batch, 2 * embed_dim]
        
        # 输入到融合层
        return self.fusion(ab_combined)




#4.11 三模态
class MultiModalTransformer(nn.Module):
    def __init__(self, embed_dim=16, num_heads=2, num_layers=2, num_classes=3):
        super().__init__()
        # 模态编码器 (输出形状: [batch, embed_dim])
        self.ehr_encoder = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.ReLU()
        )
        self.img_encoder = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.ReLU()
        )
        self.bio_encoder = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.ReLU()
        )
        

        self.bio_q = nn.Linear(embed_dim, embed_dim)
        self.bio_k = nn.Linear(embed_dim, embed_dim)
        self.bio_v = nn.Linear(embed_dim, embed_dim)

        self.ehr_q = nn.Linear(embed_dim, embed_dim)
        self.ehr_k = nn.Linear(embed_dim, embed_dim)
        self.ehr_v = nn.Linear(embed_dim, embed_dim)

        self.img_q = nn.Linear(embed_dim, embed_dim)
        self.img_k = nn.Linear(embed_dim, embed_dim)
        self.img_v = nn.Linear(embed_dim, embed_dim)
        
        # 交叉注意力机制
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )
        
        # 将 ab_combined 的维度从 2 * embed_dim 调整为 embed_dim
        self.ab_proj = nn.Linear(2 * embed_dim, embed_dim)
        
        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, 64),  # 输入维度改为 2 * embed_dim
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )
        
    def forward(self, x):
        # 输入分解
        ehr = x[:, 0].unsqueeze(1)  # [batch, 1]
        img = x[:, 1].unsqueeze(1)
        bio = x[:, 2].unsqueeze(1)
        
        # 模态编码
        ehr_feat = self.ehr_encoder(ehr).unsqueeze(1)  # [batch, 1, embed_dim]
        img_feat = self.img_encoder(img).unsqueeze(1)
        bio_feat = self.bio_encoder(bio).unsqueeze(1)
 
        Qa = self.bio_q(bio_feat)  # [batch, 1, embed_dim]
        Ka = self.bio_k(bio_feat)
        Va = self.bio_v(bio_feat)

        Qb = self.ehr_q(ehr_feat)  # [batch, 1, embed_dim]
        Kb = self.ehr_k(ehr_feat)
        Vb = self.ehr_v(ehr_feat)

        Qc = self.img_q(img_feat)  # [batch, 1, embed_dim]
        Kc = self.img_k(img_feat)
        Vc = self.img_v(img_feat)
        
        # 交叉注意力机制
        # Qa 与 Kb、Vb 进行注意力计算
        bio_ehr_attn, _ = self.cross_attention(
            query=Qa,  # [batch, 1, embed_dim]
            key=Kb,    # [batch, 1, embed_dim]
            value=Vb   # [batch, 1, embed_dim]
        )
        
        # Qb 与 Ka、Va 进行注意力计算
        ehr_bio_attn, _ = self.cross_attention(
            query=Qb,  # [batch, 1, embed_dim]
            key=Ka,    # [batch, 1, embed_dim]
            value=Va   # [batch, 1, embed_dim]
        )

        ab_combined = torch.cat([bio_ehr_attn, ehr_bio_attn], dim=1)  # [batch, 2, embed_dim]
        ab_combined = ab_combined.view(ab_combined.size(0), -1)  # [batch, 2 * embed_dim]
        ab_combined = self.ab_proj(ab_combined)  # [batch, embed_dim]
        ab_combined = ab_combined.unsqueeze(1)  # [batch, 1, embed_dim]
        
        # 模态 C 进行交叉注意力
        img_ab_attn, _ = self.cross_attention(
            query=Qc,  # [batch, 1, embed_dim]
            key=ab_combined,  # [batch, 1, embed_dim]
            value=ab_combined  # [batch, 1, embed_dim]
        )

        final_combined = torch.cat([ab_combined.squeeze(1), img_ab_attn.squeeze(1)], dim=1)  # [batch, 2 * embed_dim]
        
        # 输入到融合层
        return self.fusion(final_combined)


class TransEModel(nn.Module):
    def __init__(self, num_entities, num_relations, embed_dim, num_classes=3):
        super().__init__()
        self.ent_embeddings = nn.Embedding(num_entities, embed_dim)
        self.rel_embeddings = nn.Embedding(num_relations, embed_dim) 
        self.zero_const = nn.Parameter(torch.zeros(1)) 
        self.pi_const = nn.Parameter(torch.tensor(3.14159)) 

    def forward(self, h, r, t):
        h_embed = self.ent_embeddings(h)
        r_embed = self.rel_embeddings(r)
        t_embed = self.ent_embeddings(t)
        return h_embed, r_embed, t_embed


# early
class KGMultiModalTransformer_earliest(nn.Module):
    def __init__(self, embed_dim=16, num_heads=2, num_layers=2, transe_embed_dim=32, num_classes=3):
        super().__init__()
        # 模态编码器
        self.ehr_encoder = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.ReLU()
        )
        self.img_encoder = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.ReLU()
        )
        self.bio_encoder = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.ReLU()
        )

        # TransE 嵌入投影层
        self.transe_proj = nn.Linear(transe_embed_dim, embed_dim)

        # 其他部分保持不变
        self.bio_q = nn.Linear(embed_dim, embed_dim)
        self.bio_k = nn.Linear(embed_dim, embed_dim)
        self.bio_v = nn.Linear(embed_dim, embed_dim)

        self.ehr_q = nn.Linear(embed_dim, embed_dim)
        self.ehr_k = nn.Linear(embed_dim, embed_dim)
        self.ehr_v = nn.Linear(embed_dim, embed_dim)

        self.img_q = nn.Linear(embed_dim, embed_dim)
        self.img_k = nn.Linear(embed_dim, embed_dim)
        self.img_v = nn.Linear(embed_dim, embed_dim)

        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )

        self.ab_proj = nn.Linear(4 * embed_dim, embed_dim)

        # 融合层
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 3, 64),  # 输入维度改为 3 * embed_dim
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, num_classes)
        )

    def forward(self, x, transe_embed):
        # 输入分解
        ehr = x[:, 0].unsqueeze(1)  # [batch, 1]
        img = x[:, 1].unsqueeze(1)
        bio = x[:, 2].unsqueeze(1)

        # 模态编码
        ehr_feat = self.ehr_encoder(ehr).unsqueeze(1)  # [batch, 1, embed_dim]
        img_feat = self.img_encoder(img).unsqueeze(1)
        bio_feat = self.bio_encoder(bio).unsqueeze(1)

        # TransE 嵌入投影
        transe_feat = self.transe_proj(transe_embed).unsqueeze(1)  # [batch, 1, embed_dim]

        # 将 TransE 嵌入与其他模态特征拼接
        bio_feat = torch.cat([bio_feat, transe_feat], dim=1)  # [batch, 2, embed_dim]
        ehr_feat = torch.cat([ehr_feat, transe_feat], dim=1)
        img_feat = torch.cat([img_feat, transe_feat], dim=1)


        Qa = self.bio_q(bio_feat)  # [batch, 2, embed_dim]
        Ka = self.bio_k(bio_feat)
        Va = self.bio_v(bio_feat)

        Qb = self.ehr_q(ehr_feat)  # [batch, 2, embed_dim]
        Kb = self.ehr_k(ehr_feat)
        Vb = self.ehr_v(ehr_feat)

        Qc = self.img_q(img_feat)  # [batch, 2, embed_dim]
        Kc = self.img_k(img_feat)
        Vc = self.img_v(img_feat)

    # 交叉注意力机制
        bio_ehr_attn, _ = self.cross_attention(query=Qa, key=Kb, value=Vb)  # [batch, 2, embed_dim]
        ehr_bio_attn, _ = self.cross_attention(query=Qb, key=Ka, value=Va)  # [batch, 2, embed_dim]

    # 拼接交叉注意力结果
        ab_combined = torch.cat([bio_ehr_attn, ehr_bio_attn], dim=1)  # [batch, 4, embed_dim]
        ab_combined = ab_combined.view(ab_combined.size(0), -1)  # [batch, 4 * embed_dim]
        ab_combined = self.ab_proj(ab_combined).unsqueeze(1)  # [batch, 1, embed_dim]
    
    # 模态 C 进行交叉注意力
        img_ab_attn, _ = self.cross_attention(query=Qc, key=ab_combined, value=ab_combined)  # [batch, 2, embed_dim]

        final_combined = torch.cat(
            [
                ab_combined.squeeze(1),  
                img_ab_attn.mean(dim=1),  
                transe_feat.squeeze(1)   
            ],
            dim=1
        )  
        return self.fusion(final_combined)  # [batch, 3]








#3.22
class KGMultiModalTransformer_old(nn.Module):
    """
    诊断知识引导的多模态协同交互网络 (Knowledge-Guided Multimodal Synergistic Interaction)
    保持真实的输入维度：EHR=14, IMG=1, BIO=39
    """
    def __init__(self, ehr_dim=14, img_dim=1, bio_dim=39, embed_dim=16, num_heads=2, num_layers=2, transe_embed_dim=128, num_classes=3):
        super().__init__()
        self.ehr_dim = ehr_dim
        self.img_dim = img_dim
        self.bio_dim = bio_dim
        
        # ==========================================
        # 1. 跨模态解耦编码器 (Cross-Modal Decoupling)
        # 作用：将异质、有噪声的原始模态数据映射到统一的连续低维特征空间 (embed_dim=16)
        # ==========================================
        
        # EHR (临床表型) 编码器: 14 -> 16 -> embed_dim
        self.ehr_encoder = nn.Sequential(
            nn.Linear(ehr_dim, 16),
            nn.ReLU(),
            nn.Linear(16, embed_dim),
            nn.ReLU()
        )
        
        # IMG/MRI (神经影像) 编码器: 1 -> embed_dim
        self.img_encoder = nn.Sequential(
            nn.Linear(img_dim, embed_dim),
            nn.ReLU()
        )
        
        # BIO (生物标志物) 编码器: 39 -> 32 -> 16 -> embed_dim
        # 注: 考虑到BIO噪声大、维度高，采用更深的网络进行逐层特征降维提纯
        self.bio_encoder = nn.Sequential(
            nn.Linear(bio_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, embed_dim),
            nn.ReLU()
        )

        # ==========================================
        # 2. 知识流形投影层 (Manifold Projection)
        # 作用：将来自 DementiaHKG 的高维(128维) TransE 先验知识等维映射至多模态语义空间
        # ==========================================
        self.transe_proj = nn.Linear(transe_embed_dim, embed_dim)

        # ==========================================
        # 3. 线性投影层 (QKV Projections)
        # 作用：为多头交叉注意力机制准备 Query, Key, Value 子空间
        # ==========================================
        self.bio_q = nn.Linear(embed_dim, embed_dim)
        self.bio_k = nn.Linear(embed_dim, embed_dim)
        self.bio_v = nn.Linear(embed_dim, embed_dim)

        self.ehr_q = nn.Linear(embed_dim, embed_dim)
        self.ehr_k = nn.Linear(embed_dim, embed_dim)
        self.ehr_v = nn.Linear(embed_dim, embed_dim)

        self.img_q = nn.Linear(embed_dim, embed_dim)
        self.img_k = nn.Linear(embed_dim, embed_dim)
        self.img_v = nn.Linear(embed_dim, embed_dim)

        # ==========================================
        # 4. 交叉注意力引擎 (Cross-Attention Engine)
        # 作用：执行多模态特征间的隐式映射与动态约束 (基于凸组合机理)
        # batch_first=True 保证输入输出的张量格式为 [batch_size, seq_len, embed_dim]
        # ==========================================
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True
        )

        # 底层特征融合的瓶颈投影层
        # 作用：将双向注意力结果 (2段seq * 2个方向 = 4 * embed_dim) 重新压缩为一个统一表征
        self.ab_proj = nn.Linear(4 * embed_dim, embed_dim)

        # ==========================================
        # 5. 最终决策融合层 (Fusion & Classification)
        # 作用：将底层“表型-生理”统一特征与高阶影像注意力特征进行融合分类
        # 【修复点 1】：切断了知识主干的直接残差链接，输入维度缩减为 embed_dim * 2
        # ==========================================
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, 64), 
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2), # 防止过拟合
            nn.Linear(64, num_classes) # 输出最终的3个类别概率分布 (例如: NC, MCI, AD)
        )

    def forward(self, x, transe_embed):
        # ==========================================
        # 外面拼进来的 x 总长度是动态的，按照各模态真实维度切片
        ehr_end = self.ehr_dim
        img_end = ehr_end + self.img_dim
        
        ehr = x[:, 0:ehr_end]        
        img = x[:, ehr_end:img_end]       
        bio = x[:, img_end:img_end+self.bio_dim]       


        # 模态独立编码，并通过 unsqueeze(1) 增加 sequence 维度，为后续 Transformer 处理做准备
        ehr_feat = self.ehr_encoder(ehr).unsqueeze(1)  # 形状变为 [batch, 1, embed_dim]
        img_feat = self.img_encoder(img).unsqueeze(1)  # 形状变为 [batch, 1, embed_dim]
        bio_feat = self.bio_encoder(bio).unsqueeze(1)  # 形状变为 [batch, 1, embed_dim]

        # 提取外部图谱专家知识，压缩至统一维度，并增加 sequence 维度
        transe_feat = self.transe_proj(transe_embed).unsqueeze(1)  # 形状变为 [batch, 1, embed_dim]

        # ==========================================
        # 阶段 B：知识前缀注入 (Knowledge Prefix Injection)
        # 作用：在序列维度强行植入医学专家查询词，扩充序列长度为 2
        # ==========================================
        bio_feat = torch.cat([bio_feat, transe_feat], dim=1)  # [batch, 2, embed_dim]
        ehr_feat = torch.cat([ehr_feat, transe_feat], dim=1)  # [batch, 2, embed_dim]
        img_feat = torch.cat([img_feat, transe_feat], dim=1)  # [batch, 2, embed_dim]

        # ==========================================
        # 阶段 C：底层特征交互 (BIO 与 EHR 的动态拓扑约束)
        # ==========================================
        # 生成 BIO 的 Query, Key, Value
        Qa = self.bio_q(bio_feat)  
        Ka = self.bio_k(bio_feat)
        Va = self.bio_v(bio_feat)

        # 生成 EHR 的 Query, Key, Value
        Qb = self.ehr_q(ehr_feat)  
        Kb = self.ehr_k(ehr_feat)
        Vb = self.ehr_v(ehr_feat)

        # 交叉注意力机制计算 (互相以对方的 K, V 进行注意力寻优)
        bio_ehr_attn, _ = self.cross_attention(query=Qa, key=Kb, value=Vb)  # 输出形状: [batch, 2, embed_dim]
        ehr_bio_attn, _ = self.cross_attention(query=Qb, key=Ka, value=Va)  # 输出形状: [batch, 2, embed_dim]

        # 拼接双向交叉注意力结果，展平后通过全连接层压缩，提取底层健康状态统一表征 (H_ab)
        ab_combined = torch.cat([bio_ehr_attn, ehr_bio_attn], dim=1)          # 沿 seq 维度拼接: [batch, 4, embed_dim]
        ab_combined = ab_combined.view(ab_combined.size(0), -1)               # 展平: [batch, 4 * embed_dim]
        ab_combined = self.ab_proj(ab_combined).unsqueeze(1)                  # 压缩并恢复 seq 维度: [batch, 1, embed_dim]
        
        # ==========================================
        # 阶段 D：高阶脑影像对齐 (Higher-Order MRI Alignment)
        # ==========================================
        # 生成 IMG/MRI 的 Query, Key, Value
        Qc = self.img_q(img_feat)  
        Kc = self.img_k(img_feat)
        Vc = self.img_v(img_feat)

        # 以顶层影像学表观 (Qc) 为导向，去 Query 底层统一表征 (ab_combined)
        img_ab_attn, _ = self.cross_attention(query=Qc, key=ab_combined, value=ab_combined)  # 输出形状: [batch, 2, embed_dim]

        # ==========================================
        # 阶段 E：强制隐式知识内化与分类决策 (Forced Implicit Knowledge Internalization)
        # ==========================================
        # 【修复点 2】：强制截断原始 transe_feat 的直接拼接待，只拼接底层压缩表征和高阶影像池化结果
        # 这逼迫模型必须通过上述的多重交叉注意力计算，去消化和内化 DementiaHKG 的知识
        final_combined = torch.cat(
            [
                ab_combined.squeeze(1),   # 移除 sequence 维度: [batch, embed_dim]
                img_ab_attn.mean(dim=1)   # 对高阶影像特征进行序列平均池化: [batch, embed_dim]
            ],
            dim=1
        ) # 拼接后形状: [batch, embed_dim * 2]
        
        # 通过 MLP 输出预测概率空间
        return self.fusion(final_combined)  # 输出形状: [batch, 3]



#4.1
class KGMultiModalTransformer(nn.Module):
    """
    诊断知识引导的多模态协同交互网络 (Knowledge-Guided Multimodal Synergistic Interaction)
    升级版：引入模态感知知识检索 (Modality-Aware Knowledge Retrieval) 与动态掩码屏蔽
    """
    def __init__(self, ehr_dim=14, img_dim=64, bio_dim=39, embed_dim=16, num_heads=2, num_layers=2, transe_embed_dim=128, num_classes=3):
        super().__init__()
        
        self.ehr_dim = ehr_dim
        self.img_dim = img_dim
        self.bio_dim = bio_dim
        
        # ==========================================
        # 1. 跨模态解耦编码器
        # 作用：处理异质数据。将各模态观测值映射到统一的低维空间。
        # ==========================================
        self.ehr_encoder = nn.Sequential(
            nn.Linear(ehr_dim, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU()
        )
        
        self.img_encoder = nn.Sequential(
            nn.Linear(img_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU()
        )
        
        self.bio_encoder = nn.Sequential(
            nn.Linear(bio_dim, 32),
            nn.BatchNorm1d(32),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(32, 16),
            nn.BatchNorm1d(16),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(16, embed_dim),
            nn.BatchNorm1d(embed_dim),
            nn.ReLU()
        )

        # ==========================================
        # 2. 知识流形投影层
        # 作用：构建知识流形字典。将 128 维的图谱先验等维映射到多模态语义空间。
        # ==========================================
        self.transe_proj = nn.Sequential(
            nn.Linear(transe_embed_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, embed_dim)
        )

        # ==========================================
        # 3. 模态感知知识检索引擎
        # 作用：各个模态独立在图谱序列中进行寻优。
        # ==========================================
        # 定义多头注意力。batch_first=True 保证输入格式为 [batch, seq, dim]。
        self.ehr_kg_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=0.2)
        self.bio_kg_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=0.2)
        self.img_kg_attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True, dropout=0.2)

        # 定义层归一化。用于后续的残差连接。
        self.norm_ehr_kg = nn.LayerNorm(embed_dim)
        self.norm_bio_kg = nn.LayerNorm(embed_dim)
        self.norm_img_kg = nn.LayerNorm(embed_dim)

        # ==========================================
        # 4. 线性投影层
        # 作用：为底层跨模态交互准备查询 (Q)、键 (K) 和值 (V) 空间。
        # ==========================================
        self.bio_q = nn.Linear(embed_dim, embed_dim)
        self.bio_k = nn.Linear(embed_dim, embed_dim)
        self.bio_v = nn.Linear(embed_dim, embed_dim)

        self.ehr_q = nn.Linear(embed_dim, embed_dim)
        self.ehr_k = nn.Linear(embed_dim, embed_dim)
        self.ehr_v = nn.Linear(embed_dim, embed_dim)

        self.img_q = nn.Linear(embed_dim, embed_dim)

        # ==========================================
        # 5. 交叉注意力引擎与降维模块
        # ==========================================
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            batch_first=True,
            dropout=0.2
        )
        
        self.norm_bio_ehr = nn.LayerNorm(embed_dim)
        self.norm_ehr_bio = nn.LayerNorm(embed_dim)
        self.norm_img = nn.LayerNorm(embed_dim)

        # 底层交互特征拼接后长度变为 2 * embed_dim。这里将其重新压缩。
        self.ab_proj = nn.Sequential(
            nn.Linear(2 * embed_dim, embed_dim),
            nn.ReLU(),
            nn.Dropout(0.2)
        )

        # ==========================================
        # 6. 最终决策融合层
        # 作用：输出最终的疾病类别预测概率。
        # ==========================================
        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 2, 64), 
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.2), 
            nn.Linear(64, num_classes) 
        )

    def forward(self, x, kg_seq, kg_mask):
        # 外部输入维度说明：
        # kg_seq 形如 [batch, 35, 128]
        # kg_mask 形如 [batch, 35]，用于过滤无效节点

        # ==========================================
        # 阶段 A：输入切片与独立模态编码
        # ==========================================
        # 按照特征维度切分原始输入矩阵 x。
        ehr_end = self.ehr_dim
        img_end = ehr_end + self.img_dim
        
        ehr = x[:, :ehr_end]                         
        img = x[:, ehr_end:img_end]                  
        bio = x[:, img_end:img_end+self.bio_dim]     

        # 模态独立编码。增加序列维度以匹配 Transformer 输入要求。
        ehr_feat = self.ehr_encoder(ehr).unsqueeze(1) # [batch, 1, embed_dim]
        img_feat = self.img_encoder(img).unsqueeze(1) # [batch, 1, embed_dim]
        bio_feat = self.bio_encoder(bio).unsqueeze(1) # [batch, 1, embed_dim]

        # 将包含 35 个节点的图谱序列投影到隐藏空间。
        kg_feat = self.transe_proj(kg_seq)            # [batch, 35, embed_dim]

        # ==========================================
        # 阶段 B：模态感知知识检索
        # 对应理论：带有先验约束的能量最小化问题近似求解。
        # ==========================================
        # 各个模态独立作为 Query 检索局部图谱子图。动态掩码 kg_mask 此时生效。
        ehr_kg_out, _ = self.ehr_kg_attn(query=ehr_feat, key=kg_feat, value=kg_feat, key_padding_mask=kg_mask)
        bio_kg_out, _ = self.bio_kg_attn(query=bio_feat, key=kg_feat, value=kg_feat, key_padding_mask=kg_mask)
        img_kg_out, _ = self.img_kg_attn(query=img_feat, key=kg_feat, value=kg_feat, key_padding_mask=kg_mask)

        # 通过残差连接将图谱知识内化到原始特征中。这保证了数据保真度。
        ehr_enhanced = self.norm_ehr_kg(ehr_feat + ehr_kg_out)
        bio_enhanced = self.norm_bio_kg(bio_feat + bio_kg_out)
        img_enhanced = self.norm_img_kg(img_feat + img_kg_out)

        # ==========================================
        # 阶段 C：底层特征交互
        # 对应理论：在知识底座上进行精确的语义对齐。
        # ==========================================
        # 生成 BIO 的 Q、K、V。
        Qa = self.bio_q(bio_enhanced)  
        Ka = self.bio_k(bio_enhanced)
        Va = self.bio_v(bio_enhanced)

        # 生成 EHR 的 Q、K、V。
        Qb = self.ehr_q(ehr_enhanced)  
        Kb = self.ehr_k(ehr_enhanced)
        Vb = self.ehr_v(ehr_enhanced)

        # 执行双向交叉注意力计算。
        bio_ehr_attn, _ = self.cross_attention(query=Qa, key=Kb, value=Vb) 
        ehr_bio_attn, _ = self.cross_attention(query=Qb, key=Ka, value=Va) 

        # 进行残差连接与归一化。这限制了模型在参数空间中的自由度。
        bio_ehr_out = self.norm_bio_ehr(bio_enhanced + bio_ehr_attn)
        ehr_bio_out = self.norm_ehr_bio(ehr_enhanced + ehr_bio_attn)

        # 拼接交互结果并进行降维压缩。提取统一的底层健康状态表征。
        ab_combined = torch.cat([bio_ehr_out, ehr_bio_out], dim=1)        # [batch, 2, embed_dim]
        ab_combined = ab_combined.view(ab_combined.size(0), -1)           # [batch, 2 * embed_dim]
        ab_combined = self.ab_proj(ab_combined).unsqueeze(1)              # [batch, 1, embed_dim]
        
        # ==========================================
        # 阶段 D：高阶脑影像对齐
        # ==========================================
        # 生成高阶影像查询。
        Qc = self.img_q(img_enhanced)  
        
        # 影像特征查询底层统一表征。
        img_ab_attn, _ = self.cross_attention(query=Qc, key=ab_combined, value=ab_combined) 
        img_out = self.norm_img(img_enhanced + img_ab_attn)               # [batch, 1, embed_dim]

        # ==========================================
        # 阶段 E：强制隐式知识内化与分类决策
        # ==========================================
        # 直接拼接底层压缩特征和高阶影像特征。这里强制截断了外部知识的直接访问路径。
        final_combined = torch.cat(
            [
                ab_combined.squeeze(1),   # 移除序列维度: [batch, embed_dim]
                img_out.squeeze(1)        # 移除序列维度: [batch, embed_dim]
            ],
            dim=1
        ) # 最终拼接维度: [batch, embed_dim * 2]
        
        # 输出预测结果。
        return self.fusion(final_combined)



#3.1Perceiver
class KGMultiModalPerceiver(nn.Module):
    def __init__(self, embed_dim=16, num_heads=2, num_layers=2, transe_embed_dim=32, num_latents=4, dropout_rate=0.3, num_classes=3):
        super().__init__()
        self.ehr_encoder = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.ReLU()
        )
        self.img_encoder = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.ReLU()
        )
        self.bio_encoder = nn.Sequential(
            nn.Linear(1, embed_dim),
            nn.ReLU()
        )

        self.transe_proj = nn.Linear(transe_embed_dim, embed_dim)

        self.latents = nn.Parameter(torch.randn(1, num_latents, embed_dim) * 0.02)
        
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=embed_dim, 
            num_heads=num_heads, 
            batch_first=True,
            dropout=dropout_rate
        )
        
        self.latent_self_attns = nn.ModuleList([
            nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, batch_first=True, dropout=dropout_rate)
            for _ in range(num_layers)
        ])
        
        self.latent_norms = nn.ModuleList([
            nn.LayerNorm(embed_dim)
            for _ in range(num_layers)
        ])

        self.latent_ffns = nn.ModuleList([
            nn.Sequential(
                nn.Linear(embed_dim, embed_dim * 2),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(embed_dim * 2, embed_dim),
                nn.Dropout(dropout_rate)
            )
            for _ in range(num_layers)
        ])
        
        self.latent_ffn_norms = nn.ModuleList([
            nn.LayerNorm(embed_dim)
            for _ in range(num_layers)
        ])
        
        # 核心修改：由于后续采用全局平均池化，输入维度由 num_latents * embed_dim 变更为 embed_dim
        self.latent_proj = nn.Linear(embed_dim, embed_dim * 2)

        self.fusion = nn.Sequential(
            nn.Linear(embed_dim * 3, 64),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_classes)
        )

    def forward(self, x, transe_embed):
        ehr = x[:, 0].unsqueeze(1)
        img = x[:, 1].unsqueeze(1)
        bio = x[:, 2].unsqueeze(1)

        ehr_feat = self.ehr_encoder(ehr).unsqueeze(1)
        img_feat = self.img_encoder(img).unsqueeze(1)
        bio_feat = self.bio_encoder(bio).unsqueeze(1)

        transe_feat = self.transe_proj(transe_embed).unsqueeze(1)

        bio_feat = torch.cat([bio_feat, transe_feat], dim=1)
        ehr_feat = torch.cat([ehr_feat, transe_feat], dim=1)
        img_feat = torch.cat([img_feat, transe_feat], dim=1)

        inputs_seq = torch.cat([bio_feat, ehr_feat, img_feat], dim=1)

        batch_size = x.size(0)
        latents_batch = self.latents.expand(batch_size, -1, -1)

        latent_out, _ = self.cross_attention(query=latents_batch, key=inputs_seq, value=inputs_seq)

        for attn, norm, ffn, ffn_norm in zip(self.latent_self_attns, self.latent_norms, self.latent_ffns, self.latent_ffn_norms):
            attn_out, _ = attn(query=latent_out, key=latent_out, value=latent_out)
            latent_out = norm(latent_out + attn_out)
            
            ffn_out = ffn(latent_out)
            latent_out = ffn_norm(latent_out + ffn_out)

        # 核心修改：使用全局平均池化替代展平操作，构建信息瓶颈
        latent_pooled = latent_out.mean(dim=1)
        modalities_combined = self.latent_proj(latent_pooled)

        final_combined = torch.cat(
            [
                modalities_combined,
                transe_feat.squeeze(1)
            ],
            dim=1
        )
        
        return self.fusion(final_combined)