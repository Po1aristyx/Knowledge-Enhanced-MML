import sys, os, json, torch, numpy as np, pandas as pd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

sys.stdout.reconfigure(encoding='utf-8')

class TripleDataset(Dataset):
    def __init__(self, triples):
        self.triples = triples
    def __len__(self):
        return len(self.triples)
    def __getitem__(self, idx):
        return self.triples[idx]

class ConvE(nn.Module):
    def __init__(self, num_entities, num_relations, embed_dim=128,
                 emb_2d_h=8, emb_2d_w=16, num_filters=32, kernel_size=3, dropout=0.3):
        super().__init__()
        self.ent_emb = nn.Embedding(num_entities, embed_dim)
        self.rel_emb = nn.Embedding(num_relations, embed_dim)
        self.emb_2d_h, self.emb_2d_w = emb_2d_h, emb_2d_w
        self.margin = 1.0

        self.conv = nn.Conv2d(1, num_filters, kernel_size, padding=1)
        self.bn_conv = nn.BatchNorm2d(num_filters)
        self.dropout = nn.Dropout(dropout)

        fc_in = num_filters * emb_2d_h * 2 * emb_2d_w
        self.fc = nn.Linear(fc_in, embed_dim)
        self.bn_fc = nn.BatchNorm1d(embed_dim)

        nn.init.xavier_uniform_(self.ent_emb.weight)
        nn.init.xavier_uniform_(self.rel_emb.weight)

    def forward(self, h_idx, r_idx, t_idx):
        h = self.ent_emb(h_idx).view(-1, 1, self.emb_2d_h, self.emb_2d_w)
        r = self.rel_emb(r_idx).view(-1, 1, self.emb_2d_h, self.emb_2d_w)
        
        x = torch.cat([h, r], dim=2)
        x = self.bn_conv(self.conv(x))
        x = F.relu(x)
        x = self.dropout(x).view(x.size(0), -1)
        x = self.bn_fc(self.fc(x))
        x = F.relu(x)
        x = self.dropout(x)
        
        t = self.ent_emb(t_idx)
        return torch.sum(x * t, dim=-1)

class ConvR(nn.Module):
    def __init__(self, num_entities, num_relations, embed_dim=128,
                 emb_2d_h=8, emb_2d_w=16, num_filters=32, kernel_size=3, dropout=0.3):
        super().__init__()
        self.ent_emb = nn.Embedding(num_entities, embed_dim)
        self.rel_emb = nn.Embedding(num_relations, embed_dim)
        self.emb_2d_h, self.emb_2d_w = emb_2d_h, emb_2d_w
        self.num_filters, self.kernel_size = num_filters, kernel_size
        self.margin = 1.0

        self.rel_to_filter = nn.Linear(embed_dim, num_filters * kernel_size * kernel_size)
        self.bn_conv = nn.BatchNorm2d(num_filters)
        self.dropout = nn.Dropout(dropout)

        conv_out_h = emb_2d_h - kernel_size + 1
        conv_out_w = emb_2d_w - kernel_size + 1
        self.fc = nn.Linear(num_filters * conv_out_h * conv_out_w, embed_dim)
        self.bn_fc = nn.BatchNorm1d(embed_dim)

        nn.init.xavier_uniform_(self.ent_emb.weight)
        nn.init.xavier_uniform_(self.rel_emb.weight)

    def forward(self, h_idx, r_idx, t_idx):
        B = h_idx.size(0)
        h = self.ent_emb(h_idx).view(B, 1, self.emb_2d_h, self.emb_2d_w)
        t = self.ent_emb(t_idx)
        r = self.rel_emb(r_idx)
        
        filters = self.rel_to_filter(r).view(B * self.num_filters, 1, self.kernel_size, self.kernel_size)
        h_g = h.view(1, B, self.emb_2d_h, self.emb_2d_w)
        out = F.conv2d(h_g, filters, groups=B)
        out = out.view(B, self.num_filters, out.size(2), out.size(3))
        
        x = self.bn_conv(out)
        x = F.relu(x)
        x = self.dropout(x).view(B, -1)
        x = self.bn_fc(self.fc(x))
        x = F.relu(x)
        x = self.dropout(x)
        return torch.sum(x * t, dim=-1)

# Checkpoint utilities
CKPT_DIR = "checkpoints"
os.makedirs(CKPT_DIR, exist_ok=True)

def save_checkpoint(model, optimizer, epoch, dataset, method):
    ckpt_path = os.path.join(CKPT_DIR, f"ckpt_{dataset}_{method}.pth")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict()
    }, ckpt_path)
    print(f"  [Checkpoint] Saved at epoch {epoch+1} -> {ckpt_path}")

def load_checkpoint(model, optimizer, dataset, method):
    ckpt_path = os.path.join(CKPT_DIR, f"ckpt_{dataset}_{method}.pth")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        print(f"  [Resume] Loaded checkpoint from epoch {ckpt['epoch']+1} <- {ckpt_path}")
        return ckpt['epoch'] + 1
    return 0

def train_kge(dataset_name, method='ConvE', embed_dim=128, epochs=200, lr=0.001, batch_size=512):
    out_dir = r'd:\Polaris\Documents\work4\DementiaHKG\DementiaHKG'
    emb_path = os.path.join(out_dir, f'{dataset_name}-{method}-Embeddings.npy')
    
    if os.path.exists(emb_path):
        print(f'[{dataset_name} - {method}] Embeddings already generated, skipping.')
        return

    base = rf'd:\Polaris\Documents\work4\DementiaHKG\DementiaHKG\图谱构建\{dataset_name}'
    sub_csv = None
    if os.path.exists(base):
        for fname in os.listdir(base):
            if 'PrimeKG_Sub' in fname and fname.endswith('.csv'):
                sub_csv = os.path.join(base, fname)
                break
            
    if sub_csv is None:
        print(f'[{dataset_name} - {method}] No PrimeKG_Sub.csv found in {base}, skipping')
        return
        
    print(f'[{dataset_name} - {method}] Loading sub-graph from {os.path.basename(sub_csv)}...')
    df = pd.read_csv(sub_csv)
    
    all_entities = sorted(set(df['x_name'].astype(str).tolist() + df['y_name'].astype(str).tolist()))
    all_relations = sorted(set(df['relation'].astype(str).tolist()))
    
    entity2id = {e: i for i, e in enumerate(all_entities)}
    relation2id = {r: i for i, r in enumerate(all_relations)}
    
    print(f'  Entities: {len(entity2id)}, Relations: {len(relation2id)}')
    
    triples = []
    for _, row in df.iterrows():
        h = entity2id[str(row['x_name'])]
        r = relation2id[str(row['relation'])]
        t = entity2id[str(row['y_name'])]
        triples.append([h, r, t])
        
    triples_tensor = torch.LongTensor(triples)
    dataset = TripleDataset(triples_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_entities = len(entity2id)
    
    if method == 'ConvE':
        model = ConvE(num_entities, len(relation2id), embed_dim=embed_dim).to(device)
    else:
        model = ConvR(num_entities, len(relation2id), embed_dim=embed_dim).to(device)
        
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    start_epoch = load_checkpoint(model, optimizer, dataset_name, method)
    
    if start_epoch < epochs:
        print(f'  Training {method} on {device} from epoch {start_epoch+1}...')
        for epoch in range(start_epoch, epochs):
            total_loss = 0
            model.train()
            for batch in loader:
                h, r, t = batch[:, 0].to(device), batch[:, 1].to(device), batch[:, 2].to(device)
                neg_t = torch.randint(0, num_entities, t.shape, device=device)
                
                pos_score = model(h, r, t)
                neg_score = model(h, r, neg_t)
                
                loss = torch.relu(model.margin - pos_score + neg_score).mean()
                
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                total_loss += loss.item()
            
            if (epoch + 1) % 20 == 0 or epoch == 0:
                print(f'  Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.4f}')
                
            if (epoch + 1) % 10 == 0:
                save_checkpoint(model, optimizer, epoch, dataset_name, method)
    else:
        print(f'  Training already completed to {epochs} epochs.')

    # Save embeddings
    model.eval()
    embeddings = model.ent_emb.weight.detach().cpu().numpy()
    
    e2id_path = os.path.join(out_dir, f'{dataset_name}-{method}-Entity2ID.json')
    entities_path = os.path.join(out_dir, f'{dataset_name}-{method}-Entities.json')
    
    np.save(emb_path, embeddings)
    with open(e2id_path, 'w', encoding='utf-8') as f:
        json.dump(entity2id, f, ensure_ascii=False)
        
    # Copy Entities.json from DementiaHKG base
    source_entities_path = os.path.join(out_dir, f'{dataset_name}-DementiaHKG-Entities.json')
    if os.path.exists(source_entities_path):
        import shutil
        shutil.copy2(source_entities_path, entities_path)
    else:
        with open(entities_path, 'w', encoding='utf-8') as f:
            json.dump(all_entities, f, ensure_ascii=False)
    
    print(f'  Saved: {os.path.basename(emb_path)} ({embeddings.shape})')
    print()

if __name__ == '__main__':
    os.chdir(r'd:\Polaris\Documents\work4')
    datasets = ['ADNI', 'AIBL', 'NACC', 'PPMI']
    for ds in datasets:
        train_kge(ds, method='ConvE')
        train_kge(ds, method='ConvR')
    print('All KGE embeddings generation finished.')
