"""
Step 1: Train TransE on PrimeKG_Sub.csv for each dataset
Generates PrimeKG embeddings (same format as DementiaHKG)
"""
import sys, os, json, torch, numpy as np, pandas as pd
sys.stdout.reconfigure(encoding='utf-8')
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

class TransE(nn.Module):
    def __init__(self, num_entities, num_relations, embed_dim=128, margin=1.0):
        super().__init__()
        self.ent_emb = nn.Embedding(num_entities, embed_dim)
        self.rel_emb = nn.Embedding(num_relations, embed_dim)
        nn.init.xavier_uniform_(self.ent_emb.weight)
        nn.init.xavier_uniform_(self.rel_emb.weight)
        self.margin = margin

    def forward(self, h, r, t):
        h_e = self.ent_emb(h)
        r_e = self.rel_emb(r)
        t_e = self.ent_emb(t)
        score = torch.norm(h_e + r_e - t_e, p=2, dim=-1)
        return score

class TripleDataset(Dataset):
    def __init__(self, triples):
        self.triples = triples
    def __len__(self):
        return len(self.triples)
    def __getitem__(self, idx):
        return self.triples[idx]

def train_primekg_transe(dataset_name, embed_dim=128, epochs=100, lr=0.01, batch_size=512):
    base = rf'd:\Polaris\Documents\work4\DementiaHKG\DementiaHKG\图谱构建\{dataset_name}'
    sub_csv = None
    for fname in os.listdir(base):
        if 'PrimeKG_Sub' in fname and fname.endswith('.csv'):
            sub_csv = os.path.join(base, fname)
            break
    
    if sub_csv is None:
        print(f'[{dataset_name}] No PrimeKG_Sub.csv found, skipping')
        return
    
    print(f'[{dataset_name}] Loading PrimeKG subgraph from {os.path.basename(sub_csv)}...')
    df = pd.read_csv(sub_csv)
    print(f'  Triples: {len(df)}')
    
    # Build entity2id and relation2id
    all_entities = sorted(set(df['x_name'].astype(str).tolist() + df['y_name'].astype(str).tolist()))
    all_relations = sorted(set(df['relation'].astype(str).tolist()))
    
    entity2id = {e: i for i, e in enumerate(all_entities)}
    relation2id = {r: i for i, r in enumerate(all_relations)}
    
    print(f'  Entities: {len(entity2id)}, Relations: {len(relation2id)}')
    
    # Build triples tensor
    triples = []
    for _, row in df.iterrows():
        h = entity2id[str(row['x_name'])]
        r = relation2id[str(row['relation'])]
        t = entity2id[str(row['y_name'])]
        triples.append([h, r, t])
    
    triples_tensor = torch.LongTensor(triples)
    dataset = TripleDataset(triples_tensor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # Train TransE
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = TransE(len(entity2id), len(relation2id), embed_dim=embed_dim).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    num_entities = len(entity2id)
    
    print(f'  Training TransE on {device}...')
    for epoch in range(epochs):
        total_loss = 0
        for batch in loader:
            h, r, t = batch[:, 0].to(device), batch[:, 1].to(device), batch[:, 2].to(device)
            
            # Negative sampling
            neg_t = torch.randint(0, num_entities, t.shape, device=device)
            
            pos_score = model(h, r, t)
            neg_score = model(h, r, neg_t)
            
            loss = torch.relu(model.margin + pos_score - neg_score).mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        
        if (epoch + 1) % 20 == 0:
            print(f'  Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(loader):.4f}')
    
    # Save embeddings
    out_dir = rf'd:\Polaris\Documents\work4\DementiaHKG\DementiaHKG'
    embeddings = model.ent_emb.weight.detach().cpu().numpy()
    
    emb_path = os.path.join(out_dir, f'{dataset_name}-PrimeKG-Embeddings.npy')
    e2id_path = os.path.join(out_dir, f'{dataset_name}-PrimeKG-Entity2ID.json')
    entities_path = os.path.join(out_dir, f'{dataset_name}-PrimeKG-Entities.json')
    
    np.save(emb_path, embeddings)
    with open(e2id_path, 'w', encoding='utf-8') as f:
        json.dump(entity2id, f, ensure_ascii=False)
    with open(entities_path, 'w', encoding='utf-8') as f:
        json.dump(all_entities, f, ensure_ascii=False)
    
    print(f'  Saved: {os.path.basename(emb_path)} ({embeddings.shape})')
    print(f'  Saved: {os.path.basename(e2id_path)} ({len(entity2id)} entities)')
    print()

if __name__ == '__main__':
    for ds in ['ADNI', 'AIBL', 'NACC', 'PPMI']:
        train_primekg_transe(ds)
    print('All PrimeKG embeddings generated!')
