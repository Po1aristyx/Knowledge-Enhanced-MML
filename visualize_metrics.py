import os
import re
import matplotlib.pyplot as plt

def parse_log(filepath):
    records = []
    # Super flexible regex for matching train loss/auc and test loss/auc
    epoch_pattern = re.compile(r"Epoch\s+(\d+).*?Loss:?\s*([0-9.]+).*?AUC:?\s*([0-9.]+).*?Loss:?\s*([0-9.]+).*?AUC:?\s*([0-9.]+)", re.IGNORECASE)
    with open(filepath, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            m = epoch_pattern.search(line)
            if m:
                records.append({
                    "epoch": int(m.group(1)),
                    "train_loss": float(m.group(2)),
                    "train_auc": float(m.group(3)),
                    "test_loss": float(m.group(4)),
                    "test_auc": float(m.group(5)),
                })
    return records

def smooth_curve(points, factor=0.75):
    smoothed = []
    for point in points:
        if smoothed:
            prev = smoothed[-1]
            smoothed.append(prev * factor + point * (1 - factor))
        else:
            smoothed.append(point)
    return smoothed

def generate_loss_auc_curves(log_dir, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    
    best_logs = {
        "NACC": "log_NACC_DementiaHKG.txt",
        "AIBL": "log_AIBL_PrimeKG.txt",
        "ADNI": "log_ADNI_PrimeKG.txt",
        "PPMI": "log_PPMI_PrimeKG.txt"
    }
    
    for ds, log_name in best_logs.items():
        target_log = os.path.join(log_dir, log_name)
        if not os.path.exists(target_log):
            continue
            
        recs = parse_log(target_log)
        if not recs:
            continue
            
        epochs = [r['epoch'] for r in recs]
        train_loss = smooth_curve([r['train_loss'] for r in recs])
        test_loss = smooth_curve([r['test_loss'] for r in recs])
        train_auc = smooth_curve([r['train_auc'] for r in recs])
        test_auc = smooth_curve([r['test_auc'] for r in recs])
        
        plt.figure(figsize=(6,4))
        plt.plot(epochs, train_loss, label='Train Loss', color='#1f77b4')
        plt.plot(epochs, test_loss, label='Test Loss', color='#ff7f0e')
        plt.title(f'{ds} - Loss Curve (ConvKAMT)')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'{ds}_loss_curve.png'), dpi=150)
        plt.close()
        
        plt.figure(figsize=(6,4))
        plt.plot(epochs, train_auc, label='Train AUC', color='#2ca02c')
        plt.plot(epochs, test_auc, label='Test AUC', color='#d62728')
        plt.title(f'{ds} - AUC Curve (ConvKAMT)')
        plt.xlabel('Epoch')
        plt.ylabel('AUC')
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, f'{ds}_auc_curve.png'), dpi=150)
        plt.close()
        print(f"Successfully generated smooth curves for {ds}")

if __name__ == '__main__':
    workspace = os.path.abspath(os.path.dirname(__file__))
    generate_loss_auc_curves(workspace, os.path.join(workspace, 'figures'))
