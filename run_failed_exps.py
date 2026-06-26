import sys, os, re, subprocess, csv
sys.stdout.reconfigure(encoding='utf-8')

PYTHON = r"D:\anaconda3\envs\audiokeshe\python.exe"
RESULTS_CSV = 'experiment_results_v2.csv'

NEW_EXPERIMENTS = []
DATASETS = ['ADNI', 'AIBL', 'NACC', 'PPMI']

# 1. PrimeKG
for ds in DATASETS:
    dst = rf'd:\Polaris\Documents\work4\DementiaHKG\DementiaHKG\{ds}_PrimeKG.py'
    NEW_EXPERIMENTS.append((f'{ds}_PrimeKG', dst))

# 2. Perceiver
for ds in DATASETS:
    src = rf'd:\Polaris\Documents\work4\Baseline_CustomKG\Baseline&CustomKG\Baseline\{ds}\{ds}_CustomKG_TransE.py'
    dst = rf'd:\Polaris\Documents\work4\Baseline_CustomKG\Baseline&CustomKG\Baseline\{ds}\{ds}_Perceiver.py'
    
    if os.path.exists(src):
        with open(src, 'r', encoding='utf-8') as f:
            content = f.read()
        
        content = content.replace('from model1 import', 'from model1 import KGMultiModalPerceiver,')
        if 'KGMultiModalPerceiver,,' in content:
            content = content.replace('KGMultiModalPerceiver,,', 'KGMultiModalPerceiver,')
        
        # Regex to capture kwargs and extract what we need
        def fix_perceiver(m):
            args = m.group(1)
            new_args = []
            if 'embed_dim' in args: new_args.append('embed_dim=embed_dim')
            if 'transe_embed_dim' in args: new_args.append('transe_embed_dim=transe_embed_dim')
            
            # Special case for PPMI
            nc_match = re.search(r'num_classes=(\d+)', args)
            if nc_match:
                new_args.append(f'num_classes={nc_match.group(1)}')
                
            return f'model = KGMultiModalPerceiver({", ".join(new_args)})'
            
        content = re.sub(r'model\s*=\s*KGMultiModalTransformer_old\((.*?)\)', fix_perceiver, content, flags=re.DOTALL)
        content = re.sub(r'model\s*=\s*KGMultiModalTransformer\((.*?)\)', fix_perceiver, content, flags=re.DOTALL)
        
        with open(dst, 'w', encoding='utf-8') as f:
            f.write(content)
    
    NEW_EXPERIMENTS.append((f'{ds}_Perceiver', dst))

results = []
completed = set()
if os.path.exists(RESULTS_CSV):
    with open(RESULTS_CSV, 'r', encoding='utf-8') as f:
        for row in csv.DictReader(f):
            if row.get('Accuracy') and row['Accuracy'].strip():
                completed.add(row['Experiment'])

for name, script_path in NEW_EXPERIMENTS:
    log_file = f'log_{name}.txt'
    
    if name in completed:
        continue
    
    if os.path.exists(log_file):
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            log_content = f.read()
        if f'[{name}] Done!' in log_content:
            acc_m = re.search(r'Accuracy:\s+([\d.]+)', log_content)
            if acc_m: continue # Already collected
            
    print(f'Running {name}...')
    with open(log_file, 'w', encoding='utf-8') as log_f:
        process = subprocess.Popen(
            [PYTHON, script_path],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            cwd=os.path.dirname(script_path), encoding='utf-8', errors='replace'
        )
        output = ""
        for line in iter(process.stdout.readline, ''):
            output += line
            log_f.write(line)
            log_f.flush()
        process.wait()
        
        acc_m = re.search(r'Accuracy:\s+([\d.]+)', output)
        prec_m = re.search(r'Precision:\s+([\d.]+)', output)
        rec_m = re.search(r'Recall:\s+([\d.]+)', output)
        f1_m = re.search(r'F1 Score:\s+([\d.]+)', output)
        auc_m = re.search(r'AUC-ROC:\s+([\d.]+)', output)
        
        acc = acc_m.group(1) if acc_m else ''
        prec = prec_m.group(1) if prec_m else ''
        rec = rec_m.group(1) if rec_m else ''
        f1 = f1_m.group(1) if f1_m else ''
        auc = auc_m.group(1) if auc_m else ''
        
        with open(log_file, 'a', encoding='utf-8') as f_append:
            f_append.write(f'\n[{name}] Done! -> Acc={acc} Prec={prec} Recall={rec} F1={f1} AUC={auc}\n')
            
        print(f'Finished {name}: Acc={acc} AUC={auc}')
        
        results.append({
            'Experiment': name,
            'Accuracy': acc, 'Precision': prec, 'Recall': rec,
            'F1 Score': f1, 'AUC-ROC': auc,
        })

# Save results
if results:
    existing = []
    if os.path.exists(RESULTS_CSV):
        with open(RESULTS_CSV, 'r', encoding='utf-8') as f:
            existing = list(csv.DictReader(f))
    
    existing_names = {r['Experiment'] for r in existing}
    for r in results:
        if r['Experiment'] not in existing_names:
            existing.append(r)
        else:
            for i, e in enumerate(existing):
                if e['Experiment'] == r['Experiment']:
                    existing[i] = r
                    break
    
    with open(RESULTS_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['Experiment', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC-ROC'])
        writer.writeheader()
        writer.writerows(existing)

print('All done!')
