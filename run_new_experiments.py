"""
Extend run_all_experiments.py to include PrimeKG and Perceiver experiments.
This script:
1. Creates PrimeKG experiment scripts from DementiaHKG scripts (swap embedding paths)
2. Creates Perceiver experiment scripts from CustomKG_TransE scripts (swap model class)
3. Runs all new experiments
"""
import sys, os, re, subprocess, csv
sys.stdout.reconfigure(encoding='utf-8')

PYTHON = r"D:\anaconda3\envs\audiokeshe\python.exe"
RESULTS_CSV = 'experiment_results_v2.csv'

# All new experiments to run
NEW_EXPERIMENTS = []

DATASETS = ['ADNI', 'AIBL', 'NACC', 'PPMI']

##############################################
# 1. PrimeKG experiments: modify DementiaHKG scripts to use PrimeKG embeddings
##############################################
for ds in DATASETS:
    src = rf'd:\Polaris\Documents\work4\DementiaHKG\DementiaHKG\{ds}_DementiaHKG.py'
    dst = rf'd:\Polaris\Documents\work4\DementiaHKG\DementiaHKG\{ds}_PrimeKG.py'
    
    if os.path.exists(src):
        with open(src, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Replace DementiaHKG embedding paths with PrimeKG paths
        content = content.replace(f'{ds}-DementiaHKG-Embeddings.npy', f'{ds}-PrimeKG-Embeddings.npy')
        content = content.replace(f'{ds}-DementiaHKG-Entity2ID.json', f'{ds}-PrimeKG-Entity2ID.json')
        content = content.replace(f'{ds}-DementiaHKG-Entities.json', f'{ds}-PrimeKG-Entities.json')
        
        with open(dst, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f'Created: {os.path.basename(dst)}')
    
    NEW_EXPERIMENTS.append((f'{ds}_PrimeKG', dst))

##############################################
# 2. Perceiver experiments: modify CustomKG_TransE scripts to use Perceiver model
##############################################
for ds in DATASETS:
    src = rf'd:\Polaris\Documents\work4\Baseline_CustomKG\Baseline&CustomKG\Baseline\{ds}\{ds}_CustomKG_TransE.py'
    dst = rf'd:\Polaris\Documents\work4\Baseline_CustomKG\Baseline&CustomKG\Baseline\{ds}\{ds}_Perceiver.py'
    
    if os.path.exists(src):
        with open(src, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Add KGMultiModalPerceiver to imports
        content = content.replace(
            'from model1 import',
            'from model1 import KGMultiModalPerceiver,'
        )
        # Handle case where KGMultiModalPerceiver already imported via wildcard
        if 'KGMultiModalPerceiver,,' in content:
            content = content.replace('KGMultiModalPerceiver,,', 'KGMultiModalPerceiver,')
        
        # Replace model instantiation - KGMultiModalTransformer_old -> KGMultiModalPerceiver
        content = re.sub(
            r'model\s*=\s*KGMultiModalTransformer_old\((.*?)\)',
            lambda m: f'model = KGMultiModalPerceiver({m.group(1)})' if m.group(1).strip() else 'model = KGMultiModalPerceiver()',
            content, flags=re.DOTALL
        )
        # Also handle KGMultiModalTransformer (non-old)
        content = re.sub(
            r'model\s*=\s*KGMultiModalTransformer\((.*?)\)',
            lambda m: f'model = KGMultiModalPerceiver({m.group(1)})' if m.group(1).strip() else 'model = KGMultiModalPerceiver()',
            content, flags=re.DOTALL
        )
        
        with open(dst, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f'Created: {os.path.basename(dst)}')
    
    NEW_EXPERIMENTS.append((f'{ds}_Perceiver', dst))

##############################################
# 3. Run all new experiments
##############################################
print(f'\n{"="*50}')
print(f'Total new experiments: {len(NEW_EXPERIMENTS)}')
print(f'{"="*50}\n')

results = []

# Load existing results to skip completed
completed = set()
if os.path.exists(RESULTS_CSV):
    with open(RESULTS_CSV, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get('Accuracy') and row['Accuracy'].strip():
                completed.add(row['Experiment'])

for name, script_path in NEW_EXPERIMENTS:
    log_file = f'log_{name}.txt'
    
    # Check if already completed
    if name in completed:
        print(f'[{name}] Already completed. Skipping.')
        continue
    
    # Check log file for completion marker
    if os.path.exists(log_file):
        with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
            log_content = f.read()
        if f'[{name}] Done!' in log_content:
            print(f'[{name}] Found in log. Extracting metrics...')
            # Extract metrics from log
            acc_m = re.search(r'Accuracy:\s+([\d.]+)', log_content)
            prec_m = re.search(r'Precision:\s+([\d.]+)', log_content)
            rec_m = re.search(r'Recall:\s+([\d.]+)', log_content)
            f1_m = re.search(r'F1 Score:\s+([\d.]+)', log_content)
            auc_m = re.search(r'AUC-ROC:\s+([\d.]+)', log_content)
            if acc_m:
                results.append({
                    'Experiment': name,
                    'Accuracy': acc_m.group(1),
                    'Precision': prec_m.group(1) if prec_m else '',
                    'Recall': rec_m.group(1) if rec_m else '',
                    'F1 Score': f1_m.group(1) if f1_m else '',
                    'AUC-ROC': auc_m.group(1) if auc_m else '',
                })
                continue
    
    if not os.path.exists(script_path):
        print(f'[{name}] Script not found: {script_path}')
        results.append({'Experiment': name, 'Accuracy': '', 'Precision': '', 'Recall': '', 'F1 Score': '', 'AUC-ROC': ''})
        continue
    
    print(f'\n{"="*40}')
    print(f'[{name}] Starting experiment...')
    print(f'{"="*40}\n')
    
    cwd = os.path.dirname(script_path)
    
    try:
        with open(log_file, 'w', encoding='utf-8') as log_f:
            process = subprocess.Popen(
                [PYTHON, script_path],
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                cwd=cwd, encoding='utf-8', errors='replace'
            )
            
            output_lines = []
            for line in iter(process.stdout.readline, ''):
                print(line, end='')
                log_f.write(line)
                log_f.flush()
                output_lines.append(line)
            
            process.wait()
        
        full_output = ''.join(output_lines)
        
        # Extract metrics
        acc_m = re.search(r'Accuracy:\s+([\d.]+)', full_output)
        prec_m = re.search(r'Precision:\s+([\d.]+)', full_output)
        rec_m = re.search(r'Recall:\s+([\d.]+)', full_output)
        f1_m = re.search(r'F1 Score:\s+([\d.]+)', full_output)
        auc_m = re.search(r'AUC-ROC:\s+([\d.]+)', full_output)
        
        acc = acc_m.group(1) if acc_m else ''
        prec = prec_m.group(1) if prec_m else ''
        rec = rec_m.group(1) if rec_m else ''
        f1 = f1_m.group(1) if f1_m else ''
        auc = auc_m.group(1) if auc_m else ''
        
        print(f'\n[{name}] Done! -> Acc={acc} Prec={prec} Recall={rec} F1={f1} AUC={auc}')
        
        # Append completion marker
        with open(log_file, 'a', encoding='utf-8') as log_f:
            log_f.write(f'\n[{name}] Done! -> Acc={acc} Prec={prec} Recall={rec} F1={f1} AUC={auc}\n')
        
        results.append({
            'Experiment': name,
            'Accuracy': acc, 'Precision': prec, 'Recall': rec,
            'F1 Score': f1, 'AUC-ROC': auc,
        })
        
    except Exception as e:
        print(f'[{name}] Error: {e}')
        results.append({'Experiment': name, 'Accuracy': '', 'Precision': '', 'Recall': '', 'F1 Score': '', 'AUC-ROC': ''})

# Save results
if results:
    fieldnames = ['Experiment', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC-ROC']
    
    # Merge with existing results
    existing = []
    if os.path.exists(RESULTS_CSV):
        with open(RESULTS_CSV, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            existing = list(reader)
    
    existing_names = {r['Experiment'] for r in existing}
    for r in results:
        if r['Experiment'] not in existing_names:
            existing.append(r)
        else:
            # Update
            for i, e in enumerate(existing):
                if e['Experiment'] == r['Experiment']:
                    existing[i] = r
                    break
    
    with open(RESULTS_CSV, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(existing)
    
    print(f'\nResults saved to {RESULTS_CSV}')

print('\nAll new experiments finished!')
