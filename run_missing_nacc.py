import os, subprocess, re, csv
import sys

sys.stdout.reconfigure(encoding='utf-8')
PYTHON = r"D:\anaconda3\envs\audiokeshe\python.exe"

scripts_to_run = [
    ("NACC_Baseline_Dual", r"d:\Polaris\Documents\work4\Baseline_CustomKG\Baseline&CustomKG\Baseline\NACC\NACC_Baseline_Dual.py"),
    ("NACC_Baseline_Single", r"d:\Polaris\Documents\work4\Baseline_CustomKG\Baseline&CustomKG\Baseline\NACC\NACC_Baseline_Single.py"),
    ("NACC_CustomKG_TransE", r"d:\Polaris\Documents\work4\Baseline_CustomKG\Baseline&CustomKG\Baseline\NACC\NACC_CustomKG_TransE.py"),
    ("NACC_DementiaHKG", r"d:\Polaris\Documents\work4\DementiaHKG\DementiaHKG\NACC_DementiaHKG.py")
]

results = []

for name, script in scripts_to_run:
    print(f"Running {name}...")
    log_file = f"d:\\Polaris\\Documents\\work4\\log_{name}.txt"
    with open(log_file, "w", encoding="utf-8") as lf:
        proc = subprocess.Popen([PYTHON, script], stdout=subprocess.PIPE, stderr=subprocess.STDOUT, encoding='utf-8', errors='replace', cwd=os.path.dirname(script))
        output = ""
        for line in iter(proc.stdout.readline, ''):
            output += line
            lf.write(line)
            lf.flush()
        proc.wait()
        
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
            
        print(f"Finished {name}: Acc={acc} AUC={auc}")
        
        results.append({
            'Experiment': name,
            'Accuracy': acc, 'Precision': prec, 'Recall': rec,
            'F1 Score': f1, 'AUC-ROC': auc,
        })

# Update CSV
if results:
    csv_path = r"d:\Polaris\Documents\work4\experiment_results.csv"
    existing = []
    if os.path.exists(csv_path):
        with open(csv_path, 'r', encoding='utf-8') as f:
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
                    
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=['Experiment', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC-ROC'])
        writer.writeheader()
        writer.writerows(existing)

print("All NACC scripts finished and CSV updated.")

# Automatically run visualizations
print("Running visualize_metrics.py...")
subprocess.run([PYTHON, r"d:\Polaris\Documents\work4\visualize_metrics.py"], cwd=r"d:\Polaris\Documents\work4")

# Automatically regenerate report
print("Running gen_report_from_template.py...")
subprocess.run([PYTHON, r"d:\Polaris\Documents\work4\gen_report_from_template.py"], cwd=r"d:\Polaris\Documents\work4")

print("Done! NACC is fully processed.")
