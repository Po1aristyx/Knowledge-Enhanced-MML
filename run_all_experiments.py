import os
import subprocess
import re
import csv
import json
import sys

MODEL1_DIR = r'd:\Polaris\Documents\work4\Baseline_CustomKG\Baseline&CustomKG\Baseline\ADNI'

def convert_nb_to_py(nb_path, out_py_path, ds_name, exp_type):
    with open(nb_path, 'r', encoding='utf-8') as f:
        nb = json.load(f)

    code_lines = []

    # 1. Inject header
    header = "import sys, os\n"
    header += "sys.path.insert(0, r'" + MODEL1_DIR + "')\n"
    header += "os.chdir(r'" + os.path.dirname(nb_path) + "')\n"
    header += "import warnings\n"
    header += "warnings.filterwarnings('ignore')\n"
    code_lines.append(header + '\n\n')

    for cell in nb.get('cells', []):
        if cell.get('cell_type') == 'code':
            source = cell.get('source', [])
            for line in source:
                if line.strip().startswith('%') or line.strip().startswith('!'):
                    code_lines.append('# ' + line)
                else:
                    code_lines.append(line)
            code_lines.append('\n\n')

    content = "".join(code_lines)

    # 2. Patch relative NII paths for NACC
    if ds_name == 'NACC':
        content = re.sub(r"(['\"])NACC_nii_ad\1", r"\1E:/code/NACC_nii_ad/NACC_nii_ad\1", content)
        content = re.sub(r"(['\"])NACC_nii_no\1", r"\1E:/code/NACC_nii_no/NACC_nii_no\1", content)
        content = re.sub(r"(['\"])NACC_mci\1", r"\1E:/code/NACC_mci/NACC_mci\1", content)
        content = re.sub(r"(['\"])NACC_nii_MCI\1", r"\1E:/code/NACC_mci/NACC_mci\1", content)

    # 3. Patch relative NII paths for PPMI
    if ds_name == 'PPMI':
        content = re.sub(r"(['\"])Control\1", r"\1E:/code/PPMI/Control\1", content)
        content = re.sub(r"(['\"])PD\1", r"\1E:/code/PPMI/PD\1", content)
        content = re.sub(r"(['\"])SWEDD\1", r"\1E:/code/PPMI/SWEDD\1", content)

    # 4. Patch TransE missing ckpt
    if 'CustomKG_TransE' in exp_type:
        content = re.sub(r"transe_model\.load_state_dict\(.*?\)", "# Removed load_state_dict", content)

    # 5. Patch DementiaHKG sequence mismatch AND absolute CSV paths!
    if 'DementiaHKG' in exp_type:
        if ds_name == 'AIBL':
            csv_dir = r"d:/Polaris/Documents/work4/Baseline_CustomKG/Baseline&CustomKG/Baseline/AIBL"
        elif ds_name == 'NACC':
            csv_dir = r"d:/Polaris/Documents/work4/Baseline_CustomKG/Baseline&CustomKG/Baseline/NACC"
        elif ds_name == 'PPMI':
            csv_dir = r"d:/Polaris/Documents/work4/Baseline_CustomKG/Baseline&CustomKG/Baseline/PPMI"
        elif ds_name == 'ADNI':
            csv_dir = r"d:/Polaris/Documents/work4/Baseline_CustomKG/Baseline&CustomKG/Baseline/ADNI"
            
        content = re.sub(r"(['\"])AD\.csv\1", f"r'{csv_dir}/AD.csv'", content)
        content = re.sub(r"(['\"])mci\.csv\1", f"r'{csv_dir}/mci.csv'", content)
        content = re.sub(r"(['\"])MCI\.csv\1", f"r'{csv_dir}/MCI.csv'", content)
        content = re.sub(r"(['\"])normal\.csv\1", f"r'{csv_dir}/normal.csv'", content)
        content = re.sub(r"(['\"])NC\.csv\1", f"r'{csv_dir}/NC.csv'", content)
        content = re.sub(r"(['\"])PPMI\.csv\1", f"r'{csv_dir}/PPMI.csv'", content)
        content = re.sub(r"(['\"])control\.csv\1", f"r'{csv_dir}/control.csv'", content)
        content = re.sub(r"(['\"])PD1\.csv\1", f"r'{csv_dir}/PD1.csv'", content)
        content = re.sub(r"(['\"])prodromal\.csv\1", f"r'{csv_dir}/prodromal.csv'", content)
        content = re.sub(r"(['\"])swedd\.csv\1", f"r'{csv_dir}/swedd.csv'", content)
        
        content = re.sub(r"pd\.read_csv\(\s*r'[^']*AD\.csv'\s*\)", f"pd.DataFrame(data_ad, columns=pd.read_csv(r'{csv_dir}/AD.csv').columns)", content)
        content = re.sub(r"pd\.read_csv\(\s*r'[^']*mci\.csv'\s*\)", f"pd.DataFrame(data_mci, columns=pd.read_csv(r'{csv_dir}/mci.csv').columns)", content)
        content = re.sub(r"pd\.read_csv\(\s*r'[^']*MCI\.csv'\s*\)", f"pd.DataFrame(data_mci, columns=pd.read_csv(r'{csv_dir}/MCI.csv').columns)", content)
        content = re.sub(r"pd\.read_csv\(\s*r'[^']*normal\.csv'\s*\)", f"pd.DataFrame(data_normal, columns=pd.read_csv(r'{csv_dir}/normal.csv').columns)", content)
        content = re.sub(r"pd\.read_csv\(\s*r'[^']*NC\.csv'\s*\)", f"pd.DataFrame(data_normal, columns=pd.read_csv(r'{csv_dir}/NC.csv').columns)", content)
        
        # ALSO fix the hardcoded AD.csv in DementiaHKG's process_and_verify_seq calls for AIBL where they might use data_normal/data_mci but columns=pd.read_csv('NC.csv')
        content = re.sub(r"pd\.read_csv\(\s*r'[^']*PPMI\.csv'\s*\)", f"pd.DataFrame(data_ad, columns=pd.read_csv(r'{csv_dir}/PPMI.csv').columns)", content)
        content = re.sub(r"pd\.read_csv\(\s*r'[^']*control\.csv'\s*\)", f"pd.DataFrame(data_normal, columns=pd.read_csv(r'{csv_dir}/control.csv').columns)", content)
        content = re.sub(r"pd\.read_csv\(\s*r'[^']*swedd\.csv'\s*\)", f"pd.DataFrame(data_mci, columns=pd.read_csv(r'{csv_dir}/swedd.csv').columns)", content)

    # 6. For ADNI, AIBL, NACC, and PPMI TransE: replace nii_folders dict values directly
    if 'CustomKG_TransE' in exp_type and ds_name in ['ADNI', 'AIBL', 'NACC', 'PPMI']:
        if ds_name == 'ADNI':
            content = re.sub(r"'ad':\s*'[^']*'", "'ad': 'E:/code/ADNI/ad_nii_KG'", content)
            content = re.sub(r"'normal':\s*'[^']*'", "'normal': 'E:/code/ADNI/normal_nii_kg'", content)
            content = re.sub(r"'mci':\s*'[^']*'", "'mci': 'E:/code/ADNI/NC_nii_KG'", content)
        elif ds_name == 'AIBL':
            content = re.sub(r"'ad':\s*'[^']*'", "'ad': 'E:/code/AIBL/AD'", content)
            content = re.sub(r"'normal':\s*'[^']*'", "'normal': 'E:/code/AIBL/NC'", content)
            content = re.sub(r"'mci':\s*'[^']*'", "'mci': 'E:/code/AIBL/MCI'", content)
        elif ds_name == 'NACC':
            content = re.sub(r"'ad':\s*'[^']*'", "'ad': 'E:/code/NACC_nii_ad/NACC_nii_ad'", content)
            content = re.sub(r"'normal':\s*'[^']*'", "'normal': 'E:/code/NACC_nii_no/NACC_nii_no'", content)
            content = re.sub(r"'mci':\s*'[^']*'", "'mci': 'E:/code/NACC_mci/NACC_mci'", content)
        elif ds_name == 'PPMI':
            content = re.sub(r"'ad':\s*'[^']*'", "'ad': 'E:/code/PPMI/PD'", content)
            content = re.sub(r"'normal':\s*'[^']*'", "'normal': 'E:/code/PPMI/Control'", content)
            content = re.sub(r"'mci':\s*'[^']*'", "'mci': 'E:/code/PPMI/SWEDD'", content)
            content = re.sub(r"(['\"])Prodromal\1\s*:\s*(['\"])Prodromal\2", r"\1Prodromal\1: 'E:/code/PPMI/Prodromal'", content)
            
        # Patch TransE model initializations to use 1,1,1 since all modalities are mapped to 1
        content = re.sub(r"model\s*=\s*KGMultiModalTransformer_old\(embed_dim=embed_dim,\s*transe_embed_dim=transe_embed_dim\)", 
                         r"model = KGMultiModalTransformer_old(ehr_dim=1, img_dim=1, bio_dim=1, embed_dim=embed_dim, transe_embed_dim=transe_embed_dim)", 
                         content)
        content = re.sub(r"model\s*=\s*KGMultiModalTransformer\(embed_dim=embed_dim,\s*transe_embed_dim=transe_embed_dim\)", 
                         r"model = KGMultiModalTransformer_old(ehr_dim=1, img_dim=1, bio_dim=1, embed_dim=embed_dim, transe_embed_dim=transe_embed_dim)", 
                         content)
        # Also patch imports to import KGMultiModalTransformer_old
        content = re.sub(r"from\s+model1\s+import\s+.*?(KGMultiModalTransformer(?!_old)).*", 
                         lambda m: m.group(0).replace('KGMultiModalTransformer', 'KGMultiModalTransformer_old'), 
                         content)

    # 7. Remove max_seq_len from DementiaHKG scripts
    if 'DementiaHKG' in exp_type:
        content = re.sub(r",\s*max_seq_len\s*=\s*(?:max_seq_len|\d+)", "", content)
        # Prevent NaN from MultiheadAttention when all tokens are padded
        content = re.sub(
            r"return\s+torch\.stack\(vecs\),\s*torch\.tensor\(pad_masks,\s*dtype=torch\.bool\)",
            r"pad_t = torch.tensor(pad_masks, dtype=torch.bool)\n    if torch.all(pad_t): pad_t[0] = False\n    return torch.stack(vecs), pad_t",
            content
        )

    # 8. Add num_classes=4 for all PPMI models since it has 4 classes
    if ds_name == 'PPMI':
        content = re.sub(r"model\s*=\s*DualTransformer\((.*?)\)", 
                         lambda m: f"model = DualTransformer({m.group(1)}, num_classes=4)" if m.group(1).strip() else "model = DualTransformer(num_classes=4)", 
                         content, flags=re.DOTALL)
        content = re.sub(r"model\s*=\s*KGMultiModalTransformer_old\((.*?)\)", 
                         lambda m: f"model = KGMultiModalTransformer_old({m.group(1)}, num_classes=4)" if m.group(1).strip() else "model = KGMultiModalTransformer_old(num_classes=4)", 
                         content, flags=re.DOTALL)
        content = re.sub(r"model\s*=\s*KGMultiModalTransformer\((.*?)\)", 
                         lambda m: f"model = KGMultiModalTransformer({m.group(1)}, num_classes=4)" if m.group(1).strip() else "model = KGMultiModalTransformer(num_classes=4)", 
                         content, flags=re.DOTALL)
        
        if 'DementiaHKG' in exp_type:
            content = re.sub(r"control_df\s*=\s*pd\.DataFrame\(data_normal,\s*columns=pd\.read_csv\((.*?)\)\.columns\)", r"control_df = pd.read_csv(\1)", content)
            content = re.sub(r"swedd_df\s*=\s*pd\.DataFrame\(data_mci,\s*columns=pd\.read_csv\((.*?)\)\.columns\)", r"swedd_df = pd.read_csv(\1)", content)
            content = re.sub(r"control_df\s*=\s*pd\.DataFrame\(data_normal,\s*columns=pd\.read_csv\((.*?)\)\.columns\)", r"control_df = pd.read_csv(\1)", content)
            content = re.sub(r"swedd_df\s*=\s*pd\.DataFrame\(data_mci,\s*columns=pd\.read_csv\((.*?)\)\.columns\)", r"swedd_df = pd.read_csv(\1)", content)


    with open(out_py_path, 'w', encoding='utf-8') as f:
        f.write(content)

def should_suppress(line):
    stripped = line.strip()
    if stripped and stripped == '-' * len(stripped) and len(stripped) > 5:
        return True
    if 'UndefinedMetricWarning' in line:
        return True
    if '_warn_prf(' in line:
        return True
    if 'Use `zero_division`' in line:
        return True
    if '_prf(average, modifier' in line:
        return True
    if 'Unsupported operator' in line:
        return True
    return False

def run_experiment(name, nb_path, python_exe="python"):
    print("")
    print("======================================")
    print("[" + name + "] Starting experiment...")
    print("======================================")
    print("")

    ds_name = name.split('_')[0]
    exp_type = name.replace(ds_name + "_", "")

    py_path = nb_path.replace('.ipynb', '.py')
    convert_nb_to_py(nb_path, py_path, ds_name, exp_type)

    log_content = []
    log_path = os.path.join(os.path.dirname(nb_path), "log_" + name + ".txt")
    
    process = subprocess.Popen(
        [python_exe, py_path],
        stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
        text=True, encoding='utf-8', errors='ignore',
        cwd=os.path.dirname(nb_path)
    )

    with open(log_path, 'w', encoding='utf-8') as log_file:
        for line in iter(process.stdout.readline, ''):
            if not should_suppress(line):
                try:
                    sys.stdout.write(line)
                except UnicodeEncodeError:
                    sys.stdout.write(line.encode('gbk', errors='replace').decode('gbk'))
                sys.stdout.flush()
            log_file.write(line)
            log_content.append(line)

    process.wait()
    content = "".join(log_content)

    log_path = os.path.join(os.path.dirname(nb_path), "log_" + name + ".txt")
    with open(log_path, 'w', encoding='utf-8', errors='ignore') as log_f:
        log_f.write(content)

    metrics = {'Accuracy': '', 'Precision': '', 'Recall': '', 'F1 Score': '', 'AUC-ROC': ''}
    try:
        match_acc  = re.search(r'Accuracy:\s*([0-9.]+)', content)
        match_prec = re.search(r'Precision:\s*([0-9.]+)', content)
        match_rec  = re.search(r'Recall:\s*([0-9.]+)', content)
        match_f1   = re.search(r'F1 Score:\s*([0-9.]+)', content)
        match_auc  = re.search(r'AUC-ROC:\s*([0-9.]+)', content)

        if match_acc:  metrics['Accuracy']  = match_acc.group(1)
        if match_prec: metrics['Precision'] = match_prec.group(1)
        if match_rec:  metrics['Recall']    = match_rec.group(1)
        if match_f1:   metrics['F1 Score']  = match_f1.group(1)
        if match_auc:  metrics['AUC-ROC']   = match_auc.group(1)
    except Exception as e:
        print("[" + name + "] Error parsing metrics: " + str(e))

    print("")
    print("[" + name + "] Done! -> Acc=" + metrics['Accuracy'] + " Prec=" + metrics['Precision'] + " Recall=" + metrics['Recall'] + " F1=" + metrics['F1 Score'] + " AUC=" + metrics['AUC-ROC'])
    print("")
    return metrics


def main():
    base1 = r'd:\Polaris\Documents\work4\Baseline_CustomKG\Baseline&CustomKG\Baseline'
    base2 = r'd:\Polaris\Documents\work4\DementiaHKG\DementiaHKG'

    experiments = []
    datasets = ['ADNI', 'AIBL', 'NACC', 'PPMI']

    for ds in datasets:
        experiments.append((ds + "_Baseline_Single",  os.path.join(base1, ds, ds + "_Baseline_Single.ipynb")))
        experiments.append((ds + "_Baseline_Dual",    os.path.join(base1, ds, ds + "_Baseline_Dual.ipynb")))
        experiments.append((ds + "_CustomKG_TransE",  os.path.join(base1, ds, ds + "_CustomKG_TransE.ipynb")))
        experiments.append((ds + "_DementiaHKG",      os.path.join(base2, ds + "_DementiaHKG.ipynb")))

    results = []

    for name, path in experiments:
        if not os.path.exists(path):
            print("Skipping " + name + ": file not found at " + path)
            continue

        log_path = os.path.join(os.path.dirname(path), f"log_{name}.txt")
        metrics = None
        if os.path.exists(log_path):
            with open(log_path, 'r', encoding='utf-8') as f:
                content = f.read()
            if "Accuracy:" in content and "Traceback" not in content[-1000:]:
                print(f"[{name}] Already completed previously. Extracting metrics from log...")
                metrics = {'Accuracy': '', 'Precision': '', 'Recall': '', 'F1 Score': '', 'AUC-ROC': ''}
                acc_m = re.search(r'Accuracy:\s+([\d.]+)', content)
                rec_m = re.search(r'Recall:\s+([\d.]+)', content)
                f1_m = re.search(r'F1 Score:\s+([\d.]+)', content)
                pre_m = re.search(r'Precision:\s+([\d.]+)', content)
                auc_m = re.search(r'AUC-ROC:\s+([\d.]+)', content)
                if not auc_m:
                    auc_m = re.search(r'AUC Score.*?([\d.]+)', content)
                
                if acc_m and rec_m and f1_m and pre_m and auc_m:
                    metrics = {
                        'Accuracy': acc_m.group(1),
                        'Precision': pre_m.group(1),
                        'Recall': rec_m.group(1),
                        'F1 Score': f1_m.group(1),
                        'AUC-ROC': auc_m.group(1)
                    }
        
        if not metrics:
            metrics = run_experiment(name, path, python_exe=r"D:\anaconda3\envs\audiokeshe\python.exe")
            
        metrics['Experiment'] = name
        results.append(metrics)

        with open('experiment_results.csv', 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['Experiment', 'Accuracy', 'Precision', 'Recall', 'F1 Score', 'AUC-ROC']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for r in results:
                writer.writerow(r)

    print("")
    print("All experiments finished. Results saved to experiment_results.csv")


if __name__ == '__main__':
    main()
