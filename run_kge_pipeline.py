import os
import sys
import json
import traceback
import shutil

# Force matplotlib to use a non-interactive backend
import matplotlib
matplotlib.use('Agg')

base_dir = r"d:\Polaris\Documents\work4"

def run_notebook(nb_path):
    print(f"\n==================== RUNNING NOTEBOOK: {nb_path} ====================")
    if not os.path.exists(nb_path):
        print(f"Error: File {nb_path} does not exist!")
        return False

    nb_dir = os.path.dirname(os.path.abspath(nb_path))
    original_cwd = os.getcwd()
    os.chdir(nb_dir)
    sys.path.insert(0, nb_dir)

    try:
        with open(os.path.basename(nb_path), 'r', encoding='utf-8') as f:
            nb = json.load(f)
    except Exception as e:
        print(f"Error loading JSON: {e}")
        os.chdir(original_cwd)
        return False

    globals_dict = {
        '__file__': os.path.abspath(nb_path),
        '__name__': '__main__'
    }

    cells = nb.get('cells', [])
    code_cells = [c for c in cells if c.get('cell_type') == 'code']
    print(f"Found {len(code_cells)} code cells.")

    for i, cell in enumerate(code_cells):
        source = cell.get('source', [])
        if not source:
            continue
        
        clean_lines = []
        for line in source:
            stripped = line.strip()
            if stripped.startswith('%') or stripped.startswith('!'):
                clean_lines.append(f"# {line}")
            else:
                clean_lines.append(line)
        
        code_str = "".join(clean_lines)
        
        first_line = clean_lines[0].strip() if clean_lines else ""
        print(f"--- Cell {i+1}/{len(code_cells)} (Starts with: {first_line[:50]}...) ---")
        
        try:
            exec(code_str, globals_dict)
        except Exception as e:
            print(f"❌ Exception in cell {i+1}:")
            traceback.print_exc()
            os.chdir(original_cwd)
            return False

    print(f"==================== NOTEBOOK COMPLETED SUCCESSFULLY ====================\n")
    os.chdir(original_cwd)
    if nb_dir in sys.path:
        sys.path.remove(nb_dir)
    return True

# Order of notebooks to run
kge_tasks = [
    # 1. ADNI DementiaHKG Build
    {
        "notebooks": [
            r"DementiaHKG\DementiaHKG\图谱构建\ADNI\Step1-Search-ADNI-DiseaseNames.ipynb",
            r"DementiaHKG\DementiaHKG\图谱构建\ADNI\Step2-Search-ADNI-SeedNodes.ipynb",
            r"DementiaHKG\DementiaHKG\图谱构建\ADNI\Step3-Build-ADNI-DementiaHKG.ipynb"
        ],
        "copies": [
            (r"DementiaHKG\DementiaHKG\图谱构建\ADNI\ADNI-DementiaHKG-Embeddings.npy", r"DementiaHKG\DementiaHKG\ADNI-DementiaHKG-Embeddings.npy"),
            (r"DementiaHKG\DementiaHKG\图谱构建\ADNI\ADNI-DementiaHKG-Entity2ID.json", r"DementiaHKG\DementiaHKG\ADNI-DementiaHKG-Entity2ID.json"),
            (r"DementiaHKG\DementiaHKG\图谱构建\ADNI\ADNI-DementiaHKG-Entities.json", r"DementiaHKG\DementiaHKG\ADNI-DementiaHKG-Entities.json")
        ]
    },
    # 2. AIBL DementiaHKG Build
    {
        "notebooks": [
            r"DementiaHKG\DementiaHKG\图谱构建\AIBL\Step1-Search-AIBL-DiseaseNames.ipynb",
            r"DementiaHKG\DementiaHKG\图谱构建\AIBL\Step2-Search-AIBL-SeedNodes.ipynb",
            r"DementiaHKG\DementiaHKG\图谱构建\AIBL\Step3-Build-AIBL-DementiaHKG.ipynb"
        ],
        "copies": [
            (r"DementiaHKG\DementiaHKG\图谱构建\AIBL\AIBL-DementiaHKG-Embeddings.npy", r"DementiaHKG\DementiaHKG\AIBL-DementiaHKG-Embeddings.npy"),
            (r"DementiaHKG\DementiaHKG\图谱构建\AIBL\AIBL-DementiaHKG-Entity2ID.json", r"DementiaHKG\DementiaHKG\AIBL-DementiaHKG-Entity2ID.json"),
            (r"DementiaHKG\DementiaHKG\图谱构建\AIBL\AIBL-DementiaHKG-Entities.json", r"DementiaHKG\DementiaHKG\AIBL-DementiaHKG-Entities.json")
        ]
    },
    # 3. NACC DementiaHKG Build (Skipping Step 2.5 because Step 2 already has the resolved terms)
    {
        "notebooks": [
            r"DementiaHKG\DementiaHKG\图谱构建\NACC\Step1-Search-NACC-DiseaseNames.ipynb",
            r"DementiaHKG\DementiaHKG\图谱构建\NACC\Step2-Search-NACC-SeedNodes.ipynb",
            r"DementiaHKG\DementiaHKG\图谱构建\NACC\Step3-Build-NACC-DementiaHKG.ipynb"
        ],
        "copies": [
            (r"DementiaHKG\DementiaHKG\图谱构建\NACC\NACC-DementiaHKG-Embeddings.npy", r"DementiaHKG\DementiaHKG\NACC-DementiaHKG-Embeddings.npy"),
            (r"DementiaHKG\DementiaHKG\图谱构建\NACC\NACC-DementiaHKG-Entity2ID.json", r"DementiaHKG\DementiaHKG\NACC-DementiaHKG-Entity2ID.json"),
            (r"DementiaHKG\DementiaHKG\图谱构建\NACC\NACC-DementiaHKG-Entities.json", r"DementiaHKG\DementiaHKG\NACC-DementiaHKG-Entities.json")
        ]
    },
    # 4. PPMI DementiaHKG Build
    {
        "notebooks": [
            r"DementiaHKG\DementiaHKG\图谱构建\PPMI\Step1-Search-PPMI-DiseaseNames.ipynb",
            r"DementiaHKG\DementiaHKG\图谱构建\PPMI\Step2-Search-PPMI-SeedNodes.ipynb",
            r"DementiaHKG\DementiaHKG\图谱构建\PPMI\Step3-Build-PPMI-DementiaHKG.ipynb"
        ],
        "copies": [
            (r"DementiaHKG\DementiaHKG\图谱构建\PPMI\PPMI-DementiaHKG-Embeddings.npy", r"DementiaHKG\DementiaHKG\PPMI-DementiaHKG-Embeddings.npy"),
            (r"DementiaHKG\DementiaHKG\图谱构建\PPMI\PPMI-DementiaHKG-Entity2ID.json", r"DementiaHKG\DementiaHKG\PPMI-DementiaHKG-Entity2ID.json"),
            (r"DementiaHKG\DementiaHKG\图谱构建\PPMI\PPMI-DementiaHKG-Entities.json", r"DementiaHKG\DementiaHKG\PPMI-DementiaHKG-Entities.json")
        ]
    },
    # 5. CustomKG Builds
    {
        "notebooks": [
            r"Baseline_CustomKG\Baseline&CustomKG\CustomKG\图谱构建\ADNI\build-ADNI-CustomKG.ipynb",
            r"Baseline_CustomKG\Baseline&CustomKG\CustomKG\图谱构建\AIBL\build-AIBL-CustomKG.ipynb",
            r"Baseline_CustomKG\Baseline&CustomKG\CustomKG\图谱构建\NACC\build-NACC-CustomKG.ipynb",
            r"Baseline_CustomKG\Baseline&CustomKG\CustomKG\图谱构建\PPMI\build-PPMI-CustomKG.ipynb"
        ],
        "copies": [
            (r"Baseline_CustomKG\Baseline&CustomKG\CustomKG\图谱构建\ADNI\CustomKG-ADNI.npy", r"Baseline_CustomKG\Baseline&CustomKG\CustomKG\CustomKG-ADNI.npy"),
            (r"Baseline_CustomKG\Baseline&CustomKG\CustomKG\图谱构建\ADNI\CustomKG-ADNI.txt", r"Baseline_CustomKG\Baseline&CustomKG\CustomKG\CustomKG-ADNI.txt"),
            
            (r"Baseline_CustomKG\Baseline&CustomKG\CustomKG\图谱构建\AIBL\CustomKG-AIBL.npy", r"Baseline_CustomKG\Baseline&CustomKG\CustomKG\CustomKG-AIBL.npy"),
            (r"Baseline_CustomKG\Baseline&CustomKG\CustomKG\图谱构建\AIBL\CustomKG-AIBL.txt", r"Baseline_CustomKG\Baseline&CustomKG\CustomKG\CustomKG-AIBL.txt"),
            
            (r"Baseline_CustomKG\Baseline&CustomKG\CustomKG\图谱构建\NACC\CustomKG-NACC.npy", r"Baseline_CustomKG\Baseline&CustomKG\CustomKG\CustomKG-NACC.npy"),
            (r"Baseline_CustomKG\Baseline&CustomKG\CustomKG\图谱构建\NACC\CustomKG-NACC.txt", r"Baseline_CustomKG\Baseline&CustomKG\CustomKG\CustomKG-NACC.txt"),
            
            (r"Baseline_CustomKG\Baseline&CustomKG\CustomKG\图谱构建\PPMI\CustomKG-PPMI.npy", r"Baseline_CustomKG\Baseline&CustomKG\CustomKG\CustomKG-PPMI.npy"),
            (r"Baseline_CustomKG\Baseline&CustomKG\CustomKG\图谱构建\PPMI\CustomKG-PPMI.txt", r"Baseline_CustomKG\Baseline&CustomKG\CustomKG\CustomKG-PPMI.txt")
        ]
    }
]

def main():
    print("====== KGE GRAPH AND EMBEDDING CONSTRUCTION PIPELINE ======")
    for task in kge_tasks:
        # Run notebooks
        for nb in task["notebooks"]:
            nb_full = os.path.join(base_dir, nb)
            success = run_notebook(nb_full)
            if not success:
                print(f"❌ Pipeline aborted: Notebook {nb} failed.")
                sys.exit(1)
        
        # Copy output files
        for src, dest in task["copies"]:
            src_full = os.path.join(base_dir, src)
            dest_full = os.path.join(base_dir, dest)
            if os.path.exists(src_full):
                try:
                    shutil.copy2(src_full, dest_full)
                    print(f"Copied output: {src} -> {dest}")
                except Exception as e:
                    print(f"Error copying {src} to {dest}: {e}")
            else:
                print(f"Warning: Expected output {src} not found!")

    print("\n🎉 ALL KGE EMBEDDINGS AND GRAPHS BUILT SUCCESSFULLY!")

if __name__ == "__main__":
    main()
