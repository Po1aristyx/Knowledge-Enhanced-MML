import os
import csv

def main():
    mri_dir = "E:/code/ADNI/MRI"
    if not os.path.exists(mri_dir):
        print(f"Error: {mri_dir} does not exist!")
        return

    mri_files = os.listdir(mri_dir)
    print(f"Found {len(mri_files)} files in {mri_dir}")

    # CSV sources
    csv_dir = "Baseline_CustomKG/Baseline&CustomKG/诊断记录表/ADNI"
    csvs = {
        "normal.csv": "normal_nii_kg",
        "AD.csv": "ad_nii_KG",
        "mci.csv": "NC_nii_KG"
    }

    # Prepare mapping function
    def get_best_match(csv_fn):
        parts = csv_fn.split('_')
        if len(parts) < 4:
            return None
        # Find subject ID: e.g. XXX_S_XXXX
        sub_id = None
        for i in range(len(parts) - 2):
            if parts[i].isdigit() and parts[i+1] == 'S' and parts[i+2].isdigit():
                sub_id = f"{parts[i]}_S_{parts[i+2]}"
                break
        if not sub_id:
            return None
        
        # Filter files by subject ID
        sub_matches = [f for f in mri_files if sub_id in f]
        if not sub_matches:
            return None
        
        # Filter files by check date
        date_part = None
        for p in parts:
            if (p.startswith('200') or p.startswith('201') or p.startswith('202')) and len(p) >= 8:
                date_part = p[:8]
                break
        if date_part:
            date_matches = [f for f in sub_matches if date_part in f]
            if date_matches:
                return date_matches[0]
        return sub_matches[0]

    # Process each CSV
    for csv_name, target_dir_name in csvs.items():
        csv_path = os.path.join(csv_dir, csv_name)
        dest_dir = os.path.join("E:/code/ADNI", target_dir_name)
        os.makedirs(dest_dir, exist_ok=True)
        print(f"Processing {csv_path} -> dest_dir: {dest_dir}")

        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.reader(f)
            header = next(reader)
            
            total = 0
            linked = 0
            missing = 0
            
            for row in reader:
                total += 1
                csv_fn = row[1] # e.g. ADNI_100_S_0015_...nii
                best_match = get_best_match(csv_fn)
                
                if best_match:
                    src_file = os.path.join(mri_dir, best_match)
                    dst_file = os.path.join(dest_dir, csv_fn)
                    
                    if not os.path.exists(dst_file):
                        try:
                            os.link(src_file, dst_file)
                            linked += 1
                        except Exception as e:
                            print(f"Failed to link {src_file} -> {dst_file}: {e}")
                    else:
                        linked += 1
                else:
                    missing += 1

            print(f"Finished {csv_name}: Total={total}, Linked/Matched={linked}, Missing={missing}")

if __name__ == "__main__":
    main()
