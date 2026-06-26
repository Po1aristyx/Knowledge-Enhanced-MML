import os
import subprocess

datasets = ['ADNI', 'AIBL', 'NACC', 'PPMI']
methods = ['ConvE', 'ConvR']

os.chdir(r"d:\Polaris\Documents\work4\DementiaHKG\DementiaHKG")

for method in methods:
    for ds in datasets:
        script = f"{ds}_{method}.py"
        log_file = f"log_{ds}_{method}.txt"
        
        # Check if completed
        completed = False
        if os.path.exists(log_file):
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                if "=== Final Validation Metrics ===" in f.read():
                    completed = True
                    
        if completed:
            print(f"[{ds} {method}] Already completed, skipping.")
            continue
            
        print(f"[{ds} {method}] Starting training...")
        
        # Run and append to log file so we can resume and see history
        with open(log_file, 'a', encoding='utf-8') as f:
            f.write(f"\n\n--- Starting Run for {ds} {method} ---\n")
            
        cmd = rf"D:\anaconda3\envs\audiokeshe\python.exe {script}"
        with open(log_file, 'a', encoding='utf-8') as f:
            process = subprocess.Popen(cmd, stdout=f, stderr=subprocess.STDOUT, shell=True)
            process.wait()
            
        print(f"[{ds} {method}] Finished training.")

print("All downstream tasks finished.")
