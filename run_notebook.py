import os
import sys
import json
import traceback

def run_notebook(nb_path):
    print(f"==================== RUNNING NOTEBOOK: {nb_path} ====================")
    if not os.path.exists(nb_path):
        print(f"Error: File {nb_path} does not exist!")
        return False

    # Change working directory to the notebook's directory
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

    # Create globals dictionary for execution
    globals_dict = {
        '__file__': os.path.abspath(nb_path),
        '__name__': '__main__'
    }

    cells = nb.get('cells', [])
    code_cells = [c for c in cells if c.get('cell_type') == 'code']
    print(f"Found {len(code_cells)} code cells.")

    for i, cell in enumerate(code_cells):
        source = cell.get('source', [])
        # Skip empty cells
        if not source:
            continue
        
        # Clean code: remove line magics and shell commands
        clean_lines = []
        for line in source:
            stripped = line.strip()
            if stripped.startswith('%') or stripped.startswith('!'):
                # comment it out
                clean_lines.append(f"# {line}")
            else:
                clean_lines.append(line)
        
        code_str = "".join(clean_lines)
        
        # Print a snippet of the cell code for context
        first_line = clean_lines[0].strip() if clean_lines else ""
        print(f"\n--- Running Cell {i+1}/{len(code_cells)} (Starts with: {first_line[:50]}...) ---")
        
        try:
            # Execute the cell code
            exec(code_str, globals_dict)
        except Exception as e:
            print(f"❌ Exception in cell {i+1}:")
            traceback.print_exc()
            # Clean up CWD
            os.chdir(original_cwd)
            return False

    print(f"==================== NOTEBOOK COMPLETED SUCCESSFULLY: {nb_path} ====================\n")
    os.chdir(original_cwd)
    if nb_dir in sys.path:
        sys.path.remove(nb_dir)
    return True

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python run_notebook.py <path_to_ipynb>")
        sys.exit(1)
    
    success = run_notebook(sys.argv[1])
    if not success:
        sys.exit(1)
