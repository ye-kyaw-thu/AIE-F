import os
import sys
import subprocess

def run_command(cmd):
    print(f"Executing: {cmd}")
    result = subprocess.run(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    if result.returncode != 0:
        print(f"Error executing command: {result.stderr}")
        sys.exit(1)
    return True

def find_tools():
    local_bin = os.path.abspath("kenlm_src/build/bin")
    lmplz = os.path.join(local_bin, "lmplz")
    build_binary = os.path.join(local_bin, "build_binary")
    
    if os.path.exists(lmplz) and os.path.exists(build_binary):
        return lmplz, build_binary
        
    workspace_bin = "/workspace/assignment5/kenlm_src/build/bin"
    lmplz = os.path.join(workspace_bin, "lmplz")
    build_binary = os.path.join(workspace_bin, "build_binary")
    
    if os.path.exists(lmplz) and os.path.exists(build_binary):
        return lmplz, build_binary
        
    print("Error: KenLM tools not found! Please run KenLM compilation first.")
    sys.exit(1)

def main():
    print("=== Step 3: Training Base Language Model ===")
    lmplz, build_binary = find_tools()
    
    # Create models directory
    os.makedirs("data/models", exist_ok=True)
    
    # Paths
    train_corpus = "data/tokenized/train_general.txt"
    arpa_path = "data/models/general.arpa"
    binary_path = "data/models/general.binary"
    
    if not os.path.exists(train_corpus):
        print(f"Error: Training corpus not found at {train_corpus}")
        sys.exit(1)
        
    # Train 5-gram language model
    print("Training 5-gram Language Model with lmplz...")
    cmd_train = f"{lmplz} -o 5 -S 2G -T /tmp --discount_fallback < {train_corpus} > {arpa_path}"
    run_command(cmd_train)
    
    # Convert to binary
    print("Converting ARPA model to binary format...")
    cmd_binary = f"{build_binary} {arpa_path} {binary_path}"
    run_command(cmd_binary)
    
    print("Base Language Model training and binary conversion complete!")

if __name__ == "__main__":
    main()
