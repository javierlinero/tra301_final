import os
import random
import shutil

def select_files(src_dir, count=100):
    all_files = [f for f in os.listdir(src_dir)
                 if os.path.isfile(os.path.join(src_dir, f))]
    if len(all_files) < count:
        raise ValueError(f"Found only {len(all_files)} files in {src_dir}, but need {count}.")
    return random.sample(all_files, count)

def main():
    arxiv_path = os.path.join(os.path.dirname(__file__), "data", "raw")
    chosen = select_files(arxiv_path, 100)
    dest_path = os.path.join(os.path.dirname(__file__), "subset")
    os.makedirs(dest_path, exist_ok=True)

    print("Selected files:")
    for fname in chosen:
        print(f"{fname}")

    if dest_path:
        for fname in chosen:
            shutil.copy2(os.path.join(arxiv_path, fname),
                         os.path.join(dest_path, fname))
        print(f"\nCopied {len(chosen)} files to {dest_path}")

if __name__ == "__main__":
    main()