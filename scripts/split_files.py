import os
import shutil
import sys

# Define directories relative to the current working directory or absolute
SOURCE_DIR = os.path.join(os.getcwd(), 'data', 'PMC_Part1')
DEST_DIR = os.path.join(os.getcwd(), 'split_from_pmc_part1')
START_INDEX = 10000

def main():
    # Ensure source exists
    if not os.path.exists(SOURCE_DIR):
        print(f"Error: Source directory {SOURCE_DIR} does not exist.")
        sys.exit(1)

    # Create destination if not exists
    if not os.path.exists(DEST_DIR):
        os.makedirs(DEST_DIR)
        print(f"Created directory: {DEST_DIR}")
    else:
        print(f"Destination directory already exists: {DEST_DIR}")

    # Get all files and sort them to ensure deterministic order
    print("Listing and sorting files...")
    all_files = [f for f in os.listdir(SOURCE_DIR) if os.path.isfile(os.path.join(SOURCE_DIR, f))]
    all_files.sort()

    total_files = len(all_files)
    print(f"Total files found: {total_files}")

    if total_files <= START_INDEX:
        print(f"Error: Not enough files to skip {START_INDEX}. Total only {total_files}.")
        sys.exit(1)

    # Slice the list
    files_to_copy = all_files[START_INDEX:]
    count_to_copy = len(files_to_copy)

    print(f"Copying {count_to_copy} files (from index {START_INDEX} onwards)...")

    # Copy files
    copied_count = 0
    for filename in files_to_copy:
        src = os.path.join(SOURCE_DIR, filename)
        dst = os.path.join(DEST_DIR, filename)
        
        try:
            shutil.copy2(src, dst)
            copied_count += 1
            if copied_count % 1000 == 0:
                print(f"Copied {copied_count}/{count_to_copy} files...", end='\r')
        except Exception as e:
            print(f"\nError copying {filename}: {e}")

    print(f"\nSuccessfully copied {copied_count} files to {DEST_DIR}")

if __name__ == "__main__":
    main()
