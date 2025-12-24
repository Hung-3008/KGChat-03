import os
import time
import json
import tarfile
import datetime

# Configuration
PROCESSED_FILES_PATH = 'output/processed_files.txt'
GRAPH_DIR = 'output/graph'
BACKUP_DIR = 'output/backups'
STATE_FILE = 'output/backup_state.json'
CHECK_INTERVAL = 3600  # 1 hour

def ensure_directories():
    if not os.path.exists(BACKUP_DIR):
        os.makedirs(BACKUP_DIR)

def load_processed_files():
    if not os.path.exists(PROCESSED_FILES_PATH):
        return []
    with open(PROCESSED_FILES_PATH, 'r') as f:
        # Read lines and strip whitespace
        return [line.strip() for line in f if line.strip()]

def load_state():
    if not os.path.exists(STATE_FILE):
        return []
    try:
        with open(STATE_FILE, 'r') as f:
            return json.load(f)
    except json.JSONDecodeError:
        return []

def save_state(processed_list):
    with open(STATE_FILE, 'w') as f:
        json.dump(processed_list, f)

def create_incremental_backup(new_items):
    timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_filename = f"graph_backup_{timestamp}.tar.gz"
    backup_path = os.path.join(BACKUP_DIR, backup_filename)
    
    print(f"[{timestamp}] Creating backup for {len(new_items)} new items...")
    
    with tarfile.open(backup_path, "w:gz") as tar:
        for item in new_items:
            # item is filename like "PMC_result.json", we need "PMC_result"
            folder_name = item.replace('.json', '')
            folder_path = os.path.join(GRAPH_DIR, folder_name)
            
            if os.path.exists(folder_path):
                tar.add(folder_path, arcname=folder_name)
            else:
                print(f"Warning: Folder not found {folder_path}")

    print(f"Backup created: {backup_path}")
    return backup_path

def run_backup_cycle():
    ensure_directories()
    
    # 1. Load current processed files
    current_files = load_processed_files()
    if not current_files:
        print("No processed files found.")
        return

    # 2. Load previous state (files already backed up)
    backed_up_files = load_state()
    
    # 3. Find diff
    # Convert to set for O(1) lookups, but keep order if possible or list is fine
    new_files = [f for f in current_files if f not in backed_up_files]
    
    if new_files:
        create_incremental_backup(new_files)
        # 4. Update state (add new files to backed_up_files)
        # We append to keep history or just overwrite with current_files?
        # If we overwrite with `current_files`, we assume `processed_files.txt` only grows.
        # Safest is to union them.
        updated_state = list(set(backed_up_files + new_files))
        save_state(updated_state)
    else:
        print(f"[{datetime.datetime.now()}] No new files to backup.")

def main():
    print("Starting Auto Backup Service...")
    while True:
        try:
            run_backup_cycle()
        except Exception as e:
            print(f"Error during backup cycle: {e}")
        
        print(f"Sleeping for {CHECK_INTERVAL} seconds...")
        time.sleep(CHECK_INTERVAL)

if __name__ == "__main__":
    main()
