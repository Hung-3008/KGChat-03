import os
import time
import subprocess
import datetime
import shutil

# --- CONFIGURATION ---
# Danh sách các server
SERVERS = [
    {"name": "may3", "host": "n2.ckey.vn", "port": 1873, "user": "root"},
    {"name": "may2", "host": "n2.ckey.vn", "port": 1850, "user": "root"},
    {"name": "may4", "host": "n2.ckey.vn", "port": 1920, "user": "root"},
]

# Password chung cho các server
SERVER_PASSWORD = "12345678"

# Thư mục gốc để lưu file tải về trên máy local
# Cấu trúc sẽ là: ./output/may3, ./output/may2, ...
LOCAL_BASE_DIR = "./output"

# Đường dẫn nguồn trên server
REMOTE_DIR = "/home/KGChat-03/output/backups/"

# Thời gian chờ giữa các lần sync (giây) - 1 tiếng
CHECK_INTERVAL = 3600 

def ensure_sshpass():
    """Kiểm tra xem sshpass có được cài đặt hay không."""
    if shutil.which("sshpass") is None:
        print("Error: 'sshpass' is not installed. Please install it (e.g., sudo apt install sshpass) to support password authentication.")
        return False
    return True

def get_latest_remote_file(server):
    """
    Tìm file .tar.gz mới nhất trong thư mục REMOTE_DIR trên server.
    Trả về đường dẫn đầy đủ của file hoặc None nếu không tìm thấy.
    """
    host = server["host"]
    port = server["port"]
    user = server["user"]
    
    # Lệnh tìm file mới nhất: ls -1t /path/*.tar.gz | head -n 1
    # 2>/dev/null để ẩn lỗi nếu không có file nào
    find_cmd = f"ls -1t {REMOTE_DIR}*.tar.gz 2>/dev/null | head -n 1"
    
    ssh_opts = f"-p {port} -o StrictHostKeyChecking=no"
    
    cmd = [
        "sshpass", "-p", SERVER_PASSWORD,
        "ssh", "-p", str(port),
        "-o", "StrictHostKeyChecking=no",
        f"{user}@{host}",
        find_cmd
    ]
    
    try:
        # Run command and capture output
        result = subprocess.run(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE,
            universal_newlines=True,
            check=False # Don't raise exception on non-zero return code
        )
        
        output = result.stdout.strip()
        
        if result.returncode == 0 and output:
            return output
        else:
            return None
            
    except Exception as e:
        print(f"Error checking files on {server['name']}: {e}")
        return None

def sync_from_servers():
    """
    Download file backup mới nhất từ các Server về Local.
    """
    if not ensure_sshpass():
        return

    timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    print(f"\n[{timestamp}] Checking for latest backups...")

    if not os.path.exists(LOCAL_BASE_DIR):
        os.makedirs(LOCAL_BASE_DIR, exist_ok=True)

    for server in SERVERS:
        name = server["name"]
        host = server["host"]
        port = server["port"]
        user = server["user"]
        
        # 1. Tìm file mới nhất
        latest_file = get_latest_remote_file(server)
        
        if not latest_file:
            print(f"[{timestamp}] {name}: No .tar.gz backup found in {REMOTE_DIR}.")
            continue
            
        filename = os.path.basename(latest_file)
        print(f"[{timestamp}] {name}: Found latest backup: {filename}")

        # 2. Chuẩn bị thư mục local
        local_server_dir = os.path.join(LOCAL_BASE_DIR, name)
        if not os.path.exists(local_server_dir):
            os.makedirs(local_server_dir, exist_ok=True)

        # 3. Download file đó (rsync)
        # Kiểm tra xem file đã tồn tại ở local chưa (rsync cũng tự check size/timestamp, nhưng ta log cho rõ)
        local_file_path = os.path.join(local_server_dir, filename)
        
        print(f"  -> Downloading to {local_server_dir}...")

        ssh_cmd = f"ssh -p {port} -o StrictHostKeyChecking=no"
        
        cmd = [
            "sshpass", "-p", SERVER_PASSWORD,
            "rsync", "-avz", "--progress",
            "-e", ssh_cmd,
            f"{user}@{host}:{latest_file}",
            local_server_dir
        ]

        try:
            # Run command directly, letting stdout/stderr go to console
            process = subprocess.Popen(
                cmd, 
                universal_newlines=True
            )
            
            process.wait()
            
            if process.returncode == 0:
                print(f"  -> Success: Downloaded {filename} from {name}.")
            else:
                print(f"  -> Failed: Could not download {filename} from {name}.")
                
        except Exception as e:
            print(f"  -> Error executing sync for {name}: {e}")

    print(f"[{timestamp}] All sync tasks completed.")

def main():
    print("--- Starting Local Latest Backup Sync Service ---")
    print(f"Local Base: {LOCAL_BASE_DIR}")
    print(f"Remote Source: {REMOTE_DIR}")
    print(f"Servers: {[s['name'] for s in SERVERS]}")
    print(f"Interval: {CHECK_INTERVAL} seconds")
    print("------------------------------------------")
    
    # Check sshpass once at start
    if ensure_sshpass():
        while True:
            sync_from_servers()
            print(f"\nSleeping for {CHECK_INTERVAL/60} minutes...")
            time.sleep(CHECK_INTERVAL)
    else:
        print("Service aborted due to missing dependencies.")

if __name__ == "__main__":
    main()
