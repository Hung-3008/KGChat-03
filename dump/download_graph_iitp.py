#!/usr/bin/env python3
"""
Download graph.zip from ServerIITP
"""
import subprocess
import os

# Server configuration
SERVER = {
    "name": "ServerIITP",
    "host": "168.131.30.42",
    "port": 8201,
    "user": "root",
    "password": "root@iitp"
}

# Remote file path
REMOTE_FILE = "/workspace/sdb1/KGChat-03/output/graph.zip"

# Local download directory
LOCAL_DIR = "./output"

def download_file():
    """Download graph.zip from ServerIITP using scp with sshpass."""
    
    # Ensure local directory exists
    os.makedirs(LOCAL_DIR, exist_ok=True)
    
    local_path = os.path.join(LOCAL_DIR, "graph.zip")
    
    print(f"Downloading {REMOTE_FILE} from {SERVER['name']}...")
    print(f"  -> Host: {SERVER['host']}:{SERVER['port']}")
    print(f"  -> Destination: {local_path}")
    
    cmd = [
        "sshpass", "-p", SERVER["password"],
        "scp", "-P", str(SERVER["port"]),
        "-o", "StrictHostKeyChecking=no",
        f"{SERVER['user']}@{SERVER['host']}:{REMOTE_FILE}",
        local_path
    ]
    
    try:
        result = subprocess.run(cmd, check=True)
        print(f"  -> Success: Downloaded graph.zip")
        return True
    except subprocess.CalledProcessError as e:
        print(f"  -> Error: Failed to download file. Exit code: {e.returncode}")
        return False
    except Exception as e:
        print(f"  -> Error: {e}")
        return False

if __name__ == "__main__":
    download_file()
