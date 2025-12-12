import os
import time
import logging
import httpx
from qdrant_client import QdrantClient

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def backup_collection_bg(collection_name, backup_dir="backups", qdrant_url="http://localhost:6333", api_key=None):
    client = QdrantClient(url=qdrant_url, api_key=api_key)
    headers = {}
    if api_key:
        headers["api-key"] = api_key

    # 1. Trigger snapshot background
    logger.info(f"Requesting snapshot (background) for collection '{collection_name}' …")
    resp = httpx.post(
        f"{qdrant_url}/collections/{collection_name}/snapshots?wait=false",
        headers=headers,
        timeout=30,
    )
    resp.raise_for_status()
    logger.info("Snapshot creation triggered (server working in background).")

    # 2. Poll until snapshot appears
    snapshot_name = None
    while True:
        time.sleep(5) 
        resp = httpx.get(
            f"{qdrant_url}/collections/{collection_name}/snapshots",
            headers=headers,
            timeout=30,
        )
        resp.raise_for_status()
        snap_list = resp.json().get("result", [])
        
        if snap_list:
            # Sort explicitly by creation_time to get the newest
            # snap_list example item: {'name': '...', 'creation_time': '2023-10-10T...'}
            # We assume creation_time is ISO string or sortable. Qdrant returns ISO strings usually.
            # actually creation_time might be missing or different, but 'name' usually contains timestamp or increment.
            # Let's rely on creation_time if available, else name.
            # Robust sort:
            snap_list.sort(key=lambda x: x.get('creation_time', x.get('name', '') ))
            
            snapshot_name = snap_list[-1]["name"]
            logger.info(f"Found latest snapshot: {snapshot_name}")
            break
        else:
            logger.info("Snapshot not ready yet, waiting another 5s...")

    # 3. Download or Copy
    os.makedirs(backup_dir, exist_ok=True)
    target_path = os.path.join(backup_dir, snapshot_name)
    
    # Try to find local qdrant_snapshots folder
    # This script is at backend/krissbert_custom/usage/backup_qdrant.py
    # Project root is 3 levels up: ../../../
    # qdrant_snapshots is at project root.
    
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../../"))
    local_snapshot_path = os.path.join(project_root, "qdrant_snapshots", collection_name, snapshot_name)
    
    copied_locally = False
    if os.path.exists(local_snapshot_path):
        try:
            import shutil
            logger.info(f"Local snapshot found at {local_snapshot_path}. Copying directly...")
            shutil.copy2(local_snapshot_path, target_path)
            logger.info("Snapshot copied successfully (Fast Local Copy).")
            copied_locally = True
        except PermissionError:
            logger.warning(f"Permission denied when reading {local_snapshot_path}. Docker volumes often require root.")
            logger.info("Attempting to copy via 'docker compose cp' (requires user in docker group)...")
            try:
                import subprocess
                # Check for docker compose or docker-compose
                # We assume we are in the project root or similar, but let's be safe.
                # The script runs from backend/krissbert_custom/usage/. 
                # docker-compose.yml is in project root.
                
                # Command: docker compose cp qdrant:/qdrant/snapshots/<collection_name>/<filename> <target>
                # Service name 'qdrant' from docker-compose.yml
                
                cmd = ["docker", "compose", "cp", f"qdrant:/qdrant/snapshots/{collection_name}/{snapshot_name}", target_path]
                
                # We need to run this from the project root where docker-compose.yml is located
                cwd = project_root
                
                logger.info(f"Running: {' '.join(cmd)}")
                ret = subprocess.call(cmd, cwd=cwd)
                
                if ret == 0:
                    logger.info("Snapshot copied successfully (via docker compose cp).")
                    copied_locally = True
                else:
                    logger.warning("Docker cp failed. Checking for sudo availability...")
                    
                    # Improve sudo interactivity using os.system which connects stdin/stdout directly
                    print("\n" + "!"*60)
                    print("PERMISSION DENIED. Trying to copy with SUDO.")
                    print("Please enter your sudo password if prompted below:")
                    print("!"*60 + "\n")
                    
                    # os.system is better for interactive password prompts than subprocess.call in some cases
                    # Use 'sudo -v' first to prime the credential
                    os.system("sudo -v") 
                    
                    cmd_cp = f"sudo cp '{local_snapshot_path}' '{target_path}'"
                    ret_cp = os.system(cmd_cp)
                    
                    if ret_cp == 0:
                        logger.info("Snapshot copied via sudo. Fixing file ownership...")
                        # Fix ownership
                        uid = os.getuid()
                        gid = os.getgid()
                        cmd_chown = f"sudo chown {uid}:{gid} '{target_path}'"
                        os.system(cmd_chown)
                        
                        logger.info("Snapshot copied successfully (via sudo).")
                        copied_locally = True
                    else:
                        logger.warning("Sudo copy failed or was cancelled.")

            except Exception as e:
                logger.warning(f"Copy attempt failed: {e}")
        except Exception as e:
            logger.warning(f"Failed to copy local snapshot: {e}. Falling back to HTTP download.")

    if not copied_locally:
        logger.info(f"Local snapshot not found or inaccessible (and docker cp failed). Falling back to HTTP download...")
        logger.info(f"Downloading snapshot to {target_path} ...")
        with httpx.stream("GET", f"{qdrant_url}/collections/{collection_name}/snapshots/{snapshot_name}",
                           headers=headers, timeout=3600) as response:
            try:
                response.raise_for_status()
                with open(target_path, "wb") as f:
                    for chunk in response.iter_bytes():
                        f.write(chunk)
                logger.info("Snapshot downloaded successfully.")
            except Exception as e:
                logger.error(f"Download failed: {e}")
                if os.path.exists(target_path):
                    os.remove(target_path)
                raise

if __name__ == "__main__":
    backup_collection_bg("kg_lv2_nodes")
