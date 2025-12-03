import os
import logging
import requests
from requests_toolbelt import MultipartEncoder, MultipartEncoderMonitor
from qdrant_client import QdrantClient
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def restore_snapshot(collection_name, backup_dir, qdrant_url="http://localhost:6333", api_key=None):
    # Find snapshot file
    if not os.path.exists(backup_dir):
        logger.error(f"Backup directory {backup_dir} does not exist.")
        return

    snapshots = [f for f in os.listdir(backup_dir) if f.endswith(".snapshot")]
    if not snapshots:
        logger.error("No snapshot files found in backup directory.")
        return
    
    # Sort by modification time to get the latest
    snapshots.sort(key=lambda x: os.path.getmtime(os.path.join(backup_dir, x)), reverse=True)
    snapshot_file = snapshots[0]
    snapshot_path = os.path.join(backup_dir, snapshot_file)
    
    logger.info(f"Found snapshot: {snapshot_path}")
    total_size = os.path.getsize(snapshot_path)
    logger.info(f"Snapshot size: {total_size / (1024**3):.2f} GB")
    
    # Check if collection exists
    try:
        client = QdrantClient(url=qdrant_url, api_key=api_key)
        collections = client.get_collections().collections
        exists = any(c.name == collection_name for c in collections)
        
        if exists:
            logger.warning(f"Collection '{collection_name}' already exists. It will be overwritten by the snapshot.")
    except Exception as e:
        logger.warning(f"Could not check if collection exists: {e}")
    
    # Upload and restore
    url = f"{qdrant_url}/collections/{collection_name}/snapshots/upload"
    headers = {}
    if api_key:
        headers["api-key"] = api_key
        
    logger.info(f"Starting upload and restore to collection '{collection_name}'...")
    
    try:
        # Create MultipartEncoder
        # We need to open the file in binary mode
        with open(snapshot_path, 'rb') as f:
            encoder = MultipartEncoder(
                fields={'snapshot': (snapshot_file, f, 'application/octet-stream')}
            )
            
            # Create progress bar
            pbar = tqdm(total=total_size, unit='B', unit_scale=True, desc=f"Uploading {snapshot_file}")
            last_log_pos = 0
            
            # Callback for monitor
            def callback(monitor):
                pbar.update(monitor.bytes_read - pbar.n)
                
                # Log every ~100MB
                nonlocal last_log_pos
                if monitor.bytes_read - last_log_pos > 100 * 1024 * 1024:
                    logger.info(f"Uploaded {monitor.bytes_read / (1024**3):.2f} GB / {total_size / (1024**3):.2f} GB")
                    last_log_pos = monitor.bytes_read
            
            monitor = MultipartEncoderMonitor(encoder, callback)
            
            # Set content-type header
            headers['Content-Type'] = monitor.content_type
            
            # Upload
            response = requests.post(url, headers=headers, data=monitor, timeout=None)
            
            pbar.close()
            
            response.raise_for_status()
            logger.info(f"Successfully restored snapshot to collection '{collection_name}'.")
            logger.info(f"Response: {response.json()}")
        
    except requests.exceptions.HTTPError as e:
        logger.error(f"Failed to restore snapshot: {e.response.text}")
    except Exception as e:
        logger.error(f"An error occurred during restore: {e}")

if __name__ == "__main__":
    # Determine paths
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, "../../../"))
    backup_dir = os.path.join(project_root, "backups")
    
    # Collection name
    collection_name = "kg_lv2_nodes"
    
    restore_snapshot(collection_name, backup_dir)
