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
        time.sleep(30)  # đợi 30s mỗi lần check — chỉnh tuỳ nhu cầu / tài nguyên
        resp = httpx.get(
            f"{qdrant_url}/collections/{collection_name}/snapshots",
            headers=headers,
            timeout=30,
        )
        resp.raise_for_status()
        snap_list = resp.json().get("result", [])
        if snap_list:
            # giả sử snapshot mới nhất là snapshot cuối
            snapshot_name = snap_list[-1]["name"]
            logger.info(f"Found snapshot: {snapshot_name}")
            break
        else:
            logger.info("Snapshot not ready yet, waiting another 30s…")

    # 3. Download snapshot với timeout dài
    os.makedirs(backup_dir, exist_ok=True)
    download_path = os.path.join(backup_dir, snapshot_name)
    logger.info(f"Downloading snapshot to {download_path} …")
    with httpx.stream("GET", f"{qdrant_url}/collections/{collection_name}/snapshots/{snapshot_name}",
                       headers=headers, timeout=3600) as response:
        response.raise_for_status()
        with open(download_path, "wb") as f:
            for chunk in response.iter_bytes():
                f.write(chunk)
    logger.info("Snapshot downloaded successfully.")

if __name__ == "__main__":
    backup_collection_bg("kg_lv2_nodes")
