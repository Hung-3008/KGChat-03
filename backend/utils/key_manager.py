
import threading
import time
import logging
from typing import List, Optional, Set, Dict

logger = logging.getLogger("key_manager")

class KeyManager:
    def __init__(self, keys: List[str], cooldown_seconds: int = 60):
        self.all_keys = list(set(keys)) # Deduplicate
        self.active_keys: Set[str] = set()
        self.cooldowns: Dict[str, float] = {} # key -> cooldown_expiry_timestamp
        self.cooldown_seconds = cooldown_seconds
        self.lock = threading.Lock()
        
        logger.info(f"Initialized KeyManager with {len(self.all_keys)} keys.")

    def get_key(self, current_key: Optional[str] = None) -> Optional[str]:
        """
        Returns a new key.
        If current_key is provided, marks it as cooling down.
        The returned key is guaranteed to be NOT in active_keys and NOT in cooldown.
        Blocks/Waits if no keys are available? For now, returns None or raises if all exhausted.
        """
        with self.lock:
            current_time = time.time()
            
            # 1. Handle current_key (failure case)
            if current_key:
                if current_key in self.active_keys:
                    self.active_keys.remove(current_key)
                
                # Set cooldown
                expiry = current_time + self.cooldown_seconds
                self.cooldowns[current_key] = expiry
                masked = current_key[:5] + "..." + current_key[-3:]
                logger.warning(f"Key {masked} marked for cooldown until {expiry:.0f} (failed/429).")

            # 2. Cleanup expired cooldowns
            expired = [k for k, t in self.cooldowns.items() if t < current_time]
            for k in expired:
                del self.cooldowns[k]
                
            # 3. Find available key
            # Available = in all_keys AND not in active_keys AND not in cooldowns
            available_candidates = [
                k for k in self.all_keys 
                if k not in self.active_keys and k not in self.cooldowns
            ]
            
            if not available_candidates:
                # If truly no keys, check if we can wait? 
                # Or just return None and let caller sleep?
                # For simplicity, returning None implies "System Overloaded"
                
                # Logging status
                logger.error(f"No available keys! Active: {len(self.active_keys)}, Cooldown: {len(self.cooldowns)}")
                return None
                
            # Pick one (first one for now, or random?)
            # Deterministic/First is fine for creating stable fill
            new_key = available_candidates[0]
            self.active_keys.add(new_key)
            masked_new = new_key[:5] + "..." + new_key[-3:]
            logger.info(f"Assigned new key {masked_new}. Active keys: {len(self.active_keys)}")
            
            return new_key

    def release_key(self, key: str):
        """
        Releases a key back to the pool (e.g. when worker finishes a file successfully without error).
        Alternatively, active_keys helps prevent *concurrrent* usage. 
        If we want to strictly limit 1 thread per key (which we do for rate limiting), 
        we should keep it in active_keys as long as the worker uses it.
        """
        with self.lock:
            if key in self.active_keys:
                self.active_keys.remove(key)
                masked = key[:5] + "..." + key[-3:]
                logger.debug(f"Released key {masked}.")
