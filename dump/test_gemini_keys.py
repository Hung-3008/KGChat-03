
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dotenv import dotenv_values
from google import genai
from google.genai import types

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

def test_key(key_name, api_key):
    """
    Tests a single Gemini API Key.
    Returns (key_name, is_valid, message)
    """
    if not api_key:
        return key_name, False, "Empty Key"

    try:
        client = genai.Client(api_key=api_key)
        # Use simple model for testing
        response = client.models.generate_content(
            model='models/gemini-2.5-flash', 
            contents='Hello',
            config=types.GenerateContentConfig(
                temperature=0.7,
                top_p=0.9
            ) 
        )
        
        text = response.text
        if text:
            return key_name, True, f"Success - Response: {text.strip()}"
        else:
            # Check if there are candidates but no text (blocked?)
            if hasattr(response, 'candidates') and response.candidates:
                first_cand = response.candidates[0]
                if hasattr(first_cand, 'finish_reason'):
                    return key_name, False, f"Blocked/Empty. Reason: {first_cand.finish_reason}"
            return key_name, False, "Empty Response Text"
            
    except Exception as e:
        error_msg = str(e)
        if "403" in error_msg: # INVALID_ARGUMENT or PERMISSION_DENIED
             return key_name, False, "Invalid Key (403)"
        elif "429" in error_msg:
             return key_name, True, "Valid but Rate Limited (429)"
        else:
            # Clean up error message
            msg = error_msg.split('}')[-1].strip() if '}' in error_msg else error_msg
            return key_name, False, f"Failed: {msg}"

def main():
    print("Loading keys from .env...")
    env_vars = dotenv_values(".env")
    
    # Filter for Gemini Keys
    gemini_keys = {k: v for k, v in env_vars.items() if k.startswith("GEMINI_API_KEY") and v}
    
    if not gemini_keys:
        print("No GEMINI_API_KEY* found in .env")
        return

    print(f"Found {len(gemini_keys)} keys. Testing in parallel...")
    print("-" * 60)
    
    results = []
    
    # Run tests in parallel
    with ThreadPoolExecutor(max_workers=len(gemini_keys)) as executor:
        future_to_key = {
            executor.submit(test_key, k, v): k 
            for k, v in gemini_keys.items()
        }
        
        for future in as_completed(future_to_key):
            results.append(future.result())

    # Sort results by key name for readability (e.g., KEY_1, KEY_2 is tricky with string sort but better than random)
    try:
        results.sort(key=lambda x: int(x[0].split('_')[-1]) if x[0].split('_')[-1].isdigit() else x[0])
    except:
        results.sort(key=lambda x: x[0])
    
    valid_count = 0
    print(f"{'Key Name':<25} | {'Status':<15} | {'Message'}")
    print("-" * 80)
    for name, is_valid, msg in results:
        status = "✅ VALID" if is_valid else "❌ INVALID"
        print(f"{name:<25} | {status:<15} | {msg}")
        if is_valid:
            valid_count += 1
            
    print("-" * 80)
    print(f"Summary: {valid_count}/{len(gemini_keys)} keys are potentially valid.")

if __name__ == "__main__":
    main()
