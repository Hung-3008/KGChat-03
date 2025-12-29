import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from backend.llm.llm_factory import LLMFactory


def load_config(path: Path) -> dict:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def run_request(client, prompt: str, idx: int, schema: dict | None):
    start = time.time()
    try:
        resp = client.generate(prompt=prompt, format=schema) if schema else client.generate(prompt=prompt)
        preview = str(resp)
    except Exception as e:
        return idx, time.time() - start, f"ERROR: {e}"
    return idx, time.time() - start, preview


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="backend/configs/vllm_configs.yml")
    parser.add_argument("--requests", type=int, default=3)
    parser.add_argument("--prompt", default="Return a greeting and its length as JSON")
    parser.add_argument("--structured", action="store_true")
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top_p", type=float)
    args = parser.parse_args()

    cfg_path = Path(args.config)
    cfg = load_config(cfg_path)
    llm_cfg = cfg.get("LLM", {}) if isinstance(cfg, dict) else {}

    if args.temperature is not None:
        llm_cfg["temperature"] = args.temperature
    if args.top_p is not None:
        llm_cfg["top_p"] = args.top_p

    client = LLMFactory.create_client(llm_cfg)

    schema = None
    prompt = args.prompt
    if args.structured:
        schema = {
            "type": "object",
            "properties": {
                "greeting": {"type": "string"},
                "length": {"type": "integer"}
            },
            "required": ["greeting", "length"]
        }
        prompt = "Return JSON with 'greeting' string and 'length' integer giving its character count."

    print(f"Config: model={llm_cfg.get('model')} base_url={llm_cfg.get('base_url')} structured={bool(schema)}")
    print(f"Running {args.requests} parallel requests...\n")

    overall_start = time.time()
    with ThreadPoolExecutor(max_workers=args.requests) as executor:
        futures = [executor.submit(run_request, client, prompt, i, schema) for i in range(args.requests)]
        for future in as_completed(futures):
            idx, duration, preview = future.result()
            truncated = preview.replace("\n", " ")
            if len(truncated) > 160:
                truncated = truncated[:160] + "..."
            print(f"req#{idx}: {duration:.2f}s | {truncated}")

    print(f"\nTotal wall time: {time.time() - overall_start:.2f}s")


if __name__ == "__main__":
    main()
