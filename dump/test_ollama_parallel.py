import argparse
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

# Ensure project root is on sys.path when running from scripts/
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from backend.llm.llm_factory import LLMFactory


def run_request(client, prompt: str, idx: int) -> tuple[int, float, str]:
    start = time.time()
    try:
        resp = client.generate(prompt=prompt)
    except Exception as e:
        return idx, time.time() - start, f"ERROR: {e}"
    return idx, time.time() - start, str(resp)[:120].replace("\n", " ")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="llama3.2:3b")
    parser.add_argument("--requests", type=int, default=4)
    parser.add_argument("--prompt", default="Say hello concisely")
    parser.add_argument("--temperature", type=float, default=0.85)
    parser.add_argument("--top_p", type=float, default=0.9)
    args = parser.parse_args()

    config = {
        "client": "ollama",
        "model": args.model,
        "temperature": args.temperature,
        "top_p": args.top_p,
    }
    client = LLMFactory.create_client(config)

    print(f"Running {args.requests} parallel requests against model={args.model}")
    overall_start = time.time()

    with ThreadPoolExecutor(max_workers=args.requests) as executor:
        futures = [executor.submit(run_request, client, args.prompt, i) for i in range(args.requests)]
        for future in as_completed(futures):
            idx, duration, preview = future.result()
            print(f"req#{idx}: {duration:.2f}s | {preview}")

    print(f"Total wall time: {time.time() - overall_start:.2f}s")


if __name__ == "__main__":
    main()
