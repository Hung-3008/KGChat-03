import json
import glob
import os
import tiktoken
import statistics
from collections import Counter

def analyze_chunks(data_dir):
    print(f"Analyzing files in {data_dir}...")
    files = glob.glob(os.path.join(data_dir, "*.json"))
    
    if not files:
        print("No JSON files found!")
        return

    enc = tiktoken.get_encoding("cl100k_base")
    
    file_section_counts = []
    section_token_counts = []
    file_total_token_counts = []
    
    # Track distribution
    token_dist = Counter()
    
    for fpath in files:
        try:
            with open(fpath, 'r', encoding='utf-8') as f:
                data = json.load(f)
                
            sections = data.get("content_sections", [])
            file_section_counts.append(len(sections))
            
            file_tokens = 0
            for s in sections:
                if isinstance(s, dict) and "content" in s:
                    content = s["content"]
                elif isinstance(s, str):
                    content = s 
                else:
                    continue
                    
                if not content or not isinstance(content, str):
                    continue
                    
                tokens = len(enc.encode(content))
                section_token_counts.append(tokens)
                file_tokens += tokens
                
                bucket = (tokens // 100) * 100
                token_dist[bucket] += 1
                
            file_total_token_counts.append(file_tokens)
            
        except Exception as e:
            print(f"Error reading {fpath}: {e}")

    print("\n" + "="*40)
    print(f"ANALYSIS REPORT ({len(files)} files)")
    print("="*40)
    
    if file_section_counts:
        print(f"\n[Sections per File]")
        print(f"  Mean: {statistics.mean(file_section_counts):.2f}")
        print(f"  Median: {statistics.median(file_section_counts):.2f}")
        print(f"  Min: {min(file_section_counts)}")
        print(f"  Max: {max(file_section_counts)}")
        print(f"  Stdev: {statistics.stdev(file_section_counts) if len(file_section_counts)>1 else 0:.2f}")

    if file_total_token_counts:
        print(f"\n[Total Tokens per File]")
        print(f"  Mean: {statistics.mean(file_total_token_counts):.2f}")
        print(f"  Median: {statistics.median(file_total_token_counts):.2f}")
        print(f"  Min: {min(file_total_token_counts)}")
        print(f"  Max: {max(file_total_token_counts)}")
    
    if section_token_counts:
        print(f"\n[Tokens per Section]")
        print(f"  Mean: {statistics.mean(section_token_counts):.2f}")
        print(f"  Median: {statistics.median(section_token_counts):.2f}")
        print(f"  Min: {min(section_token_counts)}")
        print(f"  Max: {max(section_token_counts)}")
        print(f"  Stdev: {statistics.stdev(section_token_counts) if len(section_token_counts)>1 else 0:.2f}")
        
        print(f"\n[Section Length Distribution (Tokens)]")
        for bucket in sorted(token_dist.keys()):
            if bucket < 2000:
                range_str = f"{bucket}-{bucket+99}"
                print(f"  {range_str:<10}: {token_dist[bucket]}")
            elif bucket == 2000:
                print(f"  2000+     : {sum(v for k,v in token_dist.items() if k >= 2000)}")

    print("\n" + "="*40)
    print("RECOMMENDATION")
    if section_token_counts:
        median_tokens = statistics.median(section_token_counts)
        print(f"Median tokens/section is {median_tokens:.0f}.")
        
        if median_tokens < 600:
            print("-> Sections are relatively small. Consider MERGING sections or using a fixed chunk size of ~1000-2000 tokens.")
            print("   If using structure-aware chunking, merge until ~1500 tokens.")
        elif median_tokens > 2000:
            print("-> Sections are large. Consider SPLITTING sections into smaller chunks of ~1000-2000 tokens.")
            print("   Overlap of 10-20% (e.g. 200 tokens) is standard.")
        else:
            print("-> Sections are in a good range (600-2000 tokens). Can likely use 1 dictionary/chunk.")

        print("Standard Overlap Recommendation: 200 tokens (provides context without excessive redundancy).")

if __name__ == "__main__":
    analyze_chunks("data/PMC_Part1")
