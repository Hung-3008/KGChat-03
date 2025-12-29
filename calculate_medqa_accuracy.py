import json
import os

def calculate_accuracy(file_path):
    if not os.path.exists(file_path):
        print(f"Error: File not found at {file_path}")
        return

    total_count = 0
    correct_count = 0

    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                    if 'is_correct' in data:
                        total_count += 1
                        correct_count += data['is_correct']
                except json.JSONDecodeError:
                    print(f"Warning: Could not decode JSON line: {line[:50]}...")
                    continue
        
        if total_count == 0:
            print("No valid entries found with 'is_correct' field.")
        else:
            accuracy = correct_count / total_count
            print(f"Total Questions: {total_count}")
            print(f"Correct Answers: {correct_count}")
            print(f"Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    # Path to the file
    file_path = "results/medqa_evaluation_results.jsonl"
    if not os.path.exists(file_path):
        # Fallback to absolute path if running from different cwd
        file_path = "/media/hung/data1/codes/projects/FHC/results/medqa_evaluation_results.jsonl"
    
    calculate_accuracy(file_path)
