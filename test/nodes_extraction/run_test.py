import os
from evaluation.runner import run_experiment

def main():
   
    config_path = "test/nodes_extraction/config.yaml"
    print(f"Using config file at: {config_path}")
    run_experiment(config_path=config_path)

if __name__ == "__main__":
    main()
