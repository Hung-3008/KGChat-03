import os
from evaluation.runner import run_experiment

def main():
   
    config_path = "test/nodes_extraction/config.yaml"
    run_experiment(config_path=config_path)

if __name__ == "__main__":
    main()
