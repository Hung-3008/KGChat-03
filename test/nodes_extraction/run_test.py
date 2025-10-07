import os
from evaluation.runner import run_experiment

def main():
    """Loads configuration from config.yaml and runs the experiment."""
    config_path = os.path.join(os.path.dirname(__file__), "config.yaml")
    print(f"Using config file at: {config_path}")
    run_experiment(config_path=config_path)

if __name__ == "__main__":
    main()
