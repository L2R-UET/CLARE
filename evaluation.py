import argparse
import os
from dotenv import load_dotenv
from utils.metrics import MetricScore

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Metric Evaluation Configuration")
    parser.add_argument('--dataset_name', type=str, default='amazon', help='Name of the dataset')
    parser.add_argument('--has_gpt', type=bool, default=True, help='Use ChatGPT for evaluation')
    return parser.parse_args()

if __name__ == "__main__":
    load_dotenv()
    args = parse_args()
    api_key = os.getenv("OPENAI_API_KEY") if args.has_gpt else None
    experiment = MetricScore(args.dataset_name, api_key=api_key)
    experiment.print_score()
    experiment.report_score()