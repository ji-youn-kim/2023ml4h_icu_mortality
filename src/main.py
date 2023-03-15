import argparse
import logging
import os
import sys
import datetime

from preprocess import *

logging.basicConfig(
    format="%(asctime)s | %(levelname)s %(name)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    level=os.environ.get("LOGLEVEL", "INFO").upper(),
    stream=sys.stdout,
)
logger = logging.getLogger(__name__)

def get_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device_num", type=str, default="4")

    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--save_prefix", type=str, default="checkpoint")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--lr", type=float, default=5e-3)

    parser.add_argument("--preprocess", action="store_true", default=False)
    parser.add_argument("--min_los", type=int, default=1, help="Minimum length-of-stay")
    parser.add_argument("--max_los", type=int, default=2, help="Maximum length-of-stay")
    parser.add_argument("--chunksize", type=int, default=100000, help="Chunk size for reading large csvs")
    parser.add_argument("--time_limit", type=int, default=3, help="First few hours of events from intime to consider")
    parser.add_argument("--max_event_cnt", type=int, default=300, help="Maximum number of events for each sample")

    # Wandb
    # parser.add_argument("--wandb_entity_name", type=str, required=True)
    # parser.add_argument("--wandb_project_name", type=str, required=True)
    
    return parser

if __name__ == "__main__":
    args = get_parser().parse_args()

    if args.preprocess:
        cohort = build_cohort(args)
        cohort, icu_dict = process_events(args, cohort)
    else:
        cohort = pd.read_csv(os.path.join(output_path, "cohort.csv"))
        icu_dict = pd.read_pickle(os.path.join(args.output_path), "icustays.pkl")
    
    #TODO