import argparse
import logging
import os
import sys
import pandas as pd

from preprocess.preprocess import build_cohort, process_events, batch_itemids
from trainers.rnn_trainer import RNNTrainer
from trainers.logistic_trainer import LRTrainer

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

    parser.add_argument("--source_path", type=str)
    parser.add_argument("--input_path", type=str, required=True)
    parser.add_argument("--output_path", type=str, required=True)
    parser.add_argument("--save_prefix", type=str, default="checkpoint")

    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--batch_size", type=int, default=1024)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--lr", type=float, default=5e-3)

    parser.add_argument("--preprocess", action="store_true", default=False)
    parser.add_argument("--min_los", type=int, default=1, help="Minimum length-of-stay")
    parser.add_argument("--max_los", type=int, default=2, help="Maximum length-of-stay")
    parser.add_argument("--chunksize", type=int, default=100000, help="Chunk size for reading large csvs")
    parser.add_argument("--time_limit", type=int, default=3, help="First few hours of events from intime to consider")
    parser.add_argument("--max_event_cnt", type=int, default=300, help="Maximum number of events for each sample")

    parser.add_argument("--embed_dim", type=int, default=128)
    parser.add_argument("--hidden_dim", type=int, default=256)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--n_class", type=int, default=2)

    # Wandb
    parser.add_argument("--wandb_entity_name", type=str, required=True)
    parser.add_argument("--wandb_project_name", type=str, required=True)
    
    return parser


if __name__ == "__main__":
    
    args = get_parser().parse_args()

    if args.preprocess:
        train_cohort, test_cohort = build_cohort(args)
        train_icu_dict, test_icu_dict = process_events(args, train_cohort, test_cohort)
        _, _ = batch_itemids(args, train_icu_dict, test_icu_dict)

    trainer = RNNTrainer(args)
    trainer.train()

    LRTrainer(args)