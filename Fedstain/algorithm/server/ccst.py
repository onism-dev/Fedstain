from argparse import ArgumentParser, Namespace
from copy import deepcopy
import os
from pathlib import Path
import sys
from typing import Dict, List, Union

from torch import Tensor
import torch

PROJECT_DIR = Path(__file__).parent.parent.parent.absolute()
sys.path.append(PROJECT_DIR.as_posix())
from algorithm.server.fedavg import FedAvgServer, get_fedavg_argparser
from algorithm.client.ccst import CCSTClient
from data.dataset import FLDataset


def get_ccst_argparser():
    parser = get_fedavg_argparser()
    parser.add_argument(
        "--decoder_path",
        type=str,
        default=os.path.join(
            PROJECT_DIR, "model", "ccst_pretrained_weight", "decoder.pth"
        ),
    )
    parser.add_argument(
        "--vgg_path",
        type=str,
        default=os.path.join(
            PROJECT_DIR, "model", "ccst_pretrained_weight", "vgg_normalised.pth"
        ),
    )
    parser.add_argument("--upload_ratio", type=float, default=0.5)
    parser.add_argument("--epsilon", type=float, default=1e-8)
    parser.add_argument("--k", type=float, default=3)
    return parser


class CCSTServer(FedAvgServer):
    def __init__(self, algo="CCST", args: Namespace = None):
        if args is None:
            args = get_ccst_argparser().parse_args()
        super().__init__(algo, args)

    def initialize_clients(self):
        self.client_list = [
            CCSTClient(
                self.args, FLDataset(self.args, client_id), client_id, self.logger
            )
            for client_id in range(self.num_client)
        ]

    def generate_style_bank(self):
        # generate style pool
        single_style_bank = {"mean": [], "std": []}
        overall_style_bank = {"mean": [], "std": []}
        for client_id in range(self.num_client):
            client_single_style_bank, client_overall_style_bank = self.client_list[
                client_id
            ].compute_statistic()
            single_style_bank["mean"].append(client_single_style_bank["mean"])
            single_style_bank["std"].append(client_single_style_bank["std"])
            overall_style_bank["mean"].append(client_overall_style_bank["mean"])
            overall_style_bank["std"].append(client_overall_style_bank["std"])
            torch.cuda.empty_cache()
        return single_style_bank, overall_style_bank

    def process_classification(self):
        # distribute original model weights to clients
        for client_id in range(self.num_client):
            self.client_list[client_id].load_model_weights(
                deepcopy(self.classification_model.state_dict())
            )

        self.best_accuracy = 0
        for round_id in range(self.args.round):
            self.logger.log("=" * 20, f"Round {round_id}", "=" * 20)
            # generate style pool
            single_style_bank, overall_style_bank = self.generate_style_bank()
            for client_id in range(self.num_client):
                self.client_list[client_id].download_statistic_pool(
                    deepcopy(single_style_bank), deepcopy(overall_style_bank)
                )
                self.client_list[client_id].train()

            aggregated_weights = self.aggregate_model()
            self.classification_model.load_state_dict(aggregated_weights)
            self.round_id = round_id
            for client_id in range(self.num_client):
                self.client_list[client_id].load_model_weights(aggregated_weights)
            if (round_id + 1) % self.args.test_gap == 0:
                self.validate_and_test()


if __name__ == "__main__":
    server = CCSTServer()
    server.process_classification()
