from argparse import ArgumentParser, Namespace
from copy import deepcopy
from pathlib import Path
import sys
from typing import List, OrderedDict


PROJECT_DIR = Path(__file__).parent.parent.parent.absolute()
sys.path.append(PROJECT_DIR.as_posix())
from algorithm.server.fedavg import FedAvgServer, get_fedavg_argparser
from algorithm.client.fedadg import FedADGClient
from data.dataset import FLDataset
from model.models import get_FedADG_models
from utils.tools import get_best_device


def get_fedadg_argparser():
    parser = get_fedavg_argparser()
    parser.add_argument("--lambda_0", type=float, default=0.55, help="lambda_1=1-lambda_0")
    parser.add_argument("--E1", type=int, default=3, help="E0 equals to num_epochs in fedavg args ")
    parser.add_argument(
        "--disc_lr", type=float, default=0.0007, help="Learning rate for discriminator"
    )
    parser.add_argument("--gen_lr", type=float, default=0.0007, help="Learning rate for generator")
    return parser


class FedADGServer(FedAvgServer):
    def __init__(self, algo="FedADG", args: Namespace = None):
        if args is None:
            args = get_fedadg_argparser().parse_args()
        super().__init__(algo, args)

    def initialize_model(self):
        self.classification_model, self.discriminator, self.generator = get_FedADG_models(
            self.args.model, self.args.dataset
        )
        self.device = get_best_device(self.args.use_cuda)

    def initialize_clients(self):
        self.client_list = [
            FedADGClient(self.args, FLDataset(self.args, client_id), client_id, self.logger)
            for client_id in range(self.num_client)
        ]

    def aggregate_model(self) -> List[OrderedDict]:
        self.agg_weight = self.get_agg_weight()
        model_weight_each_client = [client.get_model_weights() for client in self.client_list]
        classification_weights, discriminator_weights, generator_weights = zip(
            *model_weight_each_client
        )
        aggregated_weights = []
        for weights_list in [classification_weights, discriminator_weights, generator_weights]:
            new_model_weight = {}
            for key in weights_list[0].keys():
                new_model_weight[key] = sum(
                    [
                        model_weight[key] * weight
                        for model_weight, weight in zip(weights_list, self.agg_weight)
                    ]
                )
            aggregated_weights.append(new_model_weight)

        return aggregated_weights

    def process_classification(self):
        # distribute original model weights to clients
        for client_id in range(self.num_client):
            self.client_list[client_id].load_model_weights(
                [
                    deepcopy(self.classification_model.state_dict()),
                    deepcopy(self.discriminator.state_dict()),
                    deepcopy(self.generator.state_dict()),
                ]
            )
        self.best_accuracy = 0
        for round_id in range(self.args.round):
            self.logger.log("=" * 20, f"Round {round_id}", "=" * 20)
            for client_id in range(self.num_client):
                self.client_list[client_id].train()

            aggregated_weights = self.aggregate_model()
            self.classification_model.load_state_dict(aggregated_weights[0])
            for client_id in range(self.num_client):
                self.client_list[client_id].load_model_weights(aggregated_weights)
            if (round_id + 1) % self.args.test_gap == 0:
                self.validate_and_test()


if __name__ == "__main__":
    server = FedADGServer()
    server.process_classification()
