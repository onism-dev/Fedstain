from argparse import ArgumentParser, Namespace
from copy import deepcopy
from pathlib import Path
import sys

PROJECT_DIR = Path(__file__).parent.parent.parent.absolute()
sys.path.append(PROJECT_DIR.as_posix())
from algorithm.server.fedavg import FedAvgServer, get_fedavg_argparser
from algorithm.client.fedprox import FedProxClient
from data.dataset import FLDataset


def get_fedprox_argparser():
    parser = get_fedavg_argparser()
    parser.add_argument("--mu", type=float, default=0.01)
    return parser


class FedProxServer(FedAvgServer):
    def __init__(self, algo="FedProx", args: Namespace = None):
        if args is None:
            args = get_fedprox_argparser().parse_args()
        super().__init__(algo, args)

    def initialize_clients(self):
        self.client_list = [
            FedProxClient(self.args, FLDataset(self.args, client_id), client_id, self.logger)
            for client_id in range(self.num_client)
        ]


if __name__ == "__main__":
    server = FedProxServer()
    server.process_classification()
