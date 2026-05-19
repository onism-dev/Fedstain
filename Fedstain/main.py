from copy import deepcopy
import os
import pickle
import sys
from argparse import ArgumentParser
from multiprocessing import Pool, cpu_count

import numpy as np
import pandas as pd
import torch
from rich.console import Console

from data.partition_data import ALL_DOMAINS, get_partition_arguments, partition_and_statistic
from algorithm.server.fedavg import FedAvgServer, get_fedavg_argparser
from algorithm.server.fedprox import FedProxServer, get_fedprox_argparser
from algorithm.server.fedsr import FedSRServer, get_fedsr_argparser
from algorithm.server.GA import GAServer, get_GA_argparser
from algorithm.server.fediir import FedIIRServer, get_fediir_argparser
from algorithm.server.fedadg import FedADGServer, get_fedadg_argparser
from algorithm.server.ccst import CCSTServer, get_ccst_argparser
from algorithm.server.fedstain import FedStainServer, get_fedstain_argparser
from utils.tools import local_time

algo2server = {
    "FedAvg": FedAvgServer,
    "FedProx": FedProxServer,
    "FedSR": FedSRServer,
    "GA": GAServer,
    "FedIIR": FedIIRServer,
    "FedADG": FedADGServer,
    "CCST": CCSTServer,
    "FedStain": FedStainServer,
}
algo2argparser = {
    "FedAvg": get_fedavg_argparser(),
    "FedProx": get_fedprox_argparser(),
    "FedSR": get_fedsr_argparser(),
    "GA": get_GA_argparser(),
    "FedIIR": get_fediir_argparser(),
    "FedADG": get_fedadg_argparser(),
    "CCST": get_ccst_argparser(),
    "FedStain": get_fedstain_argparser(),
}


def get_output_dir(args, algo_name, begin_time):
    if algo_name in ["FedAvg", "GA", "FedADG", "FedStain"]:
        return begin_time
    if algo_name == "CCST":
        output_dir = f"k_{args.k}_upload_ratio_{args.upload_ratio}"
    elif algo_name == "FedSR":
        output_dir = f"L2R_{args.L2R_coeff}_CMI_{args.CMI_coeff}"
    elif algo_name == "FedIIR":
        output_dir = f"gamma_{args.gamma}_ema_{args.ema}"
    elif algo_name == "FedProx":
        output_dir = f"mu_{args.mu}"
    else:
        output_dir = f"default_{algo_name}"
    return f"{output_dir}_{begin_time}"


def get_main_argparser():
    parser = ArgumentParser(description="FedStain and baseline federated learning.")
    parser.add_argument(
        "-a", "--algo", type=str, default="FedStain", choices=list(algo2server.keys())
    )
    parser.add_argument(
        "-d",
        "--dataset",
        type=str,
        default="came",
        choices=["pacs", "vlcs", "office_home", "came", "midog"],
    )
    return parser


def process(test_domain):
    if resume_dataset_dir is None:
        data_args = get_partition_arguments()
        data_args.dataset = dataset
        data_args.test_domain = test_domain
        dir_name = os.path.join(begin_time, test_domain)
        data_args.directory_name = dir_name
        if data_args.dataset == "midog":
            data_args.num_clients_per_domain = 2
            data_args.hetero_method = "dirichlet"
            data_args.alpha = 0.5
        partition_and_statistic(deepcopy(data_args))
    else:
        dir_name = os.path.join(resume_dataset_dir, test_domain)

    fl_args, _ = algo2argparser[algo].parse_known_args()
    fl_args.dataset = dataset
    fl_args.partition_info_dir = dir_name
    fl_args.output_dir = (
        get_output_dir(fl_args, algo, begin_time)
        if resume_run_log_dir is None
        else resume_run_log_dir
    )
    if "domainnet" in fl_args.dataset:
        fl_args.batch_size = 128
    if algo == "FedADG":
        fl_args.optimizer = "sgd"

    if fl_args.dataset in ("came", "midog"):
        fl_args.batch_size = 32 if fl_args.dataset == "came" else 16
        fl_args.round = 3
        fl_args.num_epochs = 3
        fl_args.lr = 0.0001
        fl_args.model = "res50"
        fl_args.optimizer = "adam"
        fl_args.augment = True

    server = algo2server[algo](args=deepcopy(fl_args))
    server.process_classification()


def get_table():
    test_accuracy = {}
    args, _ = algo2argparser[algo].parse_known_args()
    path2dir = os.path.join(
        "out",
        algo,
        dataset,
        get_output_dir(args, algo, begin_time)
        if resume_run_log_dir is None
        else resume_run_log_dir,
    )
    for domain in domains:
        with open(os.path.join(path2dir, domain, "test_accuracy.pkl"), "rb") as f:
            test_accuracy[domain] = round(pickle.load(f), 2)
    test_accuracy["average"] = round(np.mean(list(test_accuracy.values())), 2)
    pd.DataFrame(test_accuracy, index=[algo]).to_csv(
        os.path.join(path2dir, "test_accuracy.csv")
    )
    return test_accuracy


if __name__ == "__main__":
    begin_time = local_time()
    algo = sys.argv[1]
    assert algo in algo2server
    del sys.argv[1]

    if "-d" in sys.argv or "--dataset" in sys.argv:
        try:
            index = sys.argv.index("-d")
        except ValueError:
            index = sys.argv.index("--dataset")
        dataset = sys.argv[index + 1]
        assert dataset in ALL_DOMAINS
    else:
        raise ValueError("Please specify the dataset with -d.")

    resume_run_log_dir = None
    resume_dataset_dir = None

    domains = ALL_DOMAINS[dataset]
    multiprocess = dataset not in ("came", "midog")

    if dataset in ("came", "midog"):
        Console().print(
            f"[bold green]Running {algo} on {dataset}[/] "
            f"(serial, ResNet50, lr=1e-4)"
        )

    if multiprocess:
        num_processes = min(len(domains), cpu_count())
        pool = Pool(processes=num_processes)
        try:
            pool.map(process, domains)
            pool.close()
            pool.join()
        except Exception as e:
            pool.terminate()
            pool.join()
            raise RuntimeError("An error occurred in a worker process.") from e
    else:
        for domain in domains:
            process(domain)
    get_table()
