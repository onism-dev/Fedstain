import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import AugMix
from torchvision import transforms
import random
from algorithm.client.fedavg import FedAvgClient
from utils.optimizers_shcedulers import get_optimizer
from model.models import get_model_arch, vgg, decoder
from utils.tools import trainable_params, get_best_device, local_time
from utils.optimizers_shcedulers import get_optimizer, CosineAnnealingLRWithWarmup


class CCSTClient(FedAvgClient):
    def __init__(self, args, dataset, client_id, logger):
        super(CCSTClient, self).__init__(args, dataset, client_id, logger)

    def initialize_model(self):
        self.classification_model = get_model_arch(model_name=self.args.model)(
            dataset=self.args.dataset
        )
        self.device = None
        self.optimizer = get_optimizer(
            self.classification_model,
            self.args.optimizer,
            self.args.lr,
            weight_decay=self.args.weight_decay,
        )
        self.scheduler = CosineAnnealingLRWithWarmup(
            optimizer=self.optimizer,
            total_epochs=self.args.num_epochs * self.args.round,
        )
        self.vgg = vgg
        self.decoder = decoder
        self.vgg.eval()
        self.decoder.eval()
        self.vgg.load_state_dict(torch.load(self.args.vgg_path, weights_only=True))
        self.decoder.load_state_dict(
            torch.load(self.args.decoder_path, weights_only=True)
        )
        self.vgg = nn.Sequential(*list(vgg.children())[:31])

    def move2new_device(self):
        device = get_best_device(self.args.use_cuda)
        # device = torch.device("cuda:0")
        self.classification_model.to(device)
        self.vgg.to(device)
        self.decoder.to(device)
        if self.device is None or self.device != device:
            optimizer_state = self.optimizer.state_dict()
            scheduler_state = self.scheduler.state_dict()
            self.optimizer = get_optimizer(
                self.classification_model,
                self.args.optimizer,
                self.args.lr,
                weight_decay=self.args.weight_decay,
            )
            self.scheduler = CosineAnnealingLRWithWarmup(
                optimizer=self.optimizer,
                total_epochs=self.args.num_epochs * self.args.round,
            )
            self.optimizer.load_state_dict(optimizer_state)
            self.scheduler.load_state_dict(scheduler_state)
            self.dataset.device = device
        self.device = device

    @torch.no_grad()
    def compute_statistic(self):
        device = get_best_device(self.args.use_cuda)
        self.vgg.to(device)
        single_style_bank = {"mean": [], "std": []}
        overall_style_bank = {"mean": [], "std": []}
        total_batches = len(self.train_loader)
        assert total_batches * self.args.upload_ratio > 1
        feature_lst = []
        for enu, (data, _) in enumerate(self.train_loader):
            if enu + 1 >= total_batches * self.args.upload_ratio:
                break
            data = data.to(device)
            feature = self.vgg(data)
            feature_lst.append(feature)
        all_feature = torch.cat(feature_lst, dim=0)
        single_feature_mean = torch.mean(all_feature, dim=(2, 3), keepdim=True)
        single_feature_var = torch.var(all_feature, dim=(2, 3), keepdim=True)
        single_feature_std: torch.Tensor = (
            single_feature_var + self.args.epsilon
        ).sqrt()
        N, C = all_feature.size()[:2]
        overall_feature_var = all_feature.view(1, C, -1).var(dim=2) + self.args.epsilon
        overall_feature_std = overall_feature_var.sqrt().view(1, C, 1, 1)
        overall_feature_mean = all_feature.view(1, C, -1).mean(dim=2).view(1, C, 1, 1)
        single_style_bank["mean"] = single_feature_mean.to(torch.device("cpu"))
        single_style_bank["std"] = single_feature_std.to(torch.device("cpu"))
        overall_style_bank["mean"] = overall_feature_mean.to(torch.device("cpu"))
        overall_style_bank["std"] = overall_feature_std.to(torch.device("cpu"))
        self.vgg.to(torch.device("cpu"))
        return single_style_bank, overall_style_bank

    def download_statistic_pool(self, single_style_bank, overall_style_bank):
        self.single_style_bank = {}
        self.overall_style_bank = {}
        single_style_bank["mean"].pop(self.client_id)
        single_style_bank["std"].pop(self.client_id)
        overall_style_bank["mean"].pop(self.client_id)
        overall_style_bank["std"].pop(self.client_id)
        self.single_style_bank["mean"] = torch.cat(single_style_bank["mean"], dim=0)
        self.single_style_bank["std"] = torch.cat(single_style_bank["std"], dim=0)
        self.overall_style_bank["mean"] = torch.cat(overall_style_bank["mean"], dim=0)
        self.overall_style_bank["std"] = torch.cat(overall_style_bank["std"], dim=0)

    def sample_statistic(self, current_batch_size):
        num = self.statistic_pool["mean"].shape[0]
        if num >= current_batch_size:
            indices = torch.randperm(num)[:current_batch_size]
        else:
            indices = torch.randint(0, num, (current_batch_size,))
        sampled_mean = self.statistic_pool["mean"][indices]
        sampled_std = self.statistic_pool["std"][indices]
        return sampled_mean, sampled_std

    @torch.no_grad()
    def style_transfer(self, data):
        rdn = random.uniform(0, 1)
        if rdn < 0.5:
            mu = self.single_style_bank["mean"].to(data.device)
            std = self.single_style_bank["std"].to(data.device)
        else:
            mu = self.overall_style_bank["mean"].to(data.device)
            std = self.overall_style_bank["std"].to(data.device)
        num = len(data)
        if num > len(mu):
            indices = torch.randint(0, len(mu), (num,))
        else:
            indices = torch.randperm(len(mu))[:num]
        mu = mu[indices]
        std = std[indices]
        feature = self.vgg(data)
        # normalize features
        feature_mean = feature.mean(dim=(2, 3), keepdim=True)
        feature_std = feature.std(dim=(2, 3), keepdim=True) + self.args.epsilon
        normalized_feature = (feature - feature_mean) / feature_std
        # style transfer
        stylized_feature = normalized_feature * std + mu
        stylized_data = self.decoder(stylized_feature)
        return stylized_data

    def train(self):
        self.move2new_device()
        self.classification_model.train()
        criterion = torch.nn.CrossEntropyLoss()
        for _ in range(self.args.num_epochs):
            total_loss = 0.0
            for batch_idx, (data, target) in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                output = self.classification_model(data)
                loss = criterion(output, target)
                output = F.softmax(output, dim=1)
                k = random.randint(0, self.args.k)
                for _ in range(k):
                    generated_data = self.style_transfer(data)
                    output = self.classification_model(generated_data)
                    loss += criterion(output, target)
                loss.backward()
                self.optimizer.step()
                total_loss += loss.item()
            self.scheduler.step()
        average_loss = total_loss / len(self.train_loader)
        self.classification_model.to(torch.device("cpu"))
        self.vgg.to(torch.device("cpu"))
        self.decoder.to(torch.device("cpu"))
        del self.single_style_bank
        del self.overall_style_bank
        torch.cuda.empty_cache()
        self.logger.log(
            f"{local_time()}, Client {self.client_id}, Avg Loss: {average_loss:.4f}"
        )
