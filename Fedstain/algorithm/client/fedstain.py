import random

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
from torchvision import transforms
from torchvision.transforms import AugMix

from algorithm.client.fedavg import FedAvgClient
from model.models import MixStyle
from utils.tools import local_time


class FedStainClient(FedAvgClient):
    def __init__(self, args, dataset, client_id, logger):
        super().__init__(args, dataset, client_id, logger)
        self.MixStyle = MixStyle(self.args.p, 0.1, self.args.epsilon)

    @torch.no_grad()
    def compute_statistic(self):
        self.move2new_device()
        # Keys "mean"/"std" hold skewness and kurtosis for wire compatibility.
        local_statistic_pool = {"mean": [], "std": []}
        num2upload = int(len(self.train_loader.dataset) * self.args.r)
        batches = int(num2upload / self.args.batch_size)
        left_num = num2upload % self.args.batch_size

        for enu, (data, target) in enumerate(self.train_loader):
            mu = torch.mean(data, dim=(2, 3), keepdim=True)
            var = torch.var(data, dim=(2, 3), keepdim=True)
            sigma = (var + self.args.epsilon).sqrt()

            centered = data - mu
            sigma_safe = torch.clamp(sigma, min=self.args.epsilon)
            skewness = torch.mean(centered ** 3, dim=(2, 3), keepdim=True) / (
                sigma_safe ** 3 + self.args.epsilon
            )
            skewness = torch.clamp(skewness, min=-10.0, max=10.0)

            kurtosis = (
                torch.mean(centered ** 4, dim=(2, 3), keepdim=True)
                / (sigma_safe ** 4 + self.args.epsilon)
            ) - 3
            kurtosis = torch.clamp(kurtosis, min=-10.0, max=10.0)

            if enu != batches:
                local_statistic_pool["mean"].append(skewness)
                local_statistic_pool["std"].append(kurtosis)
            else:
                local_statistic_pool["mean"].append(skewness[:left_num])
                local_statistic_pool["std"].append(kurtosis[:left_num])
                break

        local_statistic_pool["mean"] = torch.cat(
            local_statistic_pool["mean"], dim=0
        ).to(torch.device("cpu"))
        local_statistic_pool["std"] = torch.cat(
            local_statistic_pool["std"], dim=0
        ).to(torch.device("cpu"))
        return local_statistic_pool

    def download_statistic_pool(self, statistic_pool):
        self.statistic_pool = {}
        statistic_pool["mean"].pop(self.client_id)
        statistic_pool["std"].pop(self.client_id)
        statistic_pool["mean"] = [x.to(self.device) for x in statistic_pool["mean"]]
        statistic_pool["std"] = [x.to(self.device) for x in statistic_pool["std"]]
        self.statistic_pool["mean"] = torch.cat(statistic_pool["mean"], dim=0)
        self.statistic_pool["std"] = torch.cat(statistic_pool["std"], dim=0)

    def sample_statistic(self, current_batch_size):
        num = self.statistic_pool["mean"].shape[0]
        if num >= current_batch_size:
            indices = torch.randperm(num)[:current_batch_size]
        else:
            indices = torch.randint(0, num, (current_batch_size,))
        sampled_skew = self.statistic_pool["mean"][indices]
        sampled_kurt = self.statistic_pool["std"][indices]
        return sampled_skew, sampled_kurt

    def train(self):
        self.move2new_device()
        self.classification_model.train()
        criterion = torch.nn.CrossEntropyLoss()

        for epoch in range(self.args.num_epochs):
            total_loss = 0.0
            total_ce_loss = 0.0
            total_js_loss = 0.0
            total_contrastive_loss = 0.0
            for batch_idx, (data, target) in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                feature_1 = self.classification_model.base(data)
                output = self.classification_model.classifier(feature_1)
                ce_loss = criterion(output, target)
                output = F.softmax(output, dim=1)
                mix_output = []
                mix_feature = []

                for _ in range(2):
                    skew, kurt = self.sample_statistic(len(data))
                    generated_data = self.MixStyle(data, skew, kurt)
                    generated_data = self.AugMixAugmentation(generated_data)
                    feature = self.classification_model.base(generated_data)
                    pred = self.classification_model.classifier(feature)
                    ce_loss += criterion(pred, target)
                    if self.args.lambda2 > 0:
                        mix_output.append(F.softmax(pred, dim=1))
                    if self.args.lambda1 > 0:
                        mix_feature.append(feature)

                js_loss = 0.0
                if self.args.lambda2 > 0:
                    M = torch.clamp(
                        (output + mix_output[0] + mix_output[1]) / 3, 1e-7, 1
                    ).log()
                    kl_1 = F.kl_div(M, output, reduction="batchmean")
                    kl_2 = F.kl_div(M, mix_output[0], reduction="batchmean")
                    kl_3 = F.kl_div(M, mix_output[1], reduction="batchmean")
                    js_loss = (kl_1 + kl_2 + kl_3) / 3

                contrastive_loss = 0.0
                if self.args.lambda1 > 0:
                    contrastive_loss = (
                        self.supervised_contrastive_loss(
                            mix_feature[0],
                            feature_1,
                            target,
                            temperature=self.args.t,
                        )
                        + self.supervised_contrastive_loss(
                            mix_feature[1],
                            feature_1,
                            target,
                            temperature=self.args.t,
                        )
                    ) / 2

                total_batch_loss = (
                    ce_loss
                    + self.args.lambda2 * js_loss
                    + self.args.lambda1 * contrastive_loss
                )

                total_batch_loss.backward()
                self.optimizer.step()

                total_loss += total_batch_loss.item()
                total_ce_loss += ce_loss.item()
                total_js_loss += js_loss.item()
                total_contrastive_loss += contrastive_loss.item()

            self.scheduler.step()

            avg_total_loss = total_loss / len(self.train_loader)
            avg_ce_loss = total_ce_loss / len(self.train_loader)
            avg_js_loss = total_js_loss / len(self.train_loader)
            avg_contrastive_loss = total_contrastive_loss / len(self.train_loader)

            self.logger.log(
                f"{local_time()}, Client {self.client_id}, "
                f"Epoch {epoch + 1}/{self.args.num_epochs}: "
                f"Total: {avg_total_loss:.4f}, CE: {avg_ce_loss:.4f}, "
                f"JS: {avg_js_loss:.4f}, Contrastive: {avg_contrastive_loss:.4f}"
            )

        self.classification_model.to(torch.device("cpu"))
        del self.statistic_pool
        torch.cuda.empty_cache()

    def supervised_contrastive_loss(self, x, y, label, temperature):
        x_norm = torch.norm(x, dim=1, keepdim=True)
        y_norm = torch.norm(y, dim=1, keepdim=True)
        x = x / x_norm
        y = y / y_norm
        samples = torch.cat((x, y), dim=0)
        label = torch.cat((label, label), dim=0)
        same_label_matrix = torch.eq(label.unsqueeze(1), label.unsqueeze(0)).float()
        sim = torch.matmul(samples, samples.T) / temperature
        same_label_sim = sim * same_label_matrix
        same_label_num = torch.sum(same_label_matrix, dim=1)
        negative_sim = torch.exp(sim)
        negative_sum = torch.log(torch.sum(negative_sim, dim=1) - negative_sim.diag())
        positive_sum = torch.sum(same_label_sim, dim=1) / same_label_num
        return torch.mean(-positive_sum + negative_sum)

    def denormalize(self, tensor, mean, std):
        mean = torch.as_tensor(mean).reshape(1, -1, 1, 1).to(tensor.device)
        std = torch.as_tensor(std).reshape(1, -1, 1, 1).to(tensor.device)
        return tensor * std + mean

    def AugMixAugmentation(self, input_images):
        mean = torch.tensor([0.485, 0.456, 0.406]).to(input_images.device)
        std = torch.tensor([0.229, 0.224, 0.225]).to(input_images.device)
        input_images = self.denormalize(input_images, mean, std)
        input_images = input_images * 255.0
        input_images = input_images.to(torch.uint8)
        augmixed_images = AugMix()(input_images)
        augmixed_images = augmixed_images.float().div(255.0)
        return transforms.Normalize(mean, std)(augmixed_images)

    def scale2unit(self, tensor):
        return (tensor - tensor.min()) / (tensor.max() - tensor.min())

    def visualize_augmentation_effect(self, path2dir):
        data_iter = iter(self.train_loader)
        data, _ = next(data_iter)
        random.seed(None)
        random_index = random.randint(0, len(data) - 1)
        sample_image = data[random_index]
        skew, kurt = self.sample_statistic(1)
        mixstyled_image = self.MixStyle(
            sample_image.unsqueeze(0), skew, kurt
        ).squeeze(0)
        mixstyled_image = self.scale2unit(mixstyled_image)
        augmixed_image = self.AugMixAugmentation(
            mixstyled_image.unsqueeze(0)
        ).squeeze(0)
        sample_image = self.scale2unit(sample_image)
        mixstyled_image = self.scale2unit(mixstyled_image)
        augmixed_image = self.scale2unit(augmixed_image)

        plt.imsave(
            f"{path2dir}/original_image.png",
            sample_image.permute(1, 2, 0).cpu().numpy(),
        )
        plt.imsave(
            f"{path2dir}/mixstyle_image.png",
            mixstyled_image.permute(1, 2, 0).cpu().numpy(),
        )
        plt.imsave(
            f"{path2dir}/augmix_image.png",
            augmixed_image.permute(1, 2, 0).cpu().numpy(),
        )
