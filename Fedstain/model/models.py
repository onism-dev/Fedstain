import json
import functools
from collections import OrderedDict
import random
from typing import List, Optional
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.nn import init
from torch import Tensor
import torch.distributions as distributions

NUM_CLASSES = {
    "pacs": 7,
    "vlcs": 5,
    "office_home": 65,
    "domainnet": 345,
    "minidomainnet": 126,
    "came": 2,
    "midog": 3,
}


def get_model_arch(model_name):
    # static means the model arch is fixed.
    if "fedsr" not in model_name:
        if "res" in model_name:
            return functools.partial(ResNet, version=model_name[3:])
        if "mobile" in model_name:
            return functools.partial(MobileNet, version=model_name[6:])
    else:
        return functools.partial(
            Model_4_FedSR, base_model_name=model_name.split("_")[-1]
        )


class DecoupledModel(nn.Module):
    def __init__(self):
        super(DecoupledModel, self).__init__()
        self.need_all_features_flag = False
        self.all_features = []
        self.base: nn.Module = None
        self.classifier: nn.Module = None
        self.dropout: List[nn.Module] = []

    def need_all_features(self):
        target_modules = [
            module
            for module in self.base.modules()
            if isinstance(module, nn.Conv2d) or isinstance(module, nn.Linear)
        ]

        def get_feature_hook_fn(model, input, output):
            if self.need_all_features_flag:
                self.all_features.append(output.clone().detach())

        for module in target_modules:
            module.register_forward_hook(get_feature_hook_fn)

    def check_avaliability(self):
        if self.base is None or self.classifier is None:
            raise RuntimeError(
                "You need to re-write the base and classifier in your custom model class."
            )
        self.dropout = [
            module
            for module in list(self.base.modules()) + list(self.classifier.modules())
            if isinstance(module, nn.Dropout)
        ]

    def forward(self, x: Tensor) -> Tensor:
        return self.classifier(self.base(x))

    def get_final_features(self, x: Tensor, detach=True) -> Tensor:
        if len(self.dropout) > 0:
            for dropout in self.dropout:
                dropout.eval()

        func = (lambda x: x.clone().detach()) if detach else (lambda x: x)
        out = self.base(x)

        if len(self.dropout) > 0:
            for dropout in self.dropout:
                dropout.train()

        return func(out)

    def get_all_features(self, x: Tensor) -> Optional[List[Tensor]]:
        feature_list = None
        if len(self.dropout) > 0:
            for dropout in self.dropout:
                dropout.eval()

        self.need_all_features_flag = True
        _ = self.base(x)
        self.need_all_features_flag = False

        if len(self.all_features) > 0:
            feature_list = self.all_features
            self.all_features = []

        if len(self.dropout) > 0:
            for dropout in self.dropout:
                dropout.train()

        return feature_list


class ResNet(DecoupledModel):
    def __init__(self, version, dataset: str):
        super().__init__()
        archs = {
            "34": (models.resnet34, models.ResNet34_Weights.DEFAULT),
            "50": (models.resnet50, models.ResNet50_Weights.DEFAULT),
        }

        # NOTE: If you don't want parameters pretrained, set `pretrained` as False
        pretrained = True
        resnet: models.ResNet = archs[version][0](
            weights=archs[version][1] if pretrained else None
        )
        self.base = resnet
        self.feature_dim = self.base.fc.in_features
        self.classifier = nn.Linear(self.feature_dim, NUM_CLASSES[dataset])
        self.base.fc = nn.Identity()


class MobileNet(DecoupledModel):
    def __init__(self, version, dataset: str):
        super().__init__()
        archs = {
            "2": (models.mobilenet_v2, models.MobileNet_V2_Weights.DEFAULT),
            "3s": (
                models.mobilenet_v3_small,
                models.MobileNet_V3_Small_Weights.DEFAULT,
            ),
            "3l": (
                models.mobilenet_v3_large,
                models.MobileNet_V3_Large_Weights.DEFAULT,
            ),
        }
        # NOTE: If you don't want parameters pretrained, set `pretrained` as False
        pretrained = True
        mobilenet = archs[version][0](weights=archs[version][1] if pretrained else None)
        self.base = mobilenet
        self.feature_dim = self.base.classifier[-1].in_features
        self.classifier = nn.Linear(self.feature_dim, NUM_CLASSES[dataset])
        self.base.classifier[-1] = nn.Identity()
        pass


class Model_4_FedSR(DecoupledModel):
    # modify base model to suit FedSR
    def __init__(self, base_model_name, dataset) -> None:
        super().__init__()
        base_model = get_model_arch(base_model_name)(dataset=dataset)
        self.z_dim = base_model.classifier.in_features
        out_dim = 2 * self.z_dim
        if "mobile" in base_model_name:
            self.base = base_model.base
            # self.classifier = base_model.classifier
            self.base.classifier[-1] = nn.Linear(
                base_model.classifier.in_features, out_dim
            )
            self.classifier = nn.Linear(self.z_dim, NUM_CLASSES[dataset])
        elif "res" in base_model_name:
            self.base = base_model.base
            # self.classifier= base_model.classifier
            self.base.fc = nn.Linear(base_model.classifier.in_features, out_dim)
            self.classifier = nn.Linear(self.z_dim, NUM_CLASSES[dataset])
        self.r_mu = nn.Parameter(torch.zeros(NUM_CLASSES[dataset], self.z_dim))
        self.r_sigma = nn.Parameter(torch.ones(NUM_CLASSES[dataset], self.z_dim))
        self.C = nn.Parameter(torch.ones([]))

    def featurize(self, x, num_samples=1, return_dist=False):
        # designed for FedSR
        z_params = self.base(x)
        z_mu = z_params[:, : self.z_dim]
        z_sigma = F.softplus(z_params[:, self.z_dim :])
        z_dist = distributions.Independent(
            distributions.normal.Normal(z_mu, z_sigma), 1
        )
        z = z_dist.rsample([num_samples]).view([-1, self.z_dim])

        if return_dist:
            return z, (z_mu, z_sigma)
        else:
            return z

    def forward(self, x):
        z = self.featurize(x)
        logits = self.classifier(z)
        return logits


class FedADG_Discriminator(nn.Module):
    def __init__(self, num_labels, hidden_size, rp_size):
        super().__init__()
        self.features_pro = nn.Sequential(
            nn.Linear(rp_size, 1024),
            nn.LeakyReLU(),
            nn.Linear(1024, 1),
            nn.Sigmoid(),
        )
        self.optimizer = None
        self.projection = nn.Linear(hidden_size + num_labels, rp_size, bias=False)
        with torch.no_grad():
            self.projection.weight.div_(
                torch.norm(self.projection.weight, keepdim=True)
            )

    def forward(self, y, z):
        feature = z.view(z.size(0), -1)
        feature = torch.cat([feature, y], dim=1)
        feature = self.projection(feature)
        logit = self.features_pro(feature)
        return logit


class FedADG_GeneDistrNet(nn.Module):
    def __init__(self, num_labels, hidden_size):
        super(FedADG_GeneDistrNet, self).__init__()
        self.num_labels = num_labels
        self.input_size = hidden_size
        self.latent_size = 4096
        self.genedistri = nn.Sequential(
            nn.Linear(self.input_size + self.num_labels, self.latent_size),
            nn.LeakyReLU(),
            nn.Linear(self.latent_size, hidden_size),
            nn.ReLU(),
        )
        self.initial_params()

    def initial_params(self):
        for layer in self.modules():
            if isinstance(layer, torch.nn.Linear):
                init.xavier_uniform_(layer.weight, 0.5)

    def forward(self, x, y):
        x = torch.cat([x, y], dim=1)
        x = self.genedistri(x)
        return x


def get_FedADG_models(classification_model, dataset, rp_size=1024):
    classification_model = get_model_arch(model_name=classification_model)(
        dataset=dataset
    )
    discriminator = FedADG_Discriminator(
        NUM_CLASSES[dataset], classification_model.feature_dim, rp_size
    )
    generator = FedADG_GeneDistrNet(
        NUM_CLASSES[dataset], classification_model.feature_dim
    )
    return classification_model, discriminator, generator


class MixStyle(nn.Module):
    def __init__(self, p=1.0, alpha=0.1, eps=1e-6):
        super(MixStyle, self).__init__()
        self.p = p
        self.beta = torch.distributions.Beta(torch.tensor(alpha), torch.tensor(alpha))
        self.eps = eps
        self.alpha = alpha

    def forward(self, x, mu2, std2):  # mu2=采样的偏度，std2=采样的峰度
        if random.random() > self.p:
            return x

        B = x.size(0)
        # 1. 计算输入数据的原始均值和标准差（用于基础标准化）
        orig_mu = x.mean(dim=[2, 3], keepdim=True)  # 原始均值（临时变量）
        orig_var = x.var(dim=[2, 3], keepdim=True)  # 原始方差（临时变量）
        orig_std = (orig_var + self.eps).sqrt()  # 原始标准差（临时变量）

        # 2. 计算输入数据的偏度和峰度（对应原 mean 和 std 变量）
        centered = x - orig_mu
        # 添加数值稳定性检查
        orig_std_safe = torch.clamp(orig_std, min=self.eps)
        skewness = torch.mean(centered ** 3, dim=(2, 3), keepdim=True) / (orig_std_safe ** 3 + self.eps)  # 偏度
        kurtosis = (torch.mean(centered ** 4, dim=(2, 3), keepdim=True) / (orig_std_safe ** 4 + self.eps)) - 3  # 峰度
        
        # 限制偏度和峰度的范围以避免数值不稳定
        skewness = torch.clamp(skewness, min=-10.0, max=10.0)
        kurtosis = torch.clamp(kurtosis, min=-10.0, max=10.0)
        skewness, kurtosis = skewness.detach(), kurtosis.detach()

        # 3. 简化的风格混合（避免复杂的高阶矩校正）
        lmda = self.beta.sample((B, 1, 1, 1)).to(x.device)
        mu2 = mu2.to(x.device)  # 采样的偏度
        std2 = std2.to(x.device)  # 采样的峰度

        # 限制采样统计量的范围
        mu2 = torch.clamp(mu2, min=-10.0, max=10.0)
        std2 = torch.clamp(std2, min=-10.0, max=10.0)

        # 混合偏度和峰度
        skewness_mix = skewness * lmda + mu2 * (1 - lmda)  # 混合偏度
        kurtosis_mix = kurtosis * lmda + std2 * (1 - lmda)  # 混合峰度

        # 4. 简化的风格应用（使用更稳定的方法）
        x_normed = (x - orig_mu) / orig_std_safe  # 基础标准化
        
        # 使用更温和的风格混合，避免极端值
        style_factor = 0.1  # 降低风格混合的强度
        x_mixed = x_normed + style_factor * (skewness_mix - skewness) + style_factor * (kurtosis_mix - kurtosis)
        
        # 还原到原始尺度
        x_mixed = x_mixed * orig_std_safe + orig_mu
        
        # 最终数值稳定性检查
        x_mixed = torch.clamp(x_mixed, min=-10.0, max=10.0)

        return x_mixed


decoder = nn.Sequential(
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 256, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode="nearest"),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 128, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode="nearest"),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 64, (3, 3)),
    nn.ReLU(),
    nn.Upsample(scale_factor=2, mode="nearest"),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 3, (3, 3)),
)

vgg = nn.Sequential(
    nn.Conv2d(3, 3, (1, 1)),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(3, 64, (3, 3)),
    nn.ReLU(),  # relu1-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 64, (3, 3)),
    nn.ReLU(),  # relu1-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(64, 128, (3, 3)),
    nn.ReLU(),  # relu2-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 128, (3, 3)),
    nn.ReLU(),  # relu2-2
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(128, 256, (3, 3)),
    nn.ReLU(),  # relu3-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 256, (3, 3)),
    nn.ReLU(),  # relu3-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(256, 512, (3, 3)),
    nn.ReLU(),  # relu4-1, this is the last layer used
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu4-4
    nn.MaxPool2d((2, 2), (2, 2), (0, 0), ceil_mode=True),
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-1
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-2
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-3
    nn.ReflectionPad2d((1, 1, 1, 1)),
    nn.Conv2d(512, 512, (3, 3)),
    nn.ReLU(),  # relu5-4
)

if __name__ == "__main__":
    model = get_model_arch(model_name="res50")(dataset="pacs")
    input_tensor = torch.randn(5, 3, 224, 224)  # Generate a 3*224*224 tensor
    model.eval()  # Set the model to evaluation mode
    all_features = model.get_final_features(input_tensor)  # Get all features
    pass
