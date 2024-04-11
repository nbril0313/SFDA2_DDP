import math
from collections import OrderedDict

import numpy as np
import torch
import torch.nn as nn
import torch.nn.utils.weight_norm as weightNorm
import torchvision
from timm.models.convnext import ConvNeXt
from torch.autograd import Variable
from torchvision import models
from torchvision.models.resnet import (
    ResNet18_Weights,
    ResNet34_Weights,
    ResNet50_Weights,
    ResNet101_Weights,
    ResNet152_Weights,
    ResNeXt50_32X4D_Weights,
    ResNeXt101_32X8D_Weights,
)


def calc_coeff(iter_num, high=1.0, low=0.0, alpha=10.0, max_iter=10000.0):
    return np.float32(2.0 * (high - low) / (1.0 + np.exp(-alpha*iter_num / max_iter)) - (high - low) + low)

def init_weights(m):
    classname = m.__class__.__name__
    if classname.find('Conv2d') != -1:
        nn.init.kaiming_uniform_(m.weight)
        nn.init.zeros_(m.bias)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        nn.init.zeros_(m.bias)
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        nn.init.zeros_(m.bias)


res_dict = {
    "resnet18": (models.resnet18, ResNet18_Weights.IMAGENET1K_V1),
    "resnet34": (models.resnet34, ResNet34_Weights.IMAGENET1K_V1),
    "resnet50": (models.resnet50, ResNet50_Weights.IMAGENET1K_V1),
    "resnet101": (models.resnet101, ResNet101_Weights.IMAGENET1K_V1),
    "resnet152": (models.resnet152, ResNet152_Weights.IMAGENET1K_V1),
    "resnext50": (models.resnext50_32x4d, ResNeXt50_32X4D_Weights.IMAGENET1K_V1),
    "resnext101": (models.resnext101_32x8d, ResNeXt101_32X8D_Weights.IMAGENET1K_V1),
}

class ResBase(nn.Module):
    def __init__(self, res_name):
        super(ResBase, self).__init__()
        model_func, weights = res_dict[res_name]
        model_resnet = model_func(weights=weights)
        self.conv1 = model_resnet.conv1
        self.bn1 = model_resnet.bn1
        self.relu = model_resnet.relu
        self.maxpool = model_resnet.maxpool
        self.layer1 = model_resnet.layer1
        self.layer2 = model_resnet.layer2
        self.layer3 = model_resnet.layer3
        self.layer4 = model_resnet.layer4
        self.avgpool = model_resnet.avgpool
        self.in_features = model_resnet.fc.in_features

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return x



class feat_bootleneck(nn.Module):
    def __init__(self, feature_dim, bottleneck_dim=256, type="ori"):
        super().__init__()
        self.bn = nn.BatchNorm1d(bottleneck_dim, affine=True, )
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p=0.5)
        self.bottleneck = nn.Linear(feature_dim, bottleneck_dim)
        self.bottleneck.apply(init_weights)
        self.type = type

    def forward(self, x):
        x = self.bottleneck(x)
        if self.type == "bn":
            x = self.bn(x)
        out=x
        return out




class feat_classifier(nn.Module):
    def __init__(self, class_num, bottleneck_dim=256, type="linear"):
        super(feat_classifier, self).__init__()
        self.type = type
        if type == 'wn':
            self.fc = weightNorm(nn.Linear(bottleneck_dim, class_num), name="weight", )
            self.fc.apply(init_weights)
        else:
            self.fc = nn.Linear(bottleneck_dim, class_num)
            self.fc.apply(init_weights)

    def forward(self, x):
        x = self.fc(x)
        return x

class ConvNeXtBase(ConvNeXt):
    def __init__(self, conv_name="convnext_tiny", pretrained=True):
        # ConvNeXt 모델 구성에 따라 깊이와 차원을 설정
        if conv_name == "convnext_tiny":
            depths = [3, 3, 9, 3]
            dims = [96, 192, 384, 768]
        else:
            # 다른 ConvNeXt 모델 구성이 필요한 경우 여기에 추가
            raise ValueError(f"Unsupported ConvNeXt model name: {conv_name}")

        super().__init__(depths=depths, dims=dims)

        self.output_dim = dims[-1]  # 마지막 차원을 출력 차원으로 설정
        if pretrained:
            self._load_pretrained_weights(conv_name)

    def _load_pretrained_weights(
        self,
        conv_name,
        url="https://dl.fbaipublicfiles.com/convnext/convnext_tiny_22k_224.pth",
    ):
        checkpoint = torch.hub.load_state_dict_from_url(
            url=url, map_location="cpu", check_hash=True
        )
        if "model" in checkpoint:
            param_dict = checkpoint["model"]
        elif "state_dict" in checkpoint:
            param_dict = checkpoint["state_dict"]
        else:
            param_dict = checkpoint
        load_flag = True
        import re

        for k, v in param_dict.items():
            k = k.replace("downsample_layers.0.", "stem.")
            k = re.sub(r"stages.([0-9]+).([0-9]+)", r"stages.\1.blocks.\2", k)
            k = re.sub(
                r"downsample_layers.([0-9]+).([0-9]+)", r"stages.\1.downsample.\2", k
            )
            k = k.replace("dwconv", "conv_dw")
            k = k.replace("pwconv", "mlp.fc")
            if "grn" in k:
                k = k.replace("grn.beta", "mlp.grn.bias")
                k = k.replace("grn.gamma", "mlp.grn.weight")
                v = v.reshape(v.shape[-1])
            k = k.replace("head.", "head.fc.")
            if k.startswith("norm."):
                k = k.replace("norm", "head.norm")
            if "head" in k:
                continue
            try:
                self.state_dict()[k].copy_(v)
            except:
                print("===========================ERROR=========================")
                print(
                    "shape do not match in k :{}: param_dict{} vs self.state_dict(){}".format(
                        k, v.shape, self.state_dict()[k].shape
                    )
                )
                load_flag = False

        # if ~load_flag:
        #     raise Exception(f'load_state_dict from {url} fail')

    def forward(self, x):
        # 모델의 특징 추출 부분만 사용
        x = self.forward_features(x)
        # 전역 평균 풀링 적용
        x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)  # 특징 벡터를 평탄화
        return x

    @property
    def in_features(self):
        # 외부에서 모델의 입력 특징 차원에 접근할 수 있게 함
        return self.output_dim
