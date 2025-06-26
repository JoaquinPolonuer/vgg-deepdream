from collections import namedtuple

# Deep learning related imports
import torch
import torch.nn as nn
from torchvision import models

from config import DEVICE, SupportedModels, SupportedPretrainedWeights


class Vgg16Experimental(torch.nn.Module):

    def __init__(self, pretrained_weights, requires_grad=False, show_progress=False):
        super().__init__()

        # Only ImageNet weights are supported for now for this model
        if pretrained_weights == SupportedPretrainedWeights.IMAGENET.name:
            vgg16 = models.vgg16(
                weights=models.VGG16_Weights.IMAGENET1K_V1, progress=show_progress
            ).eval()
        else:
            raise Exception(
                f"Pretrained weights {pretrained_weights} not yet supported for {self.__class__.__name__} model."
            )

        # I just used the official PyTorch implementation to figure out how to dissect VGG16:
        # https://github.com/pytorch/vision/blob/master/torchvision/models/vgg.py
        vgg_pretrained_features = vgg16.features
        vgg_avgpool = vgg16.avgpool
        vgg_classifier = vgg16.classifier

        # I've exposed the best/most interesting layers in my subjective opinion (mp5 is not that good though)
        self.layer_names = [
            "relu3_3",
            "relu4_1",
            "relu4_2",
            "relu4_3",
            "relu5_1",
            "relu5_2",
            "relu5_3",
            "mp5",
        ]

        # 31 layers in total for the VGG16
        self.conv1_1 = vgg_pretrained_features[0]
        self.relu1_1 = vgg_pretrained_features[1]
        self.conv1_2 = vgg_pretrained_features[2]
        self.relu1_2 = vgg_pretrained_features[3]
        self.max_pooling1 = vgg_pretrained_features[4]
        self.conv2_1 = vgg_pretrained_features[5]
        self.relu2_1 = vgg_pretrained_features[6]
        self.conv2_2 = vgg_pretrained_features[7]
        self.relu2_2 = vgg_pretrained_features[8]
        self.max_pooling2 = vgg_pretrained_features[9]
        self.conv3_1 = vgg_pretrained_features[10]
        self.relu3_1 = vgg_pretrained_features[11]
        self.conv3_2 = vgg_pretrained_features[12]
        self.relu3_2 = vgg_pretrained_features[13]
        self.conv3_3 = vgg_pretrained_features[14]
        self.relu3_3 = vgg_pretrained_features[15]
        self.max_pooling3 = vgg_pretrained_features[16]
        self.conv4_1 = vgg_pretrained_features[17]
        self.relu4_1 = vgg_pretrained_features[18]
        self.conv4_2 = vgg_pretrained_features[19]
        self.relu4_2 = vgg_pretrained_features[20]
        self.conv4_3 = vgg_pretrained_features[21]
        self.relu4_3 = vgg_pretrained_features[22]
        self.max_pooling4 = vgg_pretrained_features[23]
        self.conv5_1 = vgg_pretrained_features[24]
        self.relu5_1 = vgg_pretrained_features[25]
        self.conv5_2 = vgg_pretrained_features[26]
        self.relu5_2 = vgg_pretrained_features[27]
        self.conv5_3 = vgg_pretrained_features[28]
        self.relu5_3 = vgg_pretrained_features[29]
        self.max_pooling5 = vgg_pretrained_features[30]

        # AvgPool
        self.avgpool = vgg_avgpool

        # Classifier
        self.linear1 = vgg_classifier[0]
        self.relu1 = vgg_classifier[1]
        self.linear2 = vgg_classifier[3]
        self.relu2 = vgg_classifier[4]
        self.linear3 = vgg_classifier[6]

        # Turn off these because we'll be using a pretrained network
        # if we didn't do this PyTorch would be saving gradients and eating up precious memory!
        if not requires_grad:
            for param in self.parameters():
                param.requires_grad = False

    # Just expose every single layer during the forward pass
    def forward(self, x):
        x = self.conv1_1(x)
        conv1_1 = x
        x = self.relu1_1(x)
        relu1_1 = x
        x = self.conv1_2(x)
        conv1_2 = x
        x = self.relu1_2(x)
        relu1_2 = x
        x = self.max_pooling1(x)
        max_pooling1 = x
        x = self.conv2_1(x)
        conv2_1 = x
        x = self.relu2_1(x)
        relu2_1 = x
        x = self.conv2_2(x)
        conv2_2 = x
        x = self.relu2_2(x)
        relu2_2 = x
        x = self.max_pooling2(x)
        max_pooling2 = x
        x = self.conv3_1(x)
        conv3_1 = x
        x = self.relu3_1(x)
        relu3_1 = x
        x = self.conv3_2(x)
        conv3_2 = x
        x = self.relu3_2(x)
        relu3_2 = x
        x = self.conv3_3(x)
        conv3_3 = x
        x = self.relu3_3(x)
        relu3_3 = x
        x = self.max_pooling3(x)
        max_pooling3 = x
        x = self.conv4_1(x)
        conv4_1 = x
        x = self.relu4_1(x)
        relu4_1 = x
        x = self.conv4_2(x)
        conv4_2 = x
        x = self.relu4_2(x)
        relu4_2 = x
        x = self.conv4_3(x)
        conv4_3 = x
        x = self.relu4_3(x)
        relu4_3 = x
        x = self.max_pooling4(x)
        max_pooling4 = x
        x = self.conv5_1(x)
        conv5_1 = x
        x = self.relu5_1(x)
        relu5_1 = x
        x = self.conv5_2(x)
        conv5_2 = x
        x = self.relu5_2(x)
        relu5_2 = x
        x = self.conv5_3(x)
        conv5_3 = x
        x = self.relu5_3(x)
        relu5_3 = x
        x = self.max_pooling5(x)
        max_pooling5 = x

        # AvgPool - handle MPS device limitation
        if str(x.device).startswith("mps"):
            # Move to CPU for avgpool operation due to MPS limitation
            x_cpu = x.cpu()
            avgpool_result = self.avgpool(x_cpu)
            x = avgpool_result.to(x.device)
            avgpool = x
        else:
            x = self.avgpool(x)
            avgpool = x

        # Flatten
        x = torch.flatten(x, 1)

        # Classifier
        x = self.linear1(x)
        linear1 = x
        x = self.relu1(x)
        relu1 = x
        x = self.linear2(x)
        linear2 = x
        x = self.relu2(x)
        relu2 = x
        x = self.linear3(x)
        linear3 = x

        exposed_layers = {
            "conv1_1": conv1_1,
            "relu1_1": relu1_1,
            "conv1_2": conv1_2,
            "relu1_2": relu1_2,
            "max_pooling1": max_pooling1,
            "conv2_1": conv2_1,
            "relu2_1": relu2_1,
            "conv2_2": conv2_2,
            "relu2_2": relu2_2,
            "max_pooling2": max_pooling2,
            "conv3_1": conv3_1,
            "relu3_1": relu3_1,
            "conv3_2": conv3_2,
            "relu3_2": relu3_2,
            "conv3_3": conv3_3,
            "relu3_3": relu3_3,
            "max_pooling3": max_pooling3,
            "conv4_1": conv4_1,
            "relu4_1": relu4_1,
            "conv4_2": conv4_2,
            "relu4_2": relu4_2,
            "conv4_3": conv4_3,
            "relu4_3": relu4_3,
            "max_pooling4": max_pooling4,
            "conv5_1": conv5_1,
            "relu5_1": relu5_1,
            "conv5_2": conv5_2,
            "relu5_2": relu5_2,
            "conv5_3": conv5_3,
            "relu5_3": relu5_3,
            "max_pooling5": max_pooling5,
            "avgpool": avgpool,
            "linear1": linear1,
            "relu1": relu1,
            "linear2": linear2,
            "relu2": relu2,
            "linear3": linear3,
        }
        return exposed_layers
