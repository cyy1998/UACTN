# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 17:21:12 2018

@author: shirhe-lyh
"""

import torch
import torch.nn as nn
from torchvision import models
from torchvision import transforms


class SketchModel(nn.Module):
    """ definition."""

    def __init__(self,backbone, num_classes,pretrain = True,use_gpu=True):
        super(SketchModel, self).__init__()
        self._num_classes = num_classes
        self.use_gpu = use_gpu
        self.backbone = backbone

        """Build pre-trained resnet101 model for feature extraction of sketch images
        """
        if backbone == 'alexnet':
            self.model=models.alexnet(pretrained=pretrain)
            self.feature_size = self.model.classifier[6].in_features
            del self.model.classifier[6]
            self.fc = nn.Linear(self.feature_size,self._num_classes)
        elif backbone == 'vgg16':
            self.model=models.vgg16(pretrained=pretrain)
            self.feature_size = self.model.classifier[6].in_features
            del self.model.classifier[6]
        elif backbone == 'vgg19':
            self.model=models.vgg19(pretrained=pretrain)
            self.feature_size = self.model.classifier[6].in_features
            del self.model.classifier[6]
        elif backbone == 'resnet50':
            self.model=models.resnet50(pretrained=pretrain)
            self.feature_size = self.model.fc.in_features
            self.model = nn.Sequential(*list(self.model.children())[:-1])
            # del self.model.fc

    def forward(self, x):
        """
        Args:
            x: input a batch of image

        Returns:
            feature: Extracted features,feature matrix with shape (batch_size, feat_dim),which to be passed
                to the Center Loss

            logits:  prediction tensors to be passed to the Cross Entropy Loss
        """
        feature = self.model(x)
        if self.backbone == 'alexnet':
            feature = feature.view(-1, self.feature_size)
        else:
            feature = feature.view(-1, self.feature_size)
        #feature = self.layer1(feature)
        #feature = self.layer2(feature)
        #feature = self.layer3(feature)
        #logits = self.fc(feature)

        return feature




