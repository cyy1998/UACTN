# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-

# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 17:21:12 2018

@author: shirhe-lyh
"""
import numpy as np
import random

import torch
import torch.nn as nn
from torchvision import models


class MVCNN(nn.Module):
    """definition."""

    def __init__(self, backbone,num_classes,pretrain=True,use_gpu=True):
        super(MVCNN, self).__init__()
        self._num_classes = num_classes
        self.use_gpu = use_gpu
        self.backbone = backbone

        """Build pre-trained resnet34 model for feature extraction of 3d model render images
        """

        if backbone == 'alexnet':
            self.model=models.alexnet(pretrained=pretrain)
            self.feature_size = self.model.classifier[6].in_features
            del self.model.classifier[6]
            #self.fc = nn.Linear(self.feature_size,self._num_classes)
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

    def forward(self, x):
        """
        Args:
            x: input a batch of image

        Returns:
            pooled_view: Extracted features, maxpooling of multiple features of 12 view_images of 3D model

            logits:  prediction tensors to be passed to the Cross Entropy Loss
        """
        """
        x = x.transpose(0, 1)
        rand = random.sample(range(1,12),6)
        x = x[rand]
        #print(type(x))
        #print(x.shape)

        view_pool = []

        for v in x:
            v = v.type(torch.cuda.FloatTensor)

            feature = self.model(v)
            feature = feature.view(feature.size(0), self.feature_size)  #
            feature = feature.detach().cpu().clone().numpy()
            view_pool.append(feature)

        #rand = random.randint(0, 12)
        #view_pool = view_pool[rand]
        view_pool = np.array(view_pool)
        view_pool1 = torch.from_numpy(view_pool)
        #print(view_pool1.size())
        #pooled_view = view_pool[0]
        pooled_view = torch.mean(view_pool1,dim = 0)
        #print(pooled_view.size())
        #for i in range(1, len(view_pool)):
            #pooled_view = torch.max(pooled_view, view_pool[i])  # max_pooling
        #print(pooled_view.size())
        pooled_view = pooled_view.type(torch.cuda.FloatTensor)
        #logits = self.fc(pooled_view)
        #pooled_view = self.layer1(pooled_view)
        #feature = self.layer2(feature)
        #feature = self.fc1(feature)"""

        x = x.transpose(0, 1)
        view_pool = []

        for v in x:
            v = v.type(torch.cuda.FloatTensor)
            feature = self.model(v)
            if self.backbone == 'alexnet':
                feature = feature.view(feature.size(0), self.feature_size)
                feature = feature.unsqueeze(0)
            else:
                feature = feature.view(feature.size(0), self.feature_size)  #
                feature = feature.unsqueeze(0)
            view_pool.append(feature)

        pooled_view = view_pool[0]
        for i in range(1, len(view_pool)):
            pooled_view = torch.cat((pooled_view, view_pool[i]),dim=0)  #
        pooled_view = torch.mean(pooled_view,dim=0)  #
        return pooled_view