# Author: Jie Chang <changjie@megvii.com>
# pylint: disable=not-callable, import-error,no-name-in-module
import torch
import torch.nn as nn
import torch.nn.functional as F


class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class Get_mu_logvar(nn.Module):
    def __init__(self, feature_size=4096):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1) * 1e-4)
        self.bata = nn.Parameter(torch.ones(1) * (-7))
        self.feature_size = feature_size


        self.mu_fc1 = nn.Sequential(nn.Linear(self.feature_size, 1024),nn.BatchNorm1d(1024,eps=2e-5),nn.ReLU())
        #self.mu_fc2 = nn.Sequential(nn.Linear(1024, 512),nn.BatchNorm1d(512,eps=2e-5),nn.ReLU())
        #self.mu_fc3 = nn.Sequential(nn.Linear(512, 256),nn.BatchNorm1d(256,eps=2e-5),nn.ReLU())
        self.mu_fc4 = nn.Sequential(nn.Linear(1024, 128), nn.BatchNorm1d(128, eps=2e-5))

        self.var_fc1 = nn.Sequential(nn.Linear(self.feature_size, 1024),nn.BatchNorm1d(1024,eps=2e-5),nn.ReLU())
        #self.var_fc2 = nn.Sequential(nn.Linear(1024, 512),nn.BatchNorm1d(512,eps=2e-5),nn.ReLU())
        #self.var_fc3 = nn.Sequential(nn.Linear(512, 256),nn.BatchNorm1d(256,eps=2e-5),nn.ReLU())
        self.var_fc4 = nn.Sequential(nn.Linear(1024, 128), nn.BatchNorm1d(128, eps=2e-5))


        """
        self.fc1 = nn.Linear(1024, embedding_size, bias=True)
        self.bn1 = nn.BatchNorm1d(embedding_size, eps=0.001)

        self.fc2 = nn.Linear(embedding_size, 256, bias=True)
        self.bn2 = nn.BatchNorm1d(256, eps=0.001, affine=True)
        
        self.fc3 = nn.Linear(256, 128, bias=False)
        self.bn3 = nn.BatchNorm1d(128, eps=0.001, affine=True)

        self.fc4 = nn.Linear(1024, embedding_size, bias=True)
        self.bn4 = nn.BatchNorm1d(embedding_size, eps=0.001)

        self.fc5 = nn.Linear(embedding_size, 256, bias=True)
        self.bn5 = nn.BatchNorm1d(256, eps=0.001)"""


    def forward(self, input):
        #x1 = self.getflatten(input)
        x1 = input
        x2 = input

        logvar = self.var_fc1(x1)
        #logvar = self.var_fc2(logvar)
        #logvar = self.var_fc3(logvar)
        logvar = self.var_fc4(logvar)
        logvar = self.gamma * logvar + self.bata
        logvar = torch.log(1e-6 + torch.exp(logvar))


        mu = self.mu_fc1(x2)
        #mu = self.mu_fc2(mu)
        #mu = self.mu_fc3(mu)
        mu = self.mu_fc4(mu)
        mu = nn.functional.normalize(mu, dim=1)


        return mu,logvar
