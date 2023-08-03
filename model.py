'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2023-07-24 23:08:14
LastEditors: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
LastEditTime: 2023-07-30 21:22:08
FilePath: \pytorch\model.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
# Desc: Model definition for XunFei competition
import torch
import torch.nn as nn
import torchvision.models as models

class XunFeiNet(nn.Module):
    def __init__(self):
        super(XunFeiNet, self).__init__()
                
        # model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        # model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        model.conv1 = torch.nn.Conv2d(50, 64, kernel_size=(5, 5), stride=(2, 2), padding=(3, 3), bias=False)
        model.avgpool = nn.AdaptiveAvgPool2d(1)
        model.fc = nn.Linear(model.fc.in_features, 2)
        # model.fc = nn.Linear(2048, 2)
        # model.fc = nn.Linear(512, 2)
        
        self.resnet = model
        
    def forward(self, img):        
        out = self.resnet(img)
        return out
        


