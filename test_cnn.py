'''
Author: hanxiaoyue hanxiaoyue991.com
Date: 2023-07-24 23:13:53
LastEditors: hanxiaoyue hanxiaoyue991.com
LastEditTime: 2023-07-31 11:17:51
FilePath: \ai夏令营\代码\pytorch\test.py
Description: 
'''
from model import *
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm  # 新增
import os, sys, glob, argparse
from dataset import *
import albumentations as A

def predict(test_loader, model, criterion):
    model.eval()
    val_acc = 0.0
    
    test_pred = []
    with torch.no_grad():
        for i, (input, target) in tqdm(enumerate(test_loader)):  # 使用 tqdm 显示进度条
            input = input.cuda()
            target = target.cuda()

            output = model(input)
            test_pred.append(output.data.cpu().numpy())
            
            # 清除缓存
            torch.cuda.empty_cache()  # 新增
            
    return np.vstack(test_pred)

def test():
    test_path = glob.glob('./data/脑PET图像分析和疾病预测挑战赛数据集/Test/*')
    model = XunFeiNet()
    model = model.to('cuda')
    # 加载最好模型
    model.load_state_dict(torch.load('output/best_model_new2.pt'))

    test_loader = torch.utils.data.DataLoader(
        XunFeiDataset(test_path,
                A.Compose([
                # A.Resize(112,112),
                A.RandomCrop(100, 100),
            ])
        ), batch_size=8, shuffle=False, num_workers=0, pin_memory=False
    )

    pred = None

    criterion = nn.CrossEntropyLoss().cuda()

    for _ in range(30):
        if pred is None:
            pred = predict(test_loader, model, criterion)
        else:
            pred += predict(test_loader, model, criterion)
            
    submit = pd.DataFrame(
        {
            'uuid': [int(x.split('\\')[-1][:-4]) for x in test_path],
            'label': pred.argmax(1)
    })
    submit['label'] = submit['label'].map({1:'NC', 0: 'MCI'})
    submit = submit.sort_values(by='uuid')
    submit.to_csv('submit2.csv', index=None)

# test()
