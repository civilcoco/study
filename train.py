'''
Author: hanxiaoyue hanxiaoyue991.com
Date: 2023-07-24 23:08:50
LastEditors: hanxiaoyue hanxiaoyue991.com
LastEditTime: 2023-07-31 11:18:34
FilePath: \ai夏令营\代码\pytorch\train.py
Description: 

Copyright (c) 2023 by ${git_name_email}, All Rights Reserved. 
'''


import torch
from model import *
'''
description: 
param {*} train_loader
param {*} model
param {*} criterion
param {*} optimizer
return {*}
'''
def train(train_loader, model, criterion, optimizer):
    model.train()
    train_loss = 0.0
    for i, (input, target) in enumerate(train_loader):
        # input = input.float().cuda(non_blocking=True)  # 注意这里的改动
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)
        
        output = model(input)
        loss = criterion(output, target.long())

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if i % 20 == 0:
            print("loss:",loss.item())
            
        train_loss += loss.item()
    
    return train_loss/len(train_loader)
            
# def validate(val_loader, model, criterion):
#     model.eval()
#     val_acc = 0.0
    
#     with torch.no_grad():
#         for i, (input, target) in enumerate(val_loader):
#             input = input.float().cuda()  # 注意这里的改动
#             # input = input.cuda()
#             target = target.cuda()

#             # compute output
#             output = model(input)
#             loss = criterion(output, target.long())
            
#             val_acc += (output.argmax(1) == target).sum().item()
            
#     return val_acc / len(val_loader.dataset)

def validate(val_loader, model, criterion):
    model.eval()
    true_positives = 0
    false_positives = 0
    false_negatives = 0
    
    with torch.no_grad():
        for i, (input, target) in enumerate(val_loader):
            # input = input.float().cuda()  # 注意这里的改动
            input = input.cuda()  # 注意这里的改动
            target = target.cuda()

            # compute output
            output = model(input)
            loss = criterion(output, target.long())
            
            pred = output.argmax(1)
            true_positives += (pred == target).sum().item()
            false_positives += (pred > target).sum().item()
            false_negatives += (pred < target).sum().item()

    f1_score = calculate_f1_score(true_positives, false_positives, false_negatives)

    return f1_score

def calculate_f1_score(tp, fp, fn):
    #先计算精确率和召回率，进而计算F1_score
    precision = tp / (tp + fp) if tp != 0  else 0  # 精确率
    recall = tp / (tp + fn) if tp != 0 else 0  # 召回率
    f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0

    return f1_score


