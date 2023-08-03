'''
Author: hanxiaoyue hanxiaoyue991@gmail.com
Date: 2023-08-03 13:05:20
LastEditors: hanxiaoyue hanxiaoyue991@gmail.com
LastEditTime: 2023-08-03 13:48:12
FilePath: \代码\pytorch\nlp_first\all.py
Description: 

Copyright (c) 2023 by ${git_name_email}, All Rights Reserved. 
'''
import glob
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

train_data = pd.read_csv('./ChatGPT生成文本检测器公开数据-更新/train.csv')
test_data = pd.read_csv('./ChatGPT生成文本检测器公开数据-更新/test.csv')

train_data['content'] = train_data['content'].apply(lambda x: x[1:-1])
test_data['content'] = test_data['content'].apply(lambda x: x[1:-1])

def simple_feature(s):
    if len(s) == 0:
        s = '123 123'
    
    w = s.split()
    
    # 统计字符出现次数
    w_count = np.bincount(w)
    w_count = w_count[w_count != 0]
    
    return np.array([
        
        len(s), # 原始字符长度
        len(w), # 字符个数
        len(set(w)), # 不重复字符个数
        len(w) - len(set(w)), # 字符个数 - 不重复字符个数
        len(set(w)) / (len(w) + 1), # 不重复字符个数占比
        
        np.max(w_count), # 字符的频率的最大值
        np.min(w_count), # 字符的频率的最小值
        np.mean(w_count), # 字符的频率的平均值
        np.std(w_count), # 字符的频率的方差
        np.ptp(w_count), # 字符的频率的极差
    ])
    
    
train_feature = train_data['content'].iloc[:].apply(simple_feature)
test_feature = test_data['content'].iloc[:].apply(simple_feature)

train_feature = np.vstack(train_feature.values)
test_feature = np.vstack(test_feature.values)

#模型训练
m = LogisticRegression()
m.fit(train_feature, train_data['label'])

# 生成测试集提交结果
test_data['label'] = m.predict(test_feature)
test_data[['name', 'label']].to_csv('simple.csv', index=None)