'''
Author: error: error: git config user.name & please set dead value or install git && error: git config user.email & please set dead value or install git & please set dead value or install git
Date: 2023-07-29 23:05:10
LastEditors: hanxiaoyue hanxiaoyue991.com
LastEditTime: 2023-07-31 11:22:17
FilePath: \ai夏令营\代码\pytorch\tongji.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''

import pandas as pd

# 读取csv文件
df = pd.read_csv('submit7.csv')  # 替换'filename.csv'为你的csv文件名

# 计算'MCI'和'NC'的数量
mci_count = df[df['label'] == 'MCI'].shape[0]
nc_count = df[df['label'] == 'NC'].shape[0]

print('MCI数量: ', mci_count)
print('NC数量: ', nc_count)
