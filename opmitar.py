'''
Copyright: 
file name: File name
Data: Do not edit
LastEditor: 
LastData: 
Describe: 
'''
import pandas as pd
    
# 加载两次预测结果
df1 = pd.read_csv('submit_75.325.csv')
df2 = pd.read_csv('submit_76.821.csv')

# 把两个结果合并到一起
df = pd.merge(df1, df2, on='uuid', suffixes=('_1', '_2'))

# 对比两次预测，如果结果相同，就使用这个结果，如果不同，就选择'NC'，因为在你的数据中，'NC'出现的次数更多。
df['final_label'] = df.apply(lambda row: row['label_1'] if row['label_1'] == row['label_2'] else 'MCI', axis=1)

# 保存最终结果
df[['uuid', 'final_label']].to_csv('final_result.csv', index=False)
