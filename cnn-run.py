'''
Author: hanxiaoyue hanxiaoyue991.com
Date: 2023-07-25 21:41:56
LastEditors: hanxiaoyue hanxiaoyue991.com
LastEditTime: 2023-07-31 11:23:35
FilePath: \ai夏令营\代码\pytorch\cnn-run.py
创建者：寒晓月 ，版本：1.0.0: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
from multiprocessing import freeze_support
import albumentations as A
from dataset import *
from model import *
from train import *
from test_cnn import *
import copy  # 新增
torch.manual_seed(-1)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

train_path = glob.glob('./data/脑PET图像分析和疾病预测挑战赛数据集/Train/*/*')
test_path = glob.glob('./data/脑PET图像分析和疾病预测挑战赛数据集/Test/*')

np.random.shuffle(train_path)
np.random.shuffle(test_path)


train_loader = torch.utils.data.DataLoader(
    XunFeiDataset(train_path[:-10],
        A.Compose([
            A.RandomRotate90(),
            # A.Resize(112,112),
            A.RandomCrop(100, 100),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.5),
            A.VerticalFlip(p=0.5),  # 增加垂直翻转
            A.Rotate(limit=45, p=0.5)  # 增加随机旋转
        ])
    ), batch_size=16, shuffle=True, num_workers=0, pin_memory=False
)

val_loader = torch.utils.data.DataLoader(
    XunFeiDataset(train_path[-10:],
            A.Compose([
            # A.Resize(112,112),
            A.RandomCrop(100, 100),
        ])
    ), batch_size=16, shuffle=False, num_workers=0, pin_memory=False
)

test_loader = torch.utils.data.DataLoader(
    XunFeiDataset(test_path,
            A.Compose([
            # A.Resize(112,112),
            A.RandomCrop(100, 100),
        ])
    ), batch_size=12, shuffle=False, num_workers=0, pin_memory=False
)
# 加入学习率调整策略
model = XunFeiNet()
model = model.to('cuda')

# 在训练开始前，我们可以使用预训练的模型
# model = models.resnet18(pretrained=True)  # 例如使用预训练的resnet18模型

# 在模型的某些层中添加Dropout和Batch Normalization
# model.dropout = nn.Dropout(0.5)
# model.bn = nn.BatchNorm2d(num_features=10)

# 加载模型
model.load_state_dict(torch.load('output/best_model_new2.pt'))
criterion = nn.CrossEntropyLoss().cuda()
optimizer = torch.optim.AdamW(model.parameters(), 0.0001)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max', patience=3)

best_acc = 0.0  
best_model_wts = copy.deepcopy(model.state_dict())  
no_improve_epochs = 0  # 用于早停的计数器

for _  in range(20):
    train_loss = train(train_loader, model, criterion, optimizer)
    val_acc  = validate(val_loader, model, criterion)
    train_acc = validate(train_loader, model, criterion)
    # 调整学习率
    scheduler.step(val_acc)
    
    if val_acc > best_acc:
        best_acc = val_acc
        best_model_wts = copy.deepcopy(model.state_dict())
        no_improve_epochs = 0  # 如果有提升，计数器重置
    else:
        no_improve_epochs += 1  # 如果没有提升，计数器+1
        
    print("loss:",train_loss, "train_acc:",train_acc, "val_acc:",val_acc)
    
    # 如果在8个epoch中没有提升，提前停止训练
    if no_improve_epochs >= 8:
        print("Early stopping")
        break

# 加载最好模型
model.load_state_dict(best_model_wts)

os.makedirs('output', exist_ok=True)
torch.save(model.state_dict(), 'output/best_model_new3.pt')


pred = None


for _ in range(15):
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
submit.to_csv('submit3.csv', index=None)
