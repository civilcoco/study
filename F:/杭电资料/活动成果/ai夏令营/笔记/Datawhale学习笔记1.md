# Datawhale学习笔记》机器学习在图像分类中的应用--讯飞PET图分类

## 1. 数据准备

机器学习的第一步通常是数据准备。在图像分类任务中，数据通常包括图像和标签。我们需要收集大量的图像数据，并为每个图像提供正确的类别标签。数据可以从各种来源获取，如公开数据集，或者通过自己收集和标注数据。比赛官方讯飞已经为我们提供了一个zip文件，里面有很多医学的脑图文件，用.inn结尾。

![1.png](https://s2.loli.net/2023/07/21/BxHzWC4eswOLh2G.png)

## 2. 数据预处理

数据预处理是将原始数据转换为更适合机器学习模型处理的形式的过程。这可能包括缩放或标准化图像，处理缺失值，以及将分类标签转换为机器学习算法可以理解的形式（例如，one-hot编码）。在处理图像数据时，常用的预处理步骤包括调整图像大小，归一化像素值，以及数据增强（通过应用随机变换来创建图像的修改版本）。

![2.png](https://s2.loli.net/2023/07/21/zHSK54gsGwhX39Z.png)



![3.png](https://s2.loli.net/2023/07/21/5PUB3swmJWHnYvN.png)

```python
import os
import numpy as np
import nibabel as nib
import matplotlib.pyplot as plt
from PIL import Image


def process_img(input_folder, output_folder):
    # 确保输出文件夹存在，如果不存在则创建
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # 遍历输入文件夹中的每个文件
    for filename in os.listdir(input_folder):
        # 检查文件扩展名是否为NIfTI
        if filename.endswith('.nii') or filename.endswith('.nii.gz'):
            # 构建输入和输出文件路径
            input_file = os.path.join(input_folder, filename)
            output_file_prefix = os.path.join(output_folder, filename.replace('.nii.gz', '').replace('.nii', ''))

            # 加载NIfTI文件
            nii_image = nib.load(input_file)

            # 提取图像数据
            image_data = nii_image.get_fdata()

            # 获取图像数据的形状和维度
            image_shape = image_data.shape
            num_slices = image_shape[-2]  # 倒数第二个维度表示切片数

            # 迭代处理每个切片和时间点
            t = 0  # 只有一个时间点
            for s in range(num_slices):
                # 提取当前切片和时间点的数据
                slice_data = image_data[..., s, t]  # 使用[..., s, t]选择特定切片和时间点
                # 生成保存的文件名
                file_name = f"_{str(s + 1)}.jpg"
                output_file = output_file_prefix + file_name
                # 保存为PNG文件
                plt.imsave(output_file, slice_data, cmap='gray')



if __name__ == '__main__':
    # 定义输入文件夹路径和输出文件夹路径
    input_folder_1 = 'PET/Train/MCI'
    input_folder_2 = 'PET/Train/NC'
    input_folder_3 = 'PET/Test'

    output_folder_1 = '/home/aistudio/PaddleClas/dataset/PET-dataset/MCI'
    output_folder_2 = '/home/aistudio/PaddleClas/dataset/PET-dataset/NC'
    output_folder_3 = '/home/aistudio/PaddleClas/dataset/Test'

    process_img(input_folder_1, output_folder_1)
    process_img(input_folder_2, output_folder_2)
    process_img(input_folder_3, output_folder_3)






    
```

这个脚本的主要目的是将NIfTI格式的PET图像转换为JPG格式的图片。NIfTI是一种用于存储医学图像数据（如MRI、CT、PET等）的文件格式，它可以存储三维或四维的图像数据。在这个脚本中，将每个NIfTI文件中的每个切片都转换为一个JPG图片。

以下是这个脚本每一步的详细解释：

1. **定义转换函数**：`process_img`函数是主要的转换函数，它接受输入文件夹和输出文件夹作为参数。输入文件夹是包含NIfTI文件的文件夹，输出文件夹是用于保存转换后的JPG图片的文件夹。

2. **确保输出文件夹存在**：如果输出文件夹不存在，就使用`os.makedirs`函数创建它。

3. **遍历输入文件夹中的每个文件**：对于每个文件，首先检查文件扩展名是否为NIfTI。如果是，就构建输入文件的路径和输出文件的前缀。

4. **加载NIfTI文件**：使用`nibabel`库的`load`函数加载NIfTI文件。

5. **提取图像数据**：使用`get_fdata`方法提取图像数据。

6. **获取图像数据的形状和维度**：提取图像数据的形状和切片数。

7. **迭代处理每个切片**：对于每个切片，提取当前切片的数据，生成保存的文件名，然后使用`matplotlib.pyplot.imsave`函数保存为JPG文件。`imsave`函数需要一个文件名，一个二维数组（表示图像数据），和一个颜色映射（`cmap='gray'`表示灰度图像）。

8. **调用转换函数**：最后，定义输入文件夹和输出文件夹的路径，然后调用`process_img`函数进行转换。

总的来说，这个脚本的功能是将NIfTI格式的PET图像转换为JPG格式的图片。这是一个数据预处理步骤，可以将复杂的医学图像数据转换为通用的图片格式，方便后续的机器学习或深度学习模型处理。



```python
# %cd /home/aistudio/PaddleClas/dataset

# 数据集切分
import os
import random

data_folder = 'PET-dataset'  # 数据文件夹路径
train_ratio = 0.8  # 训练集比例
val_ratio = 0.2  # 验证集比例

# 获取MCI类别文件夹路径
mci_folder = os.path.join(data_folder, 'MCI')

# 获取NC类别文件夹路径
nc_folder = os.path.join(data_folder, 'NC')

# 获取MCI类别文件列表
mci_files = [os.path.join(mci_folder, file) for file in os.listdir(mci_folder)]

# 获取NC类别文件列表
nc_files = [os.path.join(nc_folder, file) for file in os.listdir(nc_folder)]

# 获取标签列表
labels = ['MCI', 'NC']

# 创建labels.txt文件
with open('labels.txt', 'w') as f:
    for label in labels:
        f.write(label + '\n')

# 随机打乱文件列表
random.shuffle(mci_files)
random.shuffle(nc_files)

# 计算划分索引
mci_train_split = int(train_ratio * len(mci_files))
mci_val_split = int((train_ratio + val_ratio) * len(mci_files))
nc_train_split = int(train_ratio * len(nc_files))
nc_val_split = int((train_ratio + val_ratio) * len(nc_files))

# 划分文件列表
train_files = mci_files[:mci_train_split] + nc_files[:nc_train_split]
val_files = mci_files[mci_train_split:mci_val_split] + nc_files[nc_train_split:nc_val_split]
test_files = mci_files[mci_val_split:] + nc_files[nc_val_split:]

# 创建train_list.txt文件
with open('train_list.txt', 'w') as f:
    for file in train_files:
        label = '0' if 'MCI' in file else '1'  # 添加对应的标签数字
        f.write(file + ' ' + label + '\n')

# 创建val_list.txt文件
with open('val_list.txt', 'w') as f:
    for file in val_files:
        label = '0' if 'MCI' in file else '1'  # 添加对应的标签数字
        f.write(file + ' ' + label + '\n')





```

这段代码的主要目的是将整个数据集划分为训练集、验证集和测试集，并为每个数据集创建一个文件列表，这个文件列表包含了每个样本的文件路径和类别标签。

以下是这段代码每一步的详细解释：

1. **设置文件夹和数据集比例**：设置数据文件夹的路径和训练集、验证集的比例。剩余的部分将作为测试集。
2. **获取MCI和NC类别文件夹路径**：`os.path.join`函数用于将多个路径组合成一个完整的路径。
3. **获取MCI和NC类别文件列表**：`os.listdir`函数返回指定文件夹中所有文件的列表。`os.path.join`函数将文件夹路径和文件名组合成完整的文件路径。
4. **创建labels.txt文件**：这个文件包含了所有类别标签的列表。
5. **随机打乱文件列表**：使用`random.shuffle`函数随机打乱MCI和NC类别的文件列表。
6. **计算划分索引**：根据训练集和验证集的比例计算划分索引。
7. **划分文件列表**：根据划分索引将文件列表划分为训练集、验证集和测试集。
8. **创建train_list.txt和val_list.txt文件**：这两个文件包含了训练集和验证集的文件路径和类别标签。在每一行中，文件路径和类别标签之间用一个空格隔开。类别标签是一个数字，0代表MCI，1代表NC。

总的来说，这段代码的功能是将数据集划分为训练集、验证集和测试集，并创建包含每个数据集文件路径和类别标签的文件列表，这些列表可以用于后续的模型训练和验证。

## 3. 特征提取

特征提取是从原始数据中提取出对任务有用的信息的过程。在图像分类任务中，一种常用的特征提取技术是使用卷积神经网络（Convolutional Neural Networks, CNN）。**CNN可以自动从图像中学习和提取有用的特征，这使得我们无需手动设计特征**。

如下图中我们选取了一些特定的值作为一个图像的特征值，是从图像中抽取并且归纳出来的

![4.png](https://s2.loli.net/2023/07/21/isR1fhd7blOHDVG.png)



## 4. 模型训练

模型训练是使用算法从数据中学习模型参数的过程。这通常涉及定义一个损失函数（衡量模型预测与真实标签之间的差异），然后使用优化算法（如梯度下降）来最小化损失。在训练过程中，模型会尝试学习到从**输入特征**（图像）到**输出**（类别标签）的映射。

以下是一个通过逻辑回归进行训练的过程

![5.png](https://s2.loli.net/2023/07/21/EqQcfpBVOsLXRAz.png)

在`scikit-learn`（sklearn）中，除了逻辑回归（Logistic Regression）之外，还有许多其他的机器学习模型可以用于分类任务中，以下是一些常用于分类任务的机器学习模型：

1. 支持向量机（Support Vector Machines，SVM）：用于二分类和多分类问题，通过构建一个超平面来区分不同类别的样本。
2. 决策树（Decision Trees）：适用于二分类和多分类问题，通过对特征空间进行划分来分类样本。
3. 随机森林（Random Forests）：基于多个决策树的集成算法，用于二分类和多分类问题，提高了模型的泛化能力。
4. K最近邻算法（K-Nearest Neighbors，KNN）：根据最近邻样本的类别来分类新样本，适用于二分类和多分类问题。
5. 朴素贝叶斯（Naive Bayes）：基于贝叶斯定理的分类方法，适用于文本分类等问题。
6. 多层感知器（Multi-layer Perceptrons，MLP）：一种人工神经网络，用于解决复杂的分类问题。



以下是关于使用`scikit-learn`库进行分类任务的示例代码：

1. **支持向量机（SVM）**
```python
from sklearn import svm
clf = svm.SVC(gamma='scale')
clf.fit(X_train, y_train)
```

2. **决策树（Decision Trees）**
```python
from sklearn import tree
clf = tree.DecisionTreeClassifier()
clf.fit(X_train, y_train)
```

3. **随机森林（Random Forests）**
```python
from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(n_estimators=100)
clf.fit(X_train, y_train)
```

4. **K最近邻（KNN）**
```python
from sklearn.neighbors import KNeighborsClassifier
clf = KNeighborsClassifier(n_neighbors=3)
clf.fit(X_train, y_train)
```

5. **朴素贝叶斯（Naive Bayes）**
```python
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()
clf.fit(X_train, y_train)
```

6. **多层感知器（MLP）**
```python
from sklearn.neural_network import MLPClassifier
clf = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2))
clf.fit(X_train, y_train)
```

对于所有上述的模型，`X_train`和`y_train`分别代表训练数据的特征和标签。每个模型的参数可能需要根据你的具体任务和数据进行调整。例如，对于KNN，你可能需要选择一个合适的邻居数量；对于随机森林，你可能需要选择一个合适的估计器数量等。

请注意，上述模型除了卷积神经网络外，都不是专为图像数据设计的。处理图像数据时，通常首先要将图像数据转化为一维向量（即所谓的“flattening”），这可能会导致一些空间信息的丢失。为了更好地处理图像数据，通常会使用深度学习的方法，如卷积神经网络（CNN）。在Python中，可以使用`keras`或`PyTorch`这样的深度学习库来创建和训练CNN。

7. **卷积神经网络（CNN）**
```python
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

model = Sequential()
model.add(Conv2D(32, (3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dense(1, activation='sigmoid'))

model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=10, batch_size=32)
```
在这个CNN模型中，我们首先添加了一个卷积层和一个最大池化层，用于从图像中提取特征。然后我们添加了一个展平层（Flatten），将2D特征图转换为1D向量。最后，我们添加了两个全连接层（Dense），用于进行分类。模型使用Adam优化器和二元交叉熵损失函数进行编译，然后使用训练数据进行训练。

## 5. 超参数调整

超参数是在开始学习过程之前设置的参数，而不是通过训练得到的参数。这可能包括学习率，正则化参数，以及网络架构（如层数，每层的神经元数量）。超参数调整通常通过搜索（如网格搜索或随机搜索）来进行，目标是找到最优化模型性能的参数设置。

比如batch_size和numworker的大小，与环境的显存和模型的泛化性紧密挂钩

学习率则和图像拟合程度挂钩，需要调成一个合适的值，否则会导致精度始终无法趋向一个极值点。就像小球在一个碗里滚动，如果摩擦力很小，那么就会一直在周围摆动，很难停在碗底。

当然，除这些之外还有很多超参数需要我们在后续调整的时候完善比如正则化参数，以及网络架构（如层数，每层的神经元数量）等等。

![7.png](https://s2.loli.net/2023/07/21/UgKq5WtQodvukXI.png)

## 6. 预测

一旦模型被训练，它就可以用于预测新的未标记的图像的类别。这通常涉及将图像通过模型，然后选择具有最高预测概率的类别作为输出。

下图是个例子，用模型对单个样例的多次预测结果取出现最多的结果进行分类

![Image 5](https://s2.loli.net/2023/07/21/2Bb3y7EPUDdhMYr.png)

## 7. 评估

模型评估是衡量模型性能的过程。这通常涉及使用一组未参与训练的测试数据，然后计算模型预测与真实标签之间的差异。在图像分类任务中，常用的评估指标包括准确率，精度，召回率，和F1分数。

这个过程是对训练好的模型的一个打分，需要计算一些指标来衡量一个模型的质量和性能，比如精确率和召回率。

![8.png](https://s2.loli.net/2023/07/21/N3iAezVwKxtHCmP.png)

```python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import time
import platform
import paddle

from ppcls.utils.misc import AverageMeter
from ppcls.utils import logger


def classification_eval(engine, epoch_id=0):
    if hasattr(engine.eval_metric_func, "reset"):
        engine.eval_metric_func.reset()
    output_info = dict()
    time_info = {
        "batch_cost": AverageMeter(
            "batch_cost", '.5f', postfix=" s,"),
        "reader_cost": AverageMeter(
            "reader_cost", ".5f", postfix=" s,"),
    }
    print_batch_step = engine.config["Global"]["print_batch_step"]

    tic = time.time()
    accum_samples = 0
    total_samples = len(
        engine.eval_dataloader.
        dataset) if not engine.use_dali else engine.eval_dataloader.size
    max_iter = len(engine.eval_dataloader) - 1 if platform.system(
    ) == "Windows" else len(engine.eval_dataloader)
    for iter_id, batch in enumerate(engine.eval_dataloader):
        if iter_id >= max_iter:
            break
        if iter_id == 5:
            for key in time_info:
                time_info[key].reset()
        if engine.use_dali:
            batch = [
                paddle.to_tensor(batch[0]['data']),
                paddle.to_tensor(batch[0]['label'])
            ]
        time_info["reader_cost"].update(time.time() - tic)
        batch_size = batch[0].shape[0]
        batch[0] = paddle.to_tensor(batch[0])
        if not engine.config["Global"].get("use_multilabel", False):
            batch[1] = batch[1].reshape([-1, 1]).astype("int64")

        # image input
        if engine.amp and engine.amp_eval:
            with paddle.amp.auto_cast(
                    custom_black_list={
                        "flatten_contiguous_range", "greater_than"
                    },
                    level=engine.amp_level):
                out = engine.model(batch[0])
        else:
            out = engine.model(batch[0])

        # just for DistributedBatchSampler issue: repeat sampling
        current_samples = batch_size * paddle.distributed.get_world_size()
        accum_samples += current_samples

        if isinstance(out, dict) and "Student" in out:
            out = out["Student"]
        if isinstance(out, dict) and "logits" in out:
            out = out["logits"]

        # gather Tensor when distributed
        if paddle.distributed.get_world_size() > 1:
            label_list = []

            paddle.distributed.all_gather(label_list, batch[1])
            labels = paddle.concat(label_list, 0)

            if isinstance(out, list):
                preds = []
                for x in out:
                    pred_list = []
                    paddle.distributed.all_gather(pred_list, x)
                    pred_x = paddle.concat(pred_list, 0)
                    preds.append(pred_x)
            else:
                pred_list = []
                paddle.distributed.all_gather(pred_list, out)
                preds = paddle.concat(pred_list, 0)

            if accum_samples > total_samples and not engine.use_dali:
                preds = preds[:total_samples + current_samples - accum_samples]
                labels = labels[:total_samples + current_samples -
                                accum_samples]
                current_samples = total_samples + current_samples - accum_samples
        else:
            labels = batch[1]
            preds = out

        # calc loss
        if engine.eval_loss_func is not None:
            if engine.amp and engine.amp_eval:
                with paddle.amp.auto_cast(
                        custom_black_list={
                            "flatten_contiguous_range", "greater_than"
                        },
                        level=engine.amp_level):
                    loss_dict = engine.eval_loss_func(preds, labels)
            else:
                loss_dict = engine.eval_loss_func(preds, labels)

            for key in loss_dict:
                if key not in output_info:
                    output_info[key] = AverageMeter(key, '7.5f')
                output_info[key].update(loss_dict[key].numpy()[0],
                                        current_samples)

        #  calc metric
        if engine.eval_metric_func is not None:
            engine.eval_metric_func(preds, labels)
        time_info["batch_cost"].update(time.time() - tic)

        if iter_id % print_batch_step == 0:
            time_msg = "s, ".join([
                "{}: {:.5f}".format(key, time_info[key].avg)
                for key in time_info
            ])

            ips_msg = "ips: {:.5f} images/sec".format(
                batch_size / time_info["batch_cost"].avg)

            if "ATTRMetric" in engine.config["Metric"]["Eval"][0]:
                metric_msg = ""
            else:
                metric_msg = ", ".join([
                    "{}: {:.5f}".format(key, output_info[key].val)
                    for key in output_info
                ])
                metric_msg += ", {}".format(engine.eval_metric_func.avg_info)
            logger.info("[Eval][Epoch {}][Iter: {}/{}]{}, {}, {}".format(
                epoch_id, iter_id,
                len(engine.eval_dataloader), metric_msg, time_msg, ips_msg))

        tic = time.time()
    if engine.use_dali:
        engine.eval_dataloader.reset()

    if "ATTRMetric" in engine.config["Metric"]["Eval"][0]:
        metric_msg = ", ".join([
            "evalres: ma: {:.5f} label_f1: {:.5f} label_pos_recall: {:.5f} label_neg_recall: {:.5f} instance_f1: {:.5f} instance_acc: {:.5f} instance_prec: {:.5f} instance_recall: {:.5f}".
            format(*engine.eval_metric_func.attr_res())
        ])
        logger.info("[Eval][Epoch {}][Avg]{}".format(epoch_id, metric_msg))

        # do not try to save best eval.model
        if engine.eval_metric_func is None:
            return -1
        # return 1st metric in the dict
        return engine.eval_metric_func.attr_res()[0]
    else:
        metric_msg = ", ".join([
            "{}: {:.5f}".format(key, output_info[key].avg)
            for key in output_info
        ])
        metric_msg += ", {}".format(engine.eval_metric_func.avg_info)
        logger.info("[Eval][Epoch {}][Avg]{}".format(epoch_id, metric_msg))

        # do not try to save best eval.model
        if engine.eval_metric_func is None:
            return -1
        # return 1st metric in the dict
        return engine.eval_metric_func.avg
```

接下来我来解释上方的一段评估类--分类任务，

这段代码实现了图像分类任务的模型评估过程。它使用了一个模型引擎（`engine`）来在验证集上进行预测，并计算模型的损失和评估指标。以下是一些详细的解释：

- `classification_eval`：这是主要的评估函数。它首先初始化一些用于计时和记录输出信息的对象，然后遍历验证数据集，对每个样本进行预测，并收集预测结果。对于每个样本，它都会计算模型的输出、损失和评估指标。在所有样本都处理完毕后，它会计算并返回平均损失和评估指标。
- 在每个迭代中，它首先读取一个批次的数据，并记录读取数据所花费的时间。然后，它将数据转换为Tensor格式，并用模型进行预测。预测结果可能是一个字典，包含了模型的多个输出，也可能只是一个Tensor。
- 如果使用了分布式训练，那么它会使用`paddle.distributed.all_gather`函数来收集所有进程的预测结果和标签。然后，它会根据预测结果和标签来计算损失和评估指标。
- 在计算损失时，如果指定了损失函数（`engine.eval_loss_func`），那么它会使用这个函数来计算损失。计算得到的损失会被添加到`output_info`字典中，以便后续打印和分析。
- 在计算评估指标时，如果指定了评估函数（`engine.eval_metric_func`），那么它会使用这个函数来计算评估指标。
- 最后，它会更新批次时间，并在每隔一定的迭代次数后，打印出当前的损失、评估指标和时间信息。

总的来说，这段代码的主要目的是在验证集上评估模型的性能，包括损失和评估指标。

```python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import platform
from typing import Optional

import numpy as np
import paddle
from ppcls.utils import logger


def retrieval_eval(engine, epoch_id=0):
    engine.model.eval()
    # step1. build gallery
    if engine.gallery_query_dataloader is not None:
        gallery_feas, gallery_img_id, gallery_unique_id = cal_feature(
            engine, name='gallery_query')
        query_feas, query_img_id, query_query_id = gallery_feas, gallery_img_id, gallery_unique_id
    else:
        gallery_feas, gallery_img_id, gallery_unique_id = cal_feature(
            engine, name='gallery')
        query_feas, query_img_id, query_query_id = cal_feature(
            engine, name='query')

    # step2. do evaluation
    sim_block_size = engine.config["Global"].get("sim_block_size", 64)
    sections = [sim_block_size] * (len(query_feas) // sim_block_size)
    if len(query_feas) % sim_block_size:
        sections.append(len(query_feas) % sim_block_size)
    fea_blocks = paddle.split(query_feas, num_or_sections=sections)
    if query_query_id is not None:
        query_id_blocks = paddle.split(
            query_query_id, num_or_sections=sections)
    image_id_blocks = paddle.split(query_img_id, num_or_sections=sections)
    metric_key = None

    if engine.eval_loss_func is None:
        metric_dict = {metric_key: 0.}
    else:
        reranking_flag = engine.config['Global'].get('re_ranking', False)
        logger.info(f"re_ranking={reranking_flag}")
        metric_dict = dict()
        if reranking_flag:
            # set the order from small to large
            for i in range(len(engine.eval_metric_func.metric_func_list)):
                if hasattr(engine.eval_metric_func.metric_func_list[i], 'descending') \
                        and engine.eval_metric_func.metric_func_list[i].descending is True:
                    engine.eval_metric_func.metric_func_list[
                        i].descending = False
                    logger.warning(
                        f"re_ranking=True,{engine.eval_metric_func.metric_func_list[i].__class__.__name__}.descending has been set to False"
                    )

            # compute distance matrix(The smaller the value, the more similar)
            distmat = re_ranking(
                query_feas, gallery_feas, k1=20, k2=6, lambda_value=0.3)

            # compute keep mask
            query_id_mask = (query_query_id != gallery_unique_id.t())
            image_id_mask = (query_img_id != gallery_img_id.t())
            keep_mask = paddle.logical_or(query_id_mask, image_id_mask)

            # set inf(1e9) distance to those exist in gallery
            distmat = distmat * keep_mask.astype("float32")
            inf_mat = (paddle.logical_not(keep_mask).astype("float32")) * 1e20
            distmat = distmat + inf_mat

            # compute metric
            metric_tmp = engine.eval_metric_func(distmat, query_img_id,
                                                 gallery_img_id, keep_mask)
            for key in metric_tmp:
                metric_dict[key] = metric_tmp[key]
        else:
            for block_idx, block_fea in enumerate(fea_blocks):
                similarity_matrix = paddle.matmul(
                    block_fea, gallery_feas, transpose_y=True)  # [n,m]
                if query_query_id is not None:
                    query_id_block = query_id_blocks[block_idx]
                    query_id_mask = (query_id_block != gallery_unique_id.t())

                    image_id_block = image_id_blocks[block_idx]
                    image_id_mask = (image_id_block != gallery_img_id.t())

                    keep_mask = paddle.logical_or(query_id_mask, image_id_mask)
                    similarity_matrix = similarity_matrix * keep_mask.astype(
                        "float32")
                else:
                    keep_mask = None

                metric_tmp = engine.eval_metric_func(
                    similarity_matrix, image_id_blocks[block_idx],
                    gallery_img_id, keep_mask)

                for key in metric_tmp:
                    if key not in metric_dict:
                        metric_dict[key] = metric_tmp[key] * block_fea.shape[
                            0] / len(query_feas)
                    else:
                        metric_dict[key] += metric_tmp[key] * block_fea.shape[
                            0] / len(query_feas)

    metric_info_list = []
    for key in metric_dict:
        if metric_key is None:
            metric_key = key
        metric_info_list.append("{}: {:.5f}".format(key, metric_dict[key]))
    metric_msg = ", ".join(metric_info_list)
    logger.info("[Eval][Epoch {}][Avg]{}".format(epoch_id, metric_msg))

    return metric_dict[metric_key]


def cal_feature(engine, name='gallery'):
    has_unique_id = False
    all_unique_id = None

    if name == 'gallery':
        dataloader = engine.gallery_dataloader
    elif name == 'query':
        dataloader = engine.query_dataloader
    elif name == 'gallery_query':
        dataloader = engine.gallery_query_dataloader
    else:
        raise RuntimeError("Only support gallery or query dataset")

    batch_feas_list = []
    img_id_list = []
    unique_id_list = []
    max_iter = len(dataloader) - 1 if platform.system() == "Windows" else len(
        dataloader)
    for idx, batch in enumerate(dataloader):  # load is very time-consuming
        if idx >= max_iter:
            break
        if idx % engine.config["Global"]["print_batch_step"] == 0:
            logger.info(
                f"{name} feature calculation process: [{idx}/{len(dataloader)}]"
            )
        if engine.use_dali:
            batch = [
                paddle.to_tensor(batch[0]['data']),
                paddle.to_tensor(batch[0]['label'])
            ]
        batch = [paddle.to_tensor(x) for x in batch]
        batch[1] = batch[1].reshape([-1, 1]).astype("int64")
        if len(batch) == 3:
            has_unique_id = True
            batch[2] = batch[2].reshape([-1, 1]).astype("int64")
        if engine.amp and engine.amp_eval:
            with paddle.amp.auto_cast(
                    custom_black_list={
                        "flatten_contiguous_range", "greater_than"
                    },
                    level=engine.amp_level):
                out = engine.model(batch[0], batch[1])
        else:
            out = engine.model(batch[0], batch[1])
        if "Student" in out:
            out = out["Student"]

        # get features
        if engine.config["Global"].get("retrieval_feature_from",
                                       "features") == "features":
            # use neck's output as features
            batch_feas = out["features"]
        else:
            # use backbone's output as features
            batch_feas = out["backbone"]

        # do norm
        if engine.config["Global"].get("feature_normalize", True):
            feas_norm = paddle.sqrt(
                paddle.sum(paddle.square(batch_feas), axis=1, keepdim=True))
            batch_feas = paddle.divide(batch_feas, feas_norm)

        # do binarize
        if engine.config["Global"].get("feature_binarize") == "round":
            batch_feas = paddle.round(batch_feas).astype("float32") * 2.0 - 1.0

        if engine.config["Global"].get("feature_binarize") == "sign":
            batch_feas = paddle.sign(batch_feas).astype("float32")

        if paddle.distributed.get_world_size() > 1:
            batch_feas_gather = []
            img_id_gather = []
            unique_id_gather = []
            paddle.distributed.all_gather(batch_feas_gather, batch_feas)
            paddle.distributed.all_gather(img_id_gather, batch[1])
            batch_feas_list.append(paddle.concat(batch_feas_gather))
            img_id_list.append(paddle.concat(img_id_gather))
            if has_unique_id:
                paddle.distributed.all_gather(unique_id_gather, batch[2])
                unique_id_list.append(paddle.concat(unique_id_gather))
        else:
            batch_feas_list.append(batch_feas)
            img_id_list.append(batch[1])
            if has_unique_id:
                unique_id_list.append(batch[2])

    if engine.use_dali:
        dataloader.reset()

    all_feas = paddle.concat(batch_feas_list)
    all_img_id = paddle.concat(img_id_list)
    if has_unique_id:
        all_unique_id = paddle.concat(unique_id_list)

    # just for DistributedBatchSampler issue: repeat sampling
    total_samples = len(
        dataloader.dataset) if not engine.use_dali else dataloader.size
    all_feas = all_feas[:total_samples]
    all_img_id = all_img_id[:total_samples]
    if has_unique_id:
        all_unique_id = all_unique_id[:total_samples]

    logger.info("Build {} done, all feat shape: {}, begin to eval..".format(
        name, all_feas.shape))
    return all_feas, all_img_id, all_unique_id


def re_ranking(query_feas: paddle.Tensor,
               gallery_feas: paddle.Tensor,
               k1: int=20,
               k2: int=6,
               lambda_value: int=0.5,
               local_distmat: Optional[np.ndarray]=None,
               only_local: bool=False) -> paddle.Tensor:
    """re-ranking, most computed with numpy

    code heavily based on
    https://github.com/michuanhaohao/reid-strong-baseline/blob/3da7e6f03164a92e696cb6da059b1cd771b0346d/utils/reid_metric.py

    Args:
        query_feas (paddle.Tensor): query features, [num_query, num_features]
        gallery_feas (paddle.Tensor): gallery features, [num_gallery, num_features]
        k1 (int, optional): k1. Defaults to 20.
        k2 (int, optional): k2. Defaults to 6.
        lambda_value (int, optional): lambda. Defaults to 0.5.
        local_distmat (Optional[np.ndarray], optional): local_distmat. Defaults to None.
        only_local (bool, optional): only_local. Defaults to False.

    Returns:
        paddle.Tensor: final_dist matrix after re-ranking, [num_query, num_gallery]
    """
    query_num = query_feas.shape[0]
    all_num = query_num + gallery_feas.shape[0]
    if only_local:
        original_dist = local_distmat
    else:
        feat = paddle.concat([query_feas, gallery_feas])
        logger.info('using GPU to compute original distance')

        # L2 distance
        distmat = paddle.pow(feat, 2).sum(axis=1, keepdim=True).expand([all_num, all_num]) + \
            paddle.pow(feat, 2).sum(axis=1, keepdim=True).expand([all_num, all_num]).t()
        distmat = distmat.addmm(x=feat, y=feat.t(), alpha=-2.0, beta=1.0)

        original_dist = distmat.cpu().numpy()
        del feat
        if local_distmat is not None:
            original_dist = original_dist + local_distmat

    gallery_num = original_dist.shape[0]
    original_dist = np.transpose(original_dist / np.max(original_dist, axis=0))
    V = np.zeros_like(original_dist).astype(np.float16)
    initial_rank = np.argsort(original_dist).astype(np.int32)
    logger.info('starting re_ranking')
    for i in range(all_num):
        # k-reciprocal neighbors
        forward_k_neigh_index = initial_rank[i, :k1 + 1]
        backward_k_neigh_index = initial_rank[forward_k_neigh_index, :k1 + 1]
        fi = np.where(backward_k_neigh_index == i)[0]
        k_reciprocal_index = forward_k_neigh_index[fi]
        k_reciprocal_expansion_index = k_reciprocal_index
        for j in range(len(k_reciprocal_index)):
            candidate = k_reciprocal_index[j]
            candidate_forward_k_neigh_index = initial_rank[candidate, :int(
                np.around(k1 / 2)) + 1]
            candidate_backward_k_neigh_index = initial_rank[
                candidate_forward_k_neigh_index, :int(np.around(k1 / 2)) + 1]
            fi_candidate = np.where(
                candidate_backward_k_neigh_index == candidate)[0]
            candidate_k_reciprocal_index = candidate_forward_k_neigh_index[
                fi_candidate]
            if len(
                    np.intersect1d(candidate_k_reciprocal_index,
                                   k_reciprocal_index)) > 2 / 3 * len(
                                       candidate_k_reciprocal_index):
                k_reciprocal_expansion_index = np.append(
                    k_reciprocal_expansion_index, candidate_k_reciprocal_index)

        k_reciprocal_expansion_index = np.unique(k_reciprocal_expansion_index)
        weight = np.exp(-original_dist[i, k_reciprocal_expansion_index])
        V[i, k_reciprocal_expansion_index] = weight / np.sum(weight)
    original_dist = original_dist[:query_num, ]
    if k2 != 1:
        V_qe = np.zeros_like(V, dtype=np.float16)
        for i in range(all_num):
            V_qe[i, :] = np.mean(V[initial_rank[i, :k2], :], axis=0)
        V = V_qe
        del V_qe
    del initial_rank
    invIndex = []
    for i in range(gallery_num):
        invIndex.append(np.where(V[:, i] != 0)[0])

    jaccard_dist = np.zeros_like(original_dist, dtype=np.float16)
    for i in range(query_num):
        temp_min = np.zeros(shape=[1, gallery_num], dtype=np.float16)
        indNonZero = np.where(V[i, :] != 0)[0]
        indImages = [invIndex[ind] for ind in indNonZero]
        for j in range(len(indNonZero)):
            temp_min[0, indImages[j]] = temp_min[0, indImages[j]] + np.minimum(
                V[i, indNonZero[j]], V[indImages[j], indNonZero[j]])
        jaccard_dist[i] = 1 - temp_min / (2 - temp_min)

    final_dist = jaccard_dist * (1 - lambda_value
                                 ) + original_dist * lambda_value
    del original_dist
    del V
    del jaccard_dist
    final_dist = final_dist[:query_num, query_num:]
    final_dist = paddle.to_tensor(final_dist)
    return final_dist

```

然后讲讲我们用到的第二个评估类--检索任务，

这段代码包含了两个函数：`retrieval_eval` 和 `cal_feature`，以及另一个附带的函数 `re_ranking`。这些函数被用于在检索任务中评估模型的性能。

1. `retrieval_eval` 函数是对检索任务的评估过程的主要实现。它首先计算查询和库中的特征，然后计算这些特征之间的相似度，最后根据相似度计算各种检索评估指标（如准确率、召回率等）。
2. `cal_feature` 函数是用来计算数据集（可以是查询集、库集或查询和库的合集）中所有样本的特征的。它首先加载数据，然后通过模型计算每个样本的特征，最后将这些特征和对应的标签返回。
3. `re_ranking` 函数是一个用于重新排列检索结果的函数，它采用了 k-reciprocal encoding 和 query expansion 等技术来提高检索的精度。该函数首先计算查询和库的特征之间的距离，然后对每个查询计算其 k-互惠最近邻，并对这些邻居进行扩展以包括更多的相关样本。最后，它根据这些邻居的距离和扩展邻居的平均距离来计算查询和库之间的最终距离。

这些函数在一起提供了一个完整的检索任务的评估流程，可以用来评估模型在检索任务上的性能。

```python
 if self.mode == "eval" or (self.mode == "train" and
                                   self.config["Global"]["eval_during_train"]):
            if self.eval_mode == "classification":
                if "Metric" in self.config and "Eval" in self.config["Metric"]:
                    self.eval_metric_func = build_metrics(self.config["Metric"]
                                                          ["Eval"])
                else:
                    self.eval_metric_func = None
            elif self.eval_mode == "retrieval":
                if "Metric" in self.config and "Eval" in self.config["Metric"]:
                    metric_config = self.config["Metric"]["Eval"]
                else:
                    metric_config = [{"name": "Recallk", "topk": (1, 5)}]
                self.eval_metric_func = build_metrics(metric_config)
        else:
            self.eval_metric_func = None
```

我们可以在引擎初始化的代码中发现这么一段，大家可以先自己理解一下这段代码在干嘛

其实，这段代码是在训练和评估模型时，用于构建和初始化评估指标函数的。

- `self.mode` 表示当前的模式，可以是 "eval" 或 "train"。如果当前模式是 "eval"，则表示正在评估模型。如果当前模式是 "train"，并且配置中设置了 "eval_during_train" 为 True，则表示在训练过程中也需要进行评估。
- `self.eval_mode` 表示评估的模式，可以是 "classification" 或 "retrieval"。"classification" 表示分类任务，"retrieval" 表示检索任务。
- `build_metrics` 是一个函数，用于根据配置信息构建评估指标函数。
- `self.config["Metric"]["Eval"]` 是一个字典，包含了评估指标的配置信息。例如，对于分类任务，这可能包括准确率（accuracy）、召回率（recall）、精确度（precision）等指标；对于检索任务，这可能包括召回率@k（Recall@k）等指标。
- `self.eval_metric_func` 是最后构建的评估指标函数，用于在训练和评估过程中计算指标。

总的来说，这段代码的主要目的是根据配置信息，构建和初始化适当的评估指标函数，以便在后续的训练和评估过程中使用。

## 8. 模型优化

基于模型评估的反馈，我们可能需要进行模型优化。这可能涉及更改模型架构，调整超参数，或使用更复杂的模型。模型优化的目标是改进模型的预测性能。

## <img src="https://s2.loli.net/2023/07/21/ea4lkCA8bhBLW2K.jpg" width="1080" height=700>



------

## 																																																																																																													by寒晓月