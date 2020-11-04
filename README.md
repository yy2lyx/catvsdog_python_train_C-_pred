# catvsdog_python_train_C-_pred

#### 1. 项目描述

主要用于测试将keras的训练后的模型（图像分类）移植到C++平台上

* 数据集：[Kaggle猫狗分类](https://www.kaggle.com/tongpython/cat-and-dog)
* keras版本：2.4.3
* tensorflow版本：2.3.1

#### 2. python代码结构

```json
├─data // 这里存放从kaggle下载好的猫狗分类数据集（解压后的）
└─model // 存放模型
	├─model.h5 // 保存的h5文件
	└─model.pt // pt文件
├─h5file2pt.py // 将h5转成pt格式的文件脚本
├─train.py // 训练
└─predict.py // 预测
```

#### 3. C++代码结构

```json
├─opencv_release.props // opencv配置文件
├─tf_release.props // tensorflow配置文件
└─test.cpp // tf预测文件
```

