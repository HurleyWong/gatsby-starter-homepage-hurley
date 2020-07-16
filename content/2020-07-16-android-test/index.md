---
title: Image Classification with Convolutional Neural Networks
tags: [ AI, Python ]
date: 2020-03-20T05:25:44.226Z
path: blog/android-test
cover: ./codesplitting.png
excerpt: Suppose the bundle size of your project is very huge and you don't want to load everything at once during the initial render you can use code splitting along with React to improve the performance and load time of your application.
---

### Requirements

#### Building the network

The first part of the assignment is to build a CNN and train it on ImageNet10. Start by building a CNN with the following depth and parameters: 

| Input_channels | Output_channels |                 | Kernel_size |
| :------------: | :-------------: | :-------------: | :---------: |
|       3        |       16        |   convolution   |    3 x 3    |
|       16       |       24        |   convolution   |    4 x 4    |
|       24       |       32        |   convolution   |    4 x 4    |
|       *        |       512       | fully-connected |             |
|      512       |       10        | fully-connected |             |

(*) Calculate the input size, which is the flattened previous layer.

<!-- more -->

Follow each convolutional layer by ReLU and a max-pool operation with kernel size 2 x 2. 

After each sequence of convolution, ReLU and max-pool, add a dropout operation with p=0.3. Dropout is a regularisation technique where layer outputs are randomly (with likelihood of p) dropped out, or ignored. This helps the neural network to avoid overfitting to noise in the training data.

Use a learning rate of 0.001, momentum 0.9, and stochastic gradient descent to train the network. Training for 10 epochs should give you an accuracy in the 40th percentile on the test set. Remember that for ten classes, random performance is 10%.

#### Experiments

Now, run two experiments testing various network architectures: 

1. How does the number of layers affect the training process and test performance? Try between 2 and 5 layers.
2. Choose one more architectural element to test (filter size, max-pool kernel size, number/dimensions of fully-connected layers, etc.).

Generate and display the confusion matrices for the various experiments.

#### Filter visualisation

Now, choose the best-performing network architecture and visualise the filters from the first layer of the network. Compare the filters before training (random weight initialisation), halfway during training, and after training is completed. Normalise the filters before displaying as an image. The filters are 3D, but are easier to visualise when displayed in grayscale. You can do this using matplotlib.pyplot’s colour map argument: `plt.imshow(filter, cmap=”gray”)`

#### Feature map visualisation

Feature maps are the filters applied to the layer input; i.e., they are the activations of the convolutional layers. Visualise some of the feature maps (four from each layer) from the fullytrained network, by passing images forward through the network. As with the filters, display them in grayscale.

Choose two input images of different classes and compare the resulting feature maps, commenting on how they change with the depth of the network.

**Tips**:

When visualising a PyTorch tensor object, you will want to first move the tensor to CPU (if using a GPU), detach it from the gradients, make a copy, and convert to a NumPy array. You can do this as follows:

```
tensor_name.cpu().detach().clone().numpy()
```

The most intuitive way to get layer activations is to directly access them in the network. 

```
conv1_activs = model.conv_layer1.forward(input_image)
```

The layer activations for the subsequent layer can then be obtained by passing in the previous layer activations: `conv2_activs = model.conv_layer2.forward(conv1_activs)`

Alternately, for a more elegant solution, you may choose to save layer activations by registering a hook into your network. Look for documentation on the `model.register_forward_hook()` function to see how to use it. 

#### Improving network performance

Consider and implement at least two adjustments which you anticipate will improve the model’s ability to generalise on unseen data. While you may completely change the network architecture, please do not use any additional data to train your model other than the ImageNet10 training data provided. Changes related to data loading, transforms, and the training process and parameters are also valid adjustments as long as you justify your choices and provide some form of validation.

### Solution

具体代码见[AI_Coursework1.ipynb](https://github.com/HurleyJames/GoogleColabExercise/blob/master/AI_Coursework1.ipynb)。

#### Building the network

第一步的**构建网络**意味着我们要根据已经给定的输入层和输出层通道的大小，以及卷积层卷积核（过滤器）的大小，计算出 * 对应的值的大小。具体的计算方法可以见[计算卷积层和池化层输出大小](http://hurley.fun/2020/02/29/PyTorch学习笔记（1）计算卷积层和池化层输出大小/#卷积操作输出的计算公式)。同时按照要求，设置使用的激活函数，池化层卷积核的大小，学习率，动量和Dropout的大小等，训练次数epoch为10次。根据以上固定数据和计算出的数据，就可以将该网络的模型搭建好。

#### Experiments

这一部分主要有两个步骤需要完成。

1. 选取2-5层卷积层，判断究竟几层的性能是最好的；
2. 选择最少一个因素（卷积核大小，池化层卷积核大小，全连接层个数等）进行改变以提高性能表现。

对于第一个步骤，则要分别搭建2-5层卷积层的网络模型。针对不同层数的网络模型，其 * 都要重新计算。

##### 网络模型

**2层**：

```
 class ConvNet2(nn.Module):
 
     def __init__(self, num_classes=10):
         super(ConvNet2, self).__init__()
 
         self.conv1 = nn.Conv2d(3, 16, 3)
         self.conv2 = nn.Conv2d(16, 24, 4)
         
         self.fc1 = nn.Linear(24*62*62, 512)
         self.fc2 = nn.Linear(512, num_classes)
         self.pool1 = nn.MaxPool2d(2)
         self.pool2 = nn.MaxPool2d(2)
         self.dropout1 = nn.Dropout2d(0.3)
         self.dropout2 = nn.Dropout2d(0.3)
 
     def forward(self, x):
         x = self.pool1(F.relu(self.conv1(x)))
         x = self.dropout1(x)
 
         x = self.pool2(F.relu(self.conv2(x)))
         x = self.dropout2(x)
 
         x = x.view(-1, 24*62*62) 
         x = F.relu(self.fc1(x))
         x = F.relu(self.fc2(x))
 
         return x
 
 model2 = ConvNet2()
 print(model2)
 class ConvNet3(nn.Module):
     
     def __init__(self, num_classes=10):
         super(ConvNet3, self).__init__()
 
         self.conv1 = nn.Conv2d(3, 16, 3)
         self.conv2 = nn.Conv2d(16, 24, 4)
         self.conv3 = nn.Conv2d(24, 32, 4)
 
         self.fc1 = nn.Linear(32*29*29, 512)
         self.fc2 = nn.Linear(512, num_classes)
         self.pool1 = nn.MaxPool2d(2)
         self.pool2 = nn.MaxPool2d(2)
         self.pool3 = nn.MaxPool2d(2)
         self.dropout1 = nn.Dropout2d(0.3)
         self.dropout2 = nn.Dropout2d(0.3)
         self.dropout3 = nn.Dropout2d(0.3)
         
     def forward(self, x):
         x = self.pool1(F.relu(self.conv1(x)))
         x = self.dropout1(x)
 
         x = self.pool2(F.relu(self.conv2(x)))
         x = self.dropout2(x)
 
         x = self.pool3(F.relu(self.conv3(x)))
         x = self.dropout3(x)
 
         x = x.view(-1, 32*29*29) 
         x = F.relu(self.fc1(x))
         x = F.relu(self.fc2(x))
 
         return x
 
 model3 = ConvNet3()
 print(model3)
 class ConvNet4(nn.Module):
     
     def __init__(self, num_classes=10):
         super(ConvNet4, self).__init__()
 
         self.conv1 = nn.Conv2d(3, 16, 3)
         self.conv2 = nn.Conv2d(16, 24, 4)
         self.conv3 = nn.Conv2d(24, 32, 4)
         self.conv4 = nn.Conv2d(32, 40, 4)
 
         self.fc1 = nn.Linear(40*13*13, 512)
         self.fc2 = nn.Linear(512, num_classes)
         self.pool1 = nn.MaxPool2d(2)
         self.pool2 = nn.MaxPool2d(2)
         self.pool3 = nn.MaxPool2d(2)
         self.pool4 = nn.MaxPool2d(2)
         self.dropout1 = nn.Dropout2d(0.3)
         self.dropout2 = nn.Dropout2d(0.3)
         self.dropout3 = nn.Dropout2d(0.3)
         self.dropout4 = nn.Dropout2d(0.3)
         
     def forward(self, x):
 
         x = self.pool1(F.relu(self.conv1(x)))
         x = self.dropout1(x)
 
         x = self.pool2(F.relu(self.conv2(x)))
         x = self.dropout2(x)
 
         x = self.pool3(F.relu(self.conv3(x)))
         x = self.dropout3(x)
 
         x = self.pool4(F.relu(self.conv4(x)))
         x = self.dropout4(x)
 
         x = x.view(-1, 40*13*13) 
         x = F.relu(self.fc1(x))
         x = F.relu(self.fc2(x))
 
         return x
 
 model4 = ConvNet4()
 print(model4)
 class ConvNet5(nn.Module):
     
     def __init__(self, num_classes=10):
         super(ConvNet5, self).__init__()
 
         self.conv1 = nn.Conv2d(3, 16, 3)
         self.conv2 = nn.Conv2d(16, 24, 4)
         self.conv3 = nn.Conv2d(24, 32, 4)
         self.conv4 = nn.Conv2d(32, 40, 4)
         self.conv5 = nn.Conv2d(40, 48, 4)
 
         self.fc1 = nn.Linear(48*5*5, 512)
         self.fc2 = nn.Linear(512, num_classes)
         self.pool1 = nn.MaxPool2d(2)
         self.pool2 = nn.MaxPool2d(2)
         self.pool3 = nn.MaxPool2d(2)
         self.pool4 = nn.MaxPool2d(2)
         self.pool5 = nn.MaxPool2d(2)
         self.dropout1 = nn.Dropout2d(0.3)
         self.dropout2 = nn.Dropout2d(0.3)
         self.dropout3 = nn.Dropout2d(0.3)
         self.dropout4 = nn.Dropout2d(0.3)
         self.dropout5 = nn.Dropout2d(0.3)
 
         
     def forward(self, x):
 
         x = self.pool1(F.relu(self.conv1(x)))
         x = self.dropout1(x)
 
         x = self.pool2(F.relu(self.conv2(x)))
         x = self.dropout2(x)
 
         x = self.pool3(F.relu(self.conv3(x)))
         x = self.dropout3(x)
 
         x = self.pool4(F.relu(self.conv4(x)))
         x = self.dropout4(x)
 
         x = self.pool5(F.relu(self.conv5(x)))
         x = self.dropout5(x)
 
         x = x.view(-1, 48*5*5) 
         x = F.relu(self.fc1(x))
         x = F.relu(self.fc2(x))
 
         return x
 
 model5 = ConvNet5()
 print(model5)
 def train_and_valid(num_epochs, model, optimizer, flag):  
     for epoch in range(num_epochs): 
         correct = 0
         total = 0 # loop over the dataset multiple times
 
         running_loss = 0.0
         total_val_loss =0.0
 
         if (epoch ==5 and flag ==1):
             filter_visual()
         for i, data in enumerate(train_loader, 0):
             # get the inputs; data is a list of [inputs, labels]
             inputs, labels = data
             inputs = inputs.to(device)
             labels = labels.to(device)
 
             # zero the parameter gradients
             optimizer.zero_grad()
 
             # forward + backward + optimize
             outputs = model(inputs)
             loss = loss_fn(outputs, labels)
             loss.backward()
             optimizer.step()
 
             # Print our loss
             running_loss += loss.item()
             _, predicted = torch.max(outputs.data, 1)
             total += labels.size(0)
             correct += (predicted == labels).sum().item()
         print('Epoch %d - Loss : %.3f' % (epoch+1,running_loss/len(train_loader)))
         print('The accuracy of training set is %.3f %%' % (correct/total*100))
         loss_lst.append(running_loss/len(train_loader))
         acc_lst.append(correct/total*100)
 
         with torch.no_grad():
             val_correct = 0 
             val_total = 0
             for inputs, labels in valid_loader:
                 inputs = inputs.to(device)
                 labels = labels.to(device)
                     
                 val_outputs = model(inputs)
                 val_loss = loss_fn(val_outputs, labels)
                 total_val_loss += val_loss.item()
                 _,prediction = torch.max(val_outputs.data,1)
                 val_total += labels.size(0)
                 val_correct += (prediction == labels).sum().item()
         print("Validation loss = {:.2f}".format(total_val_loss / len(valid_loader)))
         print('The accuracy of Validation set is %.3f %%' %(val_correct/val_total*100))
 
         val_loss_lst.append(total_val_loss / len(valid_loader))
         val_acc_lst.append(val_correct/val_total*100)
         
     print('Finished')
 class ReConvNet(nn.Module):
     
     def __init__(self, num_classes=10):
         super(ReConvNet, self).__init__()
 
         self.conv1 = nn.Conv2d(3, 16, 3)
         self.conv2 = nn.Conv2d(16, 24, 4)
         self.conv3 = nn.Conv2d(24, 32, 4)
         self.conv4 = nn.Conv2d(32, 40, 4)
 
         self.fc1 = nn.Linear(40*13*13, 512)
         self.fc2 = nn.Linear(512, num_classes)
         self.pool1 = nn.MaxPool2d(2)
         self.pool2 = nn.MaxPool2d(2)
         self.pool3 = nn.MaxPool2d(2)
         self.pool4 = nn.MaxPool2d(2)
         self.dropout1 = nn.Dropout2d(0.1)
         self.dropout2 = nn.Dropout2d(0.1)
         self.dropout3 = nn.Dropout2d(0.1)
         self.dropout4 = nn.Dropout2d(0.1)
         
     def forward(self, x):
 
         x = self.pool1(F.relu(self.conv1(x)))
         x = self.dropout1(x)
 
         x = self.pool2(F.relu(self.conv2(x)))
         x = self.dropout2(x)
 
         x = self.pool3(F.relu(self.conv3(x)))
         x = self.dropout3(x)
 
         x = self.pool4(F.relu(self.conv4(x)))
         x = self.dropout4(x)
 
         x = x.view(-1, 40*13*13) 
         x = F.relu(self.fc1(x))
         x = F.relu(self.fc2(x))
 
         return x
 
     def retrieve_features(self, x):
         feature_map1 = F.relu(self.conv1(x))
         x = self.pool1(feature_map1)
         x = F.dropout2d(x, 0.1)
 
         feature_map2 = F.relu(self.conv2(x))
         x = self.pool2(feature_map2)
         x = F.dropout2d(x, 0.1)
 
         feature_map3 = F.relu(self.conv3(x))
         x = self.pool3(feature_map3)
         x = F.dropout2d(x, 0.1)
 
         feature_map4 = F.relu(self.conv4(x))
         x = self.pool4(feature_map4)
         x = F.dropout2d(x, 0.1)
 
         return (feature_map1, feature_map2, feature_map3, feature_map4)
 
 model = ReConvNet()
 print(model)
 correct: 714
 total: 1800
 Accuracy of the network on the images: 39 %
 Accuracy of baboon: 42 %
 Accuracy of banana: 52 %
 Accuracy of canoe: 41 %
 Accuracy of   cat: 54 %
 Accuracy of  desk: 14 %
 Accuracy of drill: 44 %
 Accuracy of dumbbell: 18 %
 Accuracy of football: 34 %
 Accuracy of   mug: 41 %
 Accuracy of orange: 80 %
 Normalized confusion matrix
 [[ 82   2  15  50   8   5   3   4   5   2]
  [  5  90   9  12   4   3   2   8  10  43]
  [ 16  11  86  26   8   6   5  11  12   3]
  [ 35  11   7  78   6   4   8  10  15   1]
  [ 20   8  21  17  38  12  21  13  18   6]
  [ 14  12  10  15  13  65  14   3  25   6]
  [ 31   7  12  21  14  25  45   8  23   9]
  [ 18  16  10  37   8   8  11  40  16   3]
  [ 25  21  11  14   9  11  16  10  61  19]
  [  1  25   1   4   2   0   1   1   5 129]]
 def filter_visual():
   figure = plt.figure(figsize = (15, 15))
   k = 0
 
   for i in range(16):
     
     filter_mix = np.zeros(shape = [3,3])
     for j in range(3):
       k = k + 1
       figure.add_subplot(8, 6, k)
       filter_mix += model.conv1.weight.data.cpu().numpy()[ i, j, :, :]
       plt.imshow(filter_mix, cmap = "gray")
   plt.show()
 idx = 1000
 idy = 500
 
 input_x = ins_dataset_test[idx][0].unsqueeze(0)
 input_y = ins_dataset_test[idy][0].unsqueeze(0)
 feature_maps_x = model.retrieve_features(input_x.to(device))
 feature_maps_y = model.retrieve_features(input_y.to(device))
 
 plt.figure(figsize=(20, 14))
 for i in range(4):
     plt.subplot(1, 4, i + 1)
     plt.axis('off')
     plt.imshow(feature_maps_x[0][0, i, ...].data.cpu().numpy(), cmap='gray')
 plt.figure(figsize = (20, 14))
 for i in range(4):
     plt.subplot(2 ,4 ,i + 1)
     plt.axis('off')
     plt.imshow(feature_maps_x[1][0, i,...].data.cpu().numpy(), cmap='gray')
 plt.figure(figsize = (20, 14))
 for i in range(4):
     plt.subplot(4 ,4 ,i + 1)
     plt.axis('off')
     plt.imshow(feature_maps_x[3][0, i,...].data.cpu().numpy(), cmap='gray')
 from torchvision import models
 
 final_model = models.resnet18(pretrained = False)
 
 for param in final_model.parameters():
     param.require_grad = False
 loss_fn = nn.CrossEntropyLoss()
 final_optimizer = optim.Adam(final_model.parameters(), lr = 0.001)
 correct: 1162
 total: 1800
 Accuracy of the network on the images: 64 %
 Accuracy of baboon: 91 %
 Accuracy of banana: 43 %
 Accuracy of canoe: 78 %
 Accuracy of   cat: 58 %
 Accuracy of  desk: 76 %
 Accuracy of drill: 41 %
 Accuracy of dumbbell: 65 %
 Accuracy of football: 48 %
 Accuracy of   mug: 49 %
 Accuracy of orange: 85 %
 Normalized confusion matrix
 [[153   2   2  14   0   0   4   1   0   0]
  [  6  93   6   9   7   1   5   3   6  50]
  [ 22   2 128   2  11   4   5   8   2   0]
  [ 33   1   2 114   5   1  10   3   4   2]
  [  0   1   1   1 140   4  13   6   8   0]
  [  3   1   5   4  23  78  46  10   5   2]
  [ 11   5   1   4  13  13 127   7  11   3]
  [ 15  11   4   8   3   8  24  77  13   4]
  [  3   6   1   7  28   2  17  11 111  11]
  [  2   5   4   1   8   1   1   0   6 141]]
```

可以发现自带的性能的确比我们自己搭建的模型要好的多，同理也可以使用其它PyTorch中自带的网络模型，例如VGG16等。

![img](data:image/png;base64,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%0ATkSuQmCC%0A)

**混淆矩阵**：

**测试集的准确率**：

![img](https://raw.githubusercontent.com/HurleyJames/ImageHosting/master/Unknown-11.png)

训练集和验证集的准确率：

![img](https://raw.githubusercontent.com/HurleyJames/ImageHosting/master/Unknown-10.png)

训练集和验证集的损失率：

同时改变优化器和损失函数：

这步规定可以使用PyTorch内置的模型（不过不能是预训练好的，即`pretrained = False`），所以我这里采用的是ResNet18模型。

#### Improving network performance

可以看见，随着层次的增加，图片越来越模糊，特征越来越不明显。

![img](https://raw.githubusercontent.com/HurleyJames/ImageHosting/master/Unknown-9.png)

**第三层**：

![img](https://raw.githubusercontent.com/HurleyJames/ImageHosting/master/Unknown-8.png)

**第二层**：

![img](https://raw.githubusercontent.com/HurleyJames/ImageHosting/master/Unknown-7.png)

**第一层**：

对特征图进行可视化。可以从数据集中随机选择图片，因此之前已经设置了`shuffle=True`，即随机打乱顺序，所以这里只要设置索引即可。

#### Feature map visualisation

![img](https://raw.githubusercontent.com/HurleyJames/ImageHosting/master/Unknown-6.png)

**训练后**：

![img](https://raw.githubusercontent.com/HurleyJames/ImageHosting/master/Unknown-5.png)

**训练中**：

![img](https://raw.githubusercontent.com/HurleyJames/ImageHosting/master/Unknown-4.png)

**训练前**：

因为Requirements的要求在训练前、中、后分别进行卷积核可视化来观察变化。所以在`model.train()`之前就要调用`filter_visual()`。总共10轮，因此在中间即第6轮时再调用一次`filter_visual()`，然后在训练结束之后，再次调用`filter_visual()`。这样就能得到训练前、中、后三次的卷积核可视化。

卷积核（过滤器）可视化，主要是因为卷积核会随着网络训练而发生自我调整。因为第一层的输出是16，而该图片集又是RGB三通道的图片，所以第一层总共会输出张卷积核的图片。

#### Filter visualisation

![img](data:image/png;base64,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%0Ax+PxBMj/A/zw4QoVBaNhAAAAAElFTkSuQmCC%0A)

通过对该模型进行训练，最终拿测试集去跑，得到的准确率如下：

因为后面还有一个Feature map visualisation即特征图可视化，所以这里新添了一个方法`retrieve_features(self, x)`的方法，记录每一层的特征图。

因为4层的收敛效果很好，所以可以适当的考虑降低Dropout的值，所以我这里仅仅改变了Dropout的值，由0.3变成0.1。

##### 改变参数

![img](https://raw.githubusercontent.com/HurleyJames/ImageHosting/master/Unknown-3.png)

4层的损失率：

最终发现4层的相对来说收敛效果最好（5层虽然收敛很好，但是准确率已经明显低于40%，不满足要求）。

可以看见图像仍然没完全收敛，存在欠拟合现象。

![img](https://raw.githubusercontent.com/HurleyJames/ImageHosting/master/Unknown-2.png)

2层的准确率：

![img](https://raw.githubusercontent.com/HurleyJames/ImageHosting/master/Unknown.png)

2层的损失率：

这里需要注意的是，在进行训练和验证时，要同步进行，而不是说在训练集训练完模型再去跑验证集。

在搭建完网络，定义损失函数和优化器之后，需要通过训练和验证来判断性能的优劣。这里需要注意的是，判断性能并不是意味着是训练模型之后去跑测试集来根据准确率的高低去判断，而是要根据训练集与验证集的损失率或者准确率的曲线是否收敛来判断，否则如果有很明显的过拟合或者欠拟合，都不是好的性能。

##### 判断性能

由于Requirements中只给出了3层卷积层的数值大小，所以当添加到4-5层时，新增的大小可以自己设置，这里我仍然设置为以8为间隔。

5卷积层时，全连接层的输入大小为。

**5层**：

4层卷积层时，全连接层的输入大小为。

**4层**：

3层卷积层时，全连接层的输入大小为。

**3层**：

2层卷积层时，全连接层的输入大小为。