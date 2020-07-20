---
title: J48 Decision Tree by Weka
tags: [ AI ]
date: 2020-04-04T06:25:44.226Z
path: blog/weka-decision-tree
cover: ./weka-decision-tree.png
excerpt: J48 algorithm is one of the best machine learning algorithms to examine the data categorically and continuously. When it is used for instance purpose, it occupies more memory space and depletes the performance and accuracy in classifying medical data.
---

## WEKA

WEKA（Waikato Environment for Knowledge Analysis）诞生于University of Waikato（新西兰），并在1997年首次以现代的格式实现。

为了将数据加载到WEKA，我们必须将数据放入一个我们能够理解的格式。WEKA建议加载的数据格式是Attribute Relation File Format（ARFF）。其中含有三个重要的注解：

* @RELATION
* @ATTRIBUTE
* @DATA

<!-- more -->

## J48决策树算法

J48的全名是`weka.classifiers.trees.J48`。J48算法是著名的C4.5算法的改进，Weka对于这个算法赋予了默认的参数：-C 0.25 -M 2。该命令给出了分类器的默认参数配置，一般很少需要为提高性能而修改参数配置。前者是用于剪枝的置信因子，后者指定了每个叶结点最小的实例数。

通过运行weather.nominal.arff文件，在分类器面板的Test options部分选择Use training set，然后点击Start按钮创建分类器并进行评估。

运行完成后，可以在右侧的Classifier output中查看结果：

```
=== Run information ===

Scheme:       weka.classifiers.trees.J48 -C 0.25 -M 2
Relation:     weather.symbolic
Instances:    14
Attributes:   5
              outlook
              temperature
              humidity
              windy
              play
Test mode:    evaluate on training data

=== Classifier model (full training set) ===

J48 pruned tree
------------------

outlook = sunny
|   humidity = high: no (3.0)
|   humidity = normal: yes (2.0)
outlook = overcast: yes (4.0)
outlook = rainy
|   windy = TRUE: no (2.0)
|   windy = FALSE: yes (3.0)

Number of Leaves  : 	5

Size of the tree : 	8


Time taken to build model: 0.01 seconds

=== Evaluation on training set ===

Time taken to test model on training data: 0 seconds

=== Summary ===

Correctly Classified Instances          14              100      %
Incorrectly Classified Instances         0                0      %
Kappa statistic                          1     
Mean absolute error                      0     
Root mean squared error                  0     
Relative absolute error                  0      %
Root relative squared error              0      %
Total Number of Instances               14     

=== Detailed Accuracy By Class ===

                 TP Rate  FP Rate  Precision  Recall   F-Measure  MCC      ROC Area  PRC Area  Class
                 1.000    0.000    1.000      1.000    1.000      1.000    1.000     1.000     yes
                 1.000    0.000    1.000      1.000    1.000      1.000    1.000     1.000     no
Weighted Avg.    1.000    0.000    1.000      1.000    1.000      1.000    1.000     1.000     

=== Confusion Matrix ===

 a b   <-- classified as
 9 0 | a = yes
 0 5 | b = no
```

### 分析

#### Instances

代表输入的数据量

#### Attributes

代表数据中有哪些数据分类，即属性

#### Number of Leaves

叶子树

#### Size of the tree

决策树大小

#### Kappa statistic

这个参数是把分类器与随机分类器作比较得出的一个对分类器的评价值。

#### Mean absolute error和Root mean squared error

平均绝对误差，用来衡量分类器预测值和实际结果的差异，越小越好。

#### Relative absolute error和Root relative squared error

有时候绝对误差不能体现误差的真实大小，而**相对误差**通过体现误差占真值的比重来反映误差大小的效果会更好。

#### Confusion Matrix

混淆矩阵。这个矩阵上对角线的数字越大，说明预测得越好。

