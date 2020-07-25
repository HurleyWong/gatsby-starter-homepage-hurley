---
title: NNs
tags: [ AI ]
date: 2020-02-19T06:25:44.226Z
path: project/nns
slug: nns
cover: ./nns.png
excerpt: The NLP and CV exercises by different ANNs (CNN, RNN, GAN, .etc).
---

## Overview
* Intro_to_PyTorch.ipynb: The purpose of this notebook is to give you a general understanding of how to use the [PyTorch](https://pytorch.org/) Python package for writing, training and analysing neural networks.
* Linear_Classifier.ipynb: Train a multiclass linear classifier on the CIFAR10 dataset in [Keras](https://keras.io/).
* Using_Word2Vec_Embeddings.ipynb: News Category Classification using Word2Vec embeddings with two Python libraries, [scikit-learn](https://scikit-learn.org/stable/) and [gensim](https://radimrehurek.com/gensim/).
* Image_Classification_with_Convolutional_Neural_Networks_ImageNet10.ipynb: Use convolutional neural networks, from development to training and testing. Plot feature maps and filters during the training process and testing. Explore methods of improving performance on a network. The details can be found in this article [Image Classification with Convolutional Neural Networks - ImageNet10](https://hurleyjames.github.io/2020/03/20/Image-Classification-with-Convolutional-Neural-Networks---ImageNet10/).
* [Image Caption Generation](Image_Caption_Generation.ipynb): Do text pre-processing and text embeddings with an image to text model, compare the performance and usage of RNNs versus LSTMs as sequence generators.
* Perceptron: Compare a simple perceptron and a simple network built by myself to a network model by Keras. The details can be found in this article [Perceptron Algorithm and Backpropagation](https://hurleyjames.github.io/2020/04/23/Perceptron-Algorithm-and-Backpropagation/#introduction).

### Training and test dataset
* [CIFAR-10](https://www.cs.toronto.edu/~kriz/cifar.html): The CIFAR-10 dataset consists of 60000 32x32 colour images in 10 classes, with 6000 images per class. There are 50000 training images and 10000 test images.
* [ImageNet10](https://github.com/MohammedAlghamdi/imagenet10): A subset of images from the ImageNet dataset, which contains 9000 images belonging to ten object classes.
* [Flickr8k](https://github.com/jbrownlee/Datasets/releases/tag/Flickr8k): This repository contains a copy of machine learning datasets used in tutorials on [MachineLearningMastery.com](https://machinelearningmastery.com/). 
* [Iris](http://archive.ics.uci.edu/ml/datasets/iris): This is perhaps the best known database to be found in the pattern recognition literature.  The data set contains 3 classes of 50 instances each, where each class refers to a type of iris plant.

### Network models
* ResNet18: Use in Image_Classification_with_Convolutional_Neural_Networks_ImageNet10.ipynb
* ResNet152: Use in Image_Caption_Generation.ipynb

## Building and Runnning
For most `.ipynb` files, you can directly open it in Colab to run.  
As for Image_Caption_Generation.ipynb, [Flickr8k](https://github.com/jbrownlee/Datasets/releases/tag/Flickr8k) dataset is very large, so if you want to work on Colab, it is recommended to download the zip files, unzip them, and then upload all the files to your Google Drive. This initial upload may take a while, but from then on you will only need to mount your Drive every time you start a new Colab session and the data will be immediately accessible. Mounting only takes a few second. Do not forget to replace the path with your own root data directory of your Google Drive.  
As for .py files in the Perceptron folder, especially the perceptron.py. You have to open terminal to type: `python perceptron.py <class> with/without`, the class can be `setosa`, `versicolor` and `verginica`. And `with` means with learning rate 0.01, `without` means without using learning rate.

## Source Code

Available at: https://github.com/HurleyJames/NNs.

## License
<a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="知识共享许可协议" style="border-width:0" src="https://i.creativecommons.org/l/by/4.0/88x31.png" /></a>  

本作品采用<a rel="license" href="http://creativecommons.org/licenses/by/4.0/">知识共享署名 4.0 国际许可协议</a>进行许可。