---
title: Bio GA
tags: [ AI ]
date: 2020-04-01T06:25:44.226Z
path: project/bio-ga
slug: bio-ga
cover: ./bio-ga.png
excerpt: Biological and bio-inspired computation by genetic algorithm with BEAST.
---

## What is BEAST

BEAST is a cross-platform program for Bayesian analysis of molecular sequences using MCMC. It is entirely orientated towards rooted, time-measured phylogenies inferred using strict or relaxed molecular clock models. It can be used as a method of reconstructing phylogenies but is also a framework for testing evolutionary hypotheses without conditioning on a single tree topology. BEAST uses MCMC to average over tree space, so that each tree is weighted proportional to its posterior probability. We include a simple to use user-interface program for setting up standard analyses and a suit of programs for analysing the results.

## Introduction

[BEAST/beast](BEAST/beast) 中包含整个项目运行的BEAST源码（注意：代码需运行在配置好 BEAST 和 CMake 环境的操作系统中）。

**运行实例**：

1. Open terminal and type `cd beast/build`. If the build directory is not there, simply create it by `mkdir build`
2. Call CMake: `ccmake ..`
3. Configure: `c`
4. It will ask you for the `build type`: press 'enter' and type `Release` in the relevant field, then press 'enter' again
5. Again configure: `c`
6. Generate Makefile: `g`
7. type `make`
8. In the `build` directory, currently in there are two new directories called 'apps' and 'projects'. The 'apps' directory contains the executable for the beast program called 'beast' and the 'projects' directory contain the required dynamic library needed to run the program called 'libdemos.so'
9. From the build directory run the beast executable and pass the path to the dynamic library as a prameter as follows: `./apps/beast ./projects/libdemos.so`
10. When inside the program verify that you have a working build by going to 'File' and then 'Start simulation: Mice'

[BEAST/Mice](BEAST/Mice) 中包含了运行 Mice 模拟器结果处理过程，通过收集 log 日志的数据，用 Python 文件处理数据并绘制图片。`.txt`文件是收集的数据，`.png`文件是绘制的图片，`mouse_fitness.cc`和`mouse_network.cc`和`mouse_proximity.cc`是分别使用了不同的`Fitness`函数，神经网络和传感器后的`mouse.cc`源文件。

[BEAST/Chase](BEAST/Chase) 中包含了运行 10、50、100、4000、14000 代的数据以及 Prey 和 Predator 的关于 average fitness 和 best fitness 的折线图。

[BEAST/doc](BEAST/doc) 包括了对整个项目的配置与使用方法，可以访问 [BEAST - Bioinspired Evolutionary Agent Simulation Toolkit Documentation](https://blog.withh.life/download/bio) 进行在线阅读。

## Source Code

Available at: https://github.com/HurleyJames/BioGA.

## License

<a rel="license" href="http://creativecommons.org/licenses/by/4.0/"><img alt="知识共享许可协议" style="border-width:0" src="https://i.creativecommons.org/l/by/4.0/88x31.png" /></a>

本作品采用<a rel="license" href="http://creativecommons.org/licenses/by/4.0/">知识共享署名 4.0 国际许可协议</a>进行许可。