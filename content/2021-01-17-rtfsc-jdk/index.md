---
title: RTFSC-JDK
tags: [ Java ]
date: 2021-01-17T06:25:44.226Z
path: project/rtfsc-jdk
slug: rtfsc-jdk
cover: ./rtfsc-jdk.png
excerpt: Reading The Fucking Source Code - JDK
---

## 🥳 RTFSC-JDK 是什么？

> Reading The Fucking Source Code of JDK.

本项目主要是存放 JDK11 的**源码**与**笔记**：

* 对于代码`src`部分，建议在`OracleJDK / OpenJDK 11`的环境下阅读代码，阅读过程中产生的部分笔记会以**注释**的形式写在源码中。
* 对于文档`docs`部分，建议访问文档的[网站](https://rtfsc.withh.life/)进行阅读。

![网站截图](https://i.loli.net/2021/01/17/L2b6moOfSvx7cnz.png)

## 📝 使用说明

1. 将[本项目](https://github.com/HurleyJames/RTFSC-JDK)克隆或者下载到本地。
2. `src`部分可以使用 IntelliJ IDEA 打开阅读并调试。请注意，该源代码不支持直接编译，如果想完整编译整个 JDK 项目，请参考官方教程 [Building the JDK](https://hg.openjdk.java.net/jdk/jdk11/raw-file/tip/doc/building.html)。
3. `docs`部分是源码阅读的笔记，可以`cd docs`进入目录，输入`npm install`安装相关依赖，然后使用`npm run docs:dev`运行启动，可以自行编写。
4. 如果有缺失遗漏或者源码解读错误的地方，欢迎在 [Github Issues](https://github.com/HurleyJames/RTFSC-JDK/issues) 中提出，我会尽量及时反馈更新。

## 🔗 相关链接

* [Oracle JDK](https://www.oracle.com/java/technologies/oracle-java-archive-downloads.html)
* [OpenJDK](http://jdk.java.net/archive/)

## 🙇‍♂️ 鸣谢

此项目受到 [LeaningJDK](https://github.com/kangjianwei/LearningJDK) 项目的启发，JDK 部分的主要源码也是克隆至该项目，后期加上自己的解读，希望大家到 [kangjianwei](https://github.com/kangjianwei) 的项目中多多支持与 star。

## Source Code

Available at: https://github.com/HurleyJames/RTFSC-JDK.