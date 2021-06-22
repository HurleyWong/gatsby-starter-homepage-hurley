---
title: macOS BigSur Icons
tags: [ Icon ]
date: 2021-01-03T06:25:44.226Z
path: project/mac-bigsur-icons
slug: mac-bigsur-icons
cover: ./mac-bigsur-icons.png
excerpt: macOS Big Sur style round corner icon collection.
---

## Manual replacement

下载图标集合或者您所需的图标，在 Finder 中点击侧边栏的「应用程序」，右键单击需要更改的应用程序，选择「显示简介」，然后将下载的`.icns`文件拖放到左上角的图标那替换即可（会出现一个绿色的 ＋ 号）。

![](https://i.loli.net/2021/02/20/mUGEZxAKbLeckSY.jpg)

## Script replacement

对于某些尚未有支持的第三方图标的应用，就无法使用手动替换。那么，推荐使用这个开源项目：[iconsur](https://github.com/rikumi/iconsur)。

iconsur 是一个为应用程序生成 macOS Big Sur 风格的自适应图标的命令行工具。对于既存在 macOS 端又存在 iOS 端的应用，它可以通过在 iOS 端的 App Store 中搜寻对应的图标进行替换；对于不存在 iOS 端的应用，它也可以通过缩放、添加背景、蒙版等方式为其生成符合 macOS Big Sur 风格的应用图标并替换。具体操作说明请移步其文档：[README.md](https://github.com/rikumi/iconsur/blob/master/README.md)。

## Source Code

Available at: https://github.com/HurleyWong/macOS-Big-Sur-icon-collection.