---
title: Timing and Synchronisation
tags: [ CloudComputing ]
date: 2019-12-23T06:25:44.226Z
path: blog/timing-and-sync
cover: ./timing-and-sync.png
excerpt: J48 algorithm is one of the best machine learning algorithms to examine the data categorically and continuously. When it is used for instance purpose, it occupies more memory space and depletes the performance and accuracy in classifying medical data.
---

## The Eight Fallacies of Distributed Computing

* The network is reliable
* Latency is zero
* Bandwidth is infinite
* The network is secure
* Topology doesn't change
* There is one administrator
* Transport cost is zero
* The network is homogeneous
* <font color=red>All blocks are synchronized</font>

第9大悖论即**所有时钟是同步的**。

<!-- more -->

## 时钟同步

In a DS, there is <font color=red>no global agreement on time.</font>

### 物理时钟

几乎所有的计算机都有一个计时电路。但它们不是通常意义上的时钟，称为**计时器（timer）**更为掐当。这里还有几个概念：

* 有两个计数器与每个石英晶体相关联，一个是**计数器（counter）**，另一个是**保持寄存器（holding register）**
* 每次的中断称为一个时钟滴答（When counter reaches zero, a <font color=red>timer interrupt</font> or <font color=red>clock tick</font> is generated and counter is reloaded from a holding register）
* 时钟偏移：In a DS with *n* machines, all *n* crystals will run at slightly different rates, resulting in <font color=red>clock skew</font>.

### 时钟同步算法

#### Cristian's Algorithm

Cristian提出让客户与**时间服务器**（time server）联系。

<center>
    <img src="https://raw.githubusercontent.com/HurleyJames/ImageHosting/master/time.png" width="80%">
</center>

如图所示，best estimate of one-way propagation time is ***(T1-T0-I)/2***.

**例题：**
The client’s clock reads 5:26:08. The server’s clock reads 5:16:44 when they synchronize using Cristian’s algorithm. Assume RTT is 2 seconds. What is the time at the client after synchronization? Note: the time format is HH:MM:SS.

**解答：**

Cristian's algorithm assumes that the server has an accurate clock. The client requests the time and sets its clock to the server's time $+\frac{1}{2}(RTT)$. In this case, the RTT is 2 seconds, so the client set time after synchronization: $5:16:44 + \frac{1}{2}*2seconds = 5:16:45$

#### Berkeley Algorithm

Berkeley UNIX系统中的时间服务器（实际上是时间守护程序）是主动的，它定期地询问每台机器的时间。基于这些回答，它计算出一个平均时间，并告诉所有其他机器将它们的时钟快拨到一个新的时间，或者拨慢时间。

<center>
    <img src="https://raw.githubusercontent.com/HurleyJames/ImageHosting/master/bb.png" width="80%">
</center>

**例题1：**

The client’s clock reads 5:26:00. The server’s clock reads 5:14:00 when they synchronize using the Berkeley algorithm. Assume message delays are negligible. What is the time at the client after synchronisation? Note: the time format is HH:MM:SS.

**解答：**

The Berkeley algorithm averages clocks among the entire group. In this case, the group has two members: the client and the server. The average of the two clocks is $(5:26:00+5:14:00)/2=5:20:00$. Both the client and server will be set at $5:20:00$.

**例题2：**

Consider a network consisting of 5 computers, A (coordinator), B, C, D, and E. At 08:45 the coordinator decides to synchronise the clock of all computers in the network. The time format is HH:MM. At that moment, the clock of the computers in the network shows the following: B(08:43), C(08:49), D(08:42), E(08:46). Apply the Berkeley clock synchronisation algorithm to this situation, show the stages of computation, and explain the outcome of the synchronisation. You may assume that the time needed for computation and for network communication is negligible.

**解答：**

由上图给出的例子作为解释，这里的同步是指由coordinator服务器发起的。coordinator服务器发送请求到所有slave服务器，接收到所有的slave服务器时间后，计算时间的平均值，然后将这个值回填至所有的服务器。其中也包括coordinator服务器。A服务器即coordinator请求时的时间是08:45，B、C、D、E的服务器时间分别是08：43、08：49、08：42、08：46。所以这4台服务器收到请求后分别返回与coordinator的时间差-2、+4、-3、+1。coordinator接收到这两个值后进行计算$(-2+4-3+1)/3=0$。说明时钟走过的时间为0，可忽略。然后$0-(-2)=2，0-4=-4，0-(-3)=3，0-1=-1$。分别将这四个值回填给另外四台服务器。

#### Bully Algorithm

当任何一个进程发现协作者不再响应请求时，它就发起一次选举。进程P按如下过程主持一次选举：

1. P向所有编号比它大的进程发送一个Election消息；
2. 如果无人响应，P获胜并称为协作者;
3. 如果有编号比它大的进程响应，则由响应者接管选举工作。P的工作完成。

<center>
    <img src="https://raw.githubusercontent.com/HurleyJames/ImageHosting/master/Snipaste_2020-01-15_17-08-42.png" width="80%">
</center>

#### 网络时间协议

网络时间协议（network time protocol, NTP）在服务器之间创建了两条连接。换句话说，B也可以探查A的当前时间。

原则上，对称地应用NTP也可以让B参照A来调整它的时钟。但是，如果已知B的时钟更精确，那么这种调整就不应该了。所以，NTP把服务器分成多个层。含有**参考时钟**（reference clock）的服务器称为**1层服务器**（stratum-1 server）（时钟本身为0层）。

当A与B联系时，如果它的层比B的层要高，那么它就只调整自己的时间。经过同步化后，A将比B高一层。如果B是k层服务器，且A的初始层已经大于k，那么，经过时间调整后，A就变成（k+1）层服务器。由于NTP的对称性，如果A的层数比B的低，那么B将按照A来调整自己。

