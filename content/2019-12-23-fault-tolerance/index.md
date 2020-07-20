---
title: Fault Tolerance
tags: [ CloudComputing ]
date: 2019-12-23T06:25:44.226Z
path: blog/fault-tolerance
cover: ./fault-tolerance.png
excerpt: Fault tolerance is the property that enables a system to continue operating properly in the event of the failure of some of its components.
---

## 容错性描述

容错与可靠性（dependability）紧密相关。

* **可用性**(availability)说明系统已准备好，马上就可以使用
* **可靠性**(reliability)指系统可以无故障地持续运行
* **安全性**(safety)指在系统偶然出现故障的情况下，仍然能够正确的操作而不会造成任何灾难
* **可维护性**(maintainability)是指发生故障的系统被恢复的难易程度

<!-- more -->

### Terminology

* **Fail**：当一个系统不能兑现它的承诺时就被认为是失败了
* **Error**：是系统状态的一部分，它可能会导致失败
* **Fault**：是造成Error的原因

### Handling Faults

* Fault prevention：Prevent the occurrence of a fault
* Fault tolerance：Build a component such that it can mask the occurrence of a fault
* Fault removal：Reduce thte presence, number, or seriousness of a fault
* Fault forecasting：Estimate current presence, future incidence, and consequences of faults

### Failure Models

1. 崩溃性故障 Crash failure

    服务器停机，但是在停机之前工作正常

2. 遗漏性故障 Omission failure

    服务器不能响应到来的请求

    * 接收故障 Receive omission

        服务器不能接受到来的请求

    * 发送故障 Send omission

        服务器不能发送消息

3. 定时故障 Timing failure

    服务器的响应在指定的时间间隔之外

4. 响应故障 Response failure

    服务器的响应不正确

    * 值故障 Value failure

        响应的值错误

    * 状态转换故障 State-transition failure

        服务器偏离了正确的控制流

5. 随意性故障 Arbitrary (or Byzantine) failure

    服务器可能在随意的时间产生随意的响应

### 冗余掩盖故障 Failure Masking by Redundancy

* 信息冗余 Information redundancy
* 时间冗余 Time redundancy
* 物理冗余 Physical redundancy

## 进程恢复 Process Resilience

### 平等组与等级组

<center>
    <img src="https://raw.githubusercontent.com/HurleyJames/ImageHosting/master/fair.png" width="60%">
</center>

**平等组**是对称的，没有单独的失败点。如果一个进程崩溃，组只是简单地变得更小，但是还可以继续。它的缺点是做出决定比较复杂，比如需要进行表决，会导致一些延迟和开销。

**等级组**则相反。某个成员的故障会使整个组崩溃，但是只要它保持运行，就可以独自做出决定，不需要其他进程参加。

### 故障掩盖和复制

如果系统能够经受k个组件的故障并且还能满足规范的要求，那么就被称为**k容错**（k fault tolerant）。

如果这些进程失败了，那么k+1个组件就能提供k容错。

另一方面，如果进程发生**拜占庭失败**，继续错误运行并发送出错误或随机的应答，最少需要2k+1个进程才能获得k容错。

## 拜占庭问题

全称是**拜占庭将军问题**（Byzantine Generals Problem），是由莱斯利·兰波特提出的分布式对等网络通信容错问题。在分布式计算中，不同的计算机通过通讯交换信息达成共识而按照同一套协作策略行动。但有时，系统中的成员计算机可能出错而发送错误的信息，用于传递信息的通讯网络也可能导致信息损坏，使得网络中不同的成员关于全体协作的策略得出不同的结论，从而破坏系统一致性。拜占庭问题被认为是容错性问题中最难的问题类型之一。

> 一组拜占庭将军分别各率领一支军队共同围困一座城市。为了简化问题，将各支军队的行动策略限定为进攻或撤离两种。因为部分军队进攻部分军队撤离可能会造成灾难性后果，因此各位将军必须通过投票来达成一致策略，即所有军队一起进攻或所有军队一起撤离。因为各位将军分处城市不同方向，他们只能通过信使互相联系。在投票过程中每位将军都将自己投票给进攻还是撤退的信息通过信使分别通知其他所有将军，这样一来每位将军根据自己的投票和其他所有将军送来的信息就可以知道共同的投票结果而决定行动策略。

系统的问题在于，将军中可能出现叛徒，他们不仅可能向较为糟糕的策略投票，还可能选择性地发送投票信息。**假如那些忠诚（没有出错）的将军仍然能通过多数决定来决定他们的策略**，那么就达到了**拜占庭容错**。

上述故事映射到计算机系统里，将军就是计算机，信差就是通信系统。在分布式对等网络中需要按照共同一致策略协作的**成员计算机**即为问题中的**将军**，而各成员赖以进行通讯的**网络链路**即为**信使**。拜占庭将军描述的就是某些成员计算机或网络链路出现错误、甚至被蓄意破坏者控制的情况。

### 解决方案

#### 问题分析

如何让忠诚者（非叛徒）达成一致。

假设节点总数为N，叛徒数为F，则当**N>=3F+1**时，问题才有解，即**Byzantine Fault Tolerant(BTF)**算法。

#### 例子

N=3, F=1, 不满足N>=3F+1

1. 当提出方案的人**不是叛徒**时，提案人提出一个方案，叛徒就提出相反的方案，剩下一个人收到两个相反的意见，就无法判断谁是叛徒，也无法给出一致的意见。所以如果提案人提出方案，系统中就有N-F份确定的信息和F份不确定的信息，只有$N-F \ge F \Rightarrow N>F$的情况下达成一致。
2. 当提出方案的人是**叛徒**时，提案人提出方案，发送给另外两人。另外两人收到两份相反的消息，无法判断谁才是叛徒，系统也无法达成一致。因为提出方案的叛徒会尽量发送相反的消息给N-F个忠诚者，那么$\begin{cases}\frac{N-F}{2}个信息1\\ \frac{N-F}{2}个信息0\end{cases}$

Leslie Lamport证明，当叛徒数不超过**1/3**时，存在有效的算法让忠诚者总能达成一致。然而，如果叛徒数过多，就无法保证能够达成一致。

#### 假设

如果叛徒数超过1/3时，有无解决方案？

设有f个叛徒和g个忠诚者，叛徒可以故意使坏，可以给出错误的结果也可以不响应请求。

1. 当f个叛徒不响应，则g个忠诚者占多数，仍然能够得到正确结果
2. 当f个叛徒，每个叛徒都给出一个恶意提案，并且当g个忠诚者中有f个处于离线状态时，则剩下g-f个忠诚者想要占据多数保持正确结果，则必须有$g-f>f \Rightarrow g>2f$，而系统的整体规模：$g+f > 2f+f=3f \Rightarrow g+f>3f$