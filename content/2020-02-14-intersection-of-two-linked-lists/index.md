---
title: 160. Intersection of Two Linked Lists
tags: [ leetcode ]
date: 2020-02-14T06:25:44.226Z
path: blog/intersection-of-two-linked-lists
cover: ./intersection-of-two-linked-lists.png
excerpt: Write a program to find the node at which the intersection of two singly linked lists begins.
---

## Intersection of Two Linked Lists

### 方法一：暴力法

#### 思路

非常容易想到的就是暴力法。采用双重遍历的方式，先从链表A中选出一个节点，然后遍历整个链表B，看是否能找到与之相同的节点。如果能的话，就返回该节点；如果不能，则继续遍历链表A，然后重复。

#### 代码

```java
public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
    // 如果某个链表为空，是肯定不会相交的
    if (headA == null || headB == null) {
        return null;
    }
    while (headA != null) {
        ListNode newHead = headB;
        while (newHead != null) {
            // 如果相等，就找到了相交点
            if (headA == newHead) {
                return newHead;
            } else {
                newHead = newHead.next;
            }
        }
        headA = headA.next;
    }
    // 如果遍历完了都没有找到，说明没有相交点
    return null;
}
```

虽然这种方法思路上十分简单清晰，但是在代码中仍然有需要注意的地方。

1. 这里判断节点相同，可以直接采用`nodeA == nodeB`的方式去判断，而不是通过他们的值`val`去判断。因为，这里仅仅是值相等并没有用，必须要它们的下一个节点以及之后的节点都相等，才是找到了相交节点。

2. 在第一层循环的里面，我们又定义了`ListNode newHead = headB`。这一步看起来是多余的，但是如果不这么定义一个新的变量的话，那么在第二层循环的判断条件那里，就会变成是`while (headB != null)`。那么当第一遍遍历整个链表B却没有找到与链表A中一个节点$a_i$相同的节点的话，`headB = headB.next`最终会使得`headB == null`，即遍历到链表B的末尾。这样就没有再为链表B从头开始遍历了。

    所以，必须要使用其它元素来保存遍历的链表B的节点。这里用的是`ListNode newHead = headB`。

##### 复杂度分析

* **时间复杂度**：因为这里采用的双重循环遍历两个链表。假设链表A的长度为m，链表B的长度为n，所以时间复杂度为$O(mn)$。
* **空间复杂度**：$O(1)$。

### 方法二：哈希法

#### 思路

先遍历链表A，把链表A的所有节点放入一个set中。然后再遍历链表B，判断如果链表B中的某个节点出现在set中，那么这就是相交节点。

#### 代码

```java
public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
    Set s = new HashSet();
    // 遍历链表A，把链表A的节点全部存入set中
    while (headA != null) {
        s.add(headA);
        headA = headA.next;
    }
    while (headB != null) {
        // 遍历链表B，判断set中是否存在相同的节点
        if (s.contains(headB)) {
            return headB;
        }
        headB = headB.next;
    }
    return null;
}
```

##### 复杂度分析

* **时间复杂度**：因为这里用到的HashSet的底层是通过HashMap来实现的，所以进行`add`或者`contains`等操作的时间复杂度都是$O(1)$。因为进行了两次不是嵌套的循环，所以假设链表A的长度为m，链表B的长度为n，则时间复杂度为$O(m+n)$。
* **空间复杂度**：因为这里用了HashSet去存储节点，所以要么存储了链表A的长度要么存储了链表B的长度。所以空间复杂度为$O(m)$或者$O(n)$。

### 方法三：双指针法

#### 思路

在LeetCode的评论区，看到很多人评论这是一种很浪漫的方法。即

<font color="red">错的人迟早会走散，而对的人迟早会相逢！</font>

双指针的方式其实思考起来还是很清晰的。用两个指针a和b分别从链表A和链表B开始遍历，当a遍历完链表A之后，就去遍历链表B；同样的，当b遍历完链表B之后，就去遍历链表A。这样，如果它们是相交的，则最终走过的长度肯定是一样的，即会在交点相遇。可以用下图一样，将两个链表连成一个环。

<img src="https://raw.githubusercontent.com/HurleyJames/ImageHosting/master/6d24c0d2f451f8cfccea0edaff474d5d1e834d2199272974915d80e332f5fb50-1571538464(1).jpg" width="100%" />

然后遍历过程如下动图所示：

<img src="https://raw.githubusercontent.com/HurleyJames/ImageHosting/master/396526c47e043feb977e59f98d8df9165ae249d5042ca60ee4d3121c05fea067-%E5%8A%A8%E6%80%81%E5%9B%BE.gif" width="100%" />

因为连成一个环后，假设有相交的节点，则最终走过的链表A加上链表B的长度是一样，最终都会相遇于交点。

（今天**情人节**，值得反思一下🤔）

#### 代码

```java
public ListNode getIntersectionNode(ListNode headA, ListNode headB) {
    ListNode a = headA;
    ListNode b = headB;
    while (a != b) {
        if (a != null) {
            // 继续遍历链表A
            a = a.next;
        } else {
            // 去遍历链表B，从头结点开始
            a = headB;
        }

        if (b != null) {
            // 继续遍历链表B
            b = b.next;
        } else {
            // 去遍历链表A，从头节点开始
            b = headA;
        }
    }
    return a;
}
```

##### 复杂度分析

* **时间复杂度**：因为这个方法同样遍历了链表A和链表B，所以时间复杂度为$O(m+n)$。
* **空间复杂度**：这里没有去存储节点，所以空间复杂度为$O(1)$。

