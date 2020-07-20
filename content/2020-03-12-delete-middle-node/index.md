---
title: 面试题02.03.Delete Middle Node
tags: [ leetcode ]
date: 2020-03-12T06:25:44.226Z
path: blog/delete-middle-node
cover: ./delete-middle-node.png
excerpt: Delete middle node in singly linked list.
---

```
实现一种算法，删除单向链表中间的某个节点（除了第一个和最后一个节点，不一定是中间节点），假定你只能访问该节点。

示例：

输入：单向链表a->b->c->d->e->f中的节点c
结果：不返回任何数据，但该链表变为a->b->d->e->f
```

### 方法一：暴力法

最简单直接的想法就是遍历一遍链表，得出链表的长度，然后一半即是链表一半的长度，也就是中间节点所在的位置。但是这样缺点也是很明显的，就是在时间复杂度上。假设链表的长度为n，则遍历一遍链表的长度需要$O(n)$的时间，然后又重新遍历一半链表的长度获得中间节点，这里又需要$O(\frac{n}{2})$的时间，总时间即为$O(\frac{3n}{2})$。

<!-- more -->

### 方法二：快慢指针法

在很多场景下都可以用到快慢指针法。在这里设置两个指针，一个指针A每次指向下一个节点，一个指针B指向下两个节点，这样当B走到链表的末尾时，A才走到链表的中间节点（因此B是A速度的两倍）。所以这样A指针走到的位置就是链表的中间节点。

#### 代码

```java
public class Delete_Middle_Node {

    public void deleteNode(ListNode node) {
        // 慢指针
        node.val = node.next.val;
        // 快指针
        node.next = node.next.next;
    }

    class ListNode {
        int val;
        ListNode next;
        ListNode(int x) {
            val = x;
        }
    }
}

```

#### 复杂度

* **时间复杂度**：因为两个指针是同时进行的，快指针遍历了整个链表，慢指针遍历了半个链表，所以时间复杂度仍然是$O(n)$。