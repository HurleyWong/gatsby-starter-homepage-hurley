---
title: 21. Merge Two Sorted Lists
tags: [ leetcode ]
date: 2020-02-15T06:25:44.226Z
path: blog/merge-two-sorted-lists
cover: ./merge-two-sorted-lists.png
excerpt: Merge two sorted linked lists and return it as a new sorted list. The new list should be made by splicing together the nodes of the first two lists.
---

## Merge Two Sorted Lists

```
Merge two sorted linked lists and return it as a new sorted list. The new list should be made by splicing together the nodes of the first two lists.

Example:

Input: 1->2->4, 1->3->4
Output: 1->1->2->3->4->4
```

### 方法一：递归法

#### 思路

这道题的递归法同样不好理解。之前我一直认为递归是一个比较容易的方法，而迭代更难。看来面对不同的数据结构其实并不相同，树由于其特殊性，采用递归即深度遍历的方式是十分好理解的，而采用迭代则必须要用一个栈或者队列去保存元素反而更加繁琐。但是对于链表这种数据结构，递归反而值得更多的思考。

首先，这道题采用递归的终止条件是当`l1`或者`l2`为空时，结束。而返回值则是**每一层调用都返回排序好的链表头**。

通俗来说，就是如果`l1.val`比`l2.val`更小，那么就将`l1.next`与排序好的链表头相接；反之，如果`l2.val`更小，则将`l2.next`与排序号的链表头相连。具体是通过`l1.next = mergeTwoLists(l1.next, l2)`这句代码来实现的。

具体过程如下（但实际上仍然不是很好理解，具体还是要分析代码）：

<img src="https://raw.githubusercontent.com/HurleyJames/ImageHosting/master/s1.png" width="100%" />

<img src="https://raw.githubusercontent.com/HurleyJames/ImageHosting/master/s2.png" width="100%" />

<img src="https://raw.githubusercontent.com/HurleyJames/ImageHosting/master/s3.png" width="100%" />

通过以上图片过程可以发现，`l2.next = merge(l1, l2.next)`这句代码就是让原有的`l2.next`指向的那个链条断开，指向了新的`merge(l1, l2.next)`。虽然目前我们暂时不知道这个`merge(l1, l2.next)`是什么，但是这其实是一个持续递归的函数，最终会返回已经排序好的值。

#### 代码

```java
public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
    if (l1 == null && l2 == null) {
        return null;
    }
    if (l1 == null) {
        return l2;
    }
    if (l2 == null) {
        return l1;
    }

    if (l1.val < l2.val) {
        // 如果 l1 的 val 更小，则将 l1.next 等于排序好的链表头
        l1.next = mergeTwoLists(l1.next, l2);
        return l1;
    } else {
        // 如果 l2 的 val 更小，则将 l2.next 等于排序号的链表头
        l2.next = mergeTwoLists(l1, l2.next);
        return l2;
    }
}
```

我们通过一个例子来走一遍整个代码的流程。

##### 例子

现在有两个链表`l1`和`l2`，分别为`1->2->5`和`0->3->4`。

<img src="https://raw.githubusercontent.com/HurleyJames/ImageHosting/master/IMG_0228.jpg" width="100%" />

<img src="https://raw.githubusercontent.com/HurleyJames/ImageHosting/master/IMG_0229.jpg" width="100%" />

##### 复杂度分析

* **时间复杂度**：很明显，这个过程要把两个链表都走一遍。假设它们的长长度分别为m和n，则时间复杂度为$O(m+n)$。
* **空间复杂度**：因为这个过程会调用$m+n$个栈，所以会消耗$O(m+n)$的空间。

### 方法二：迭代法

#### 思路

关于链表的合并问题，我们都可以想到去设置一个**哨兵节点**。比如，链表`l1`和`l2`分别为`1->2->4`和`1->3->4`，然后设置一个`prehead`的哨兵节点，如下所示：

<img src="https://raw.githubusercontent.com/HurleyJames/ImageHosting/master/IMG_0230.jpg" width="100%" />

然后我们主要关注`prehead`节点，调整它的`next`指针，让它总是指向`l1`和`l2`中较小的那个节点，直到两个链表中的某一条指向`null`为止。

我在这里把步骤一步一步地写出来：

<img src="https://raw.githubusercontent.com/HurleyJames/ImageHosting/master/IMG_0232.jpg" width="100%" />

<img src="https://raw.githubusercontent.com/HurleyJames/ImageHosting/master/IMG_0233.jpg" width="100%" />

完整的过程如下：

<img src="https://raw.githubusercontent.com/HurleyJames/ImageHosting/master/IMG_0231.jpg" width="100%" />

#### 代码

```java
public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
    // 2.迭代法
    ListNode prehead = new ListNode(-1);

    ListNode prev = prehead;
    while (l1 != null && l2 != null) {
        if (l1.val <= l2.val) {
            prev.next = l1;
            l1 = l1.next;
        } else {
            prev.next = l2;
            l2 = l2.next;
        }
        prev = prev.next;
    }

    if (l1 == null) {
        prev.next = l2;
    } else {
        prev.next = l1;
    }

    return prehead.next;
}
```

迭代法的算法看起来是十分清晰明了的，如果`l1.val <= l2.val`，则让哨兵节点指向`l1`即值更小的节点，反之一样，然后再将那个更小的节点往后走一位，再重新判断值的大小。

##### 复杂度分析

* **时间复杂度**：因为这个方法要比较两条链表的每一个节点，所以时间复杂度为$O(m+n)$，即循环的次数等于两个链表的总长度。
* **空间复杂度**：迭代的过程会产生几个指针，所以所需空间是常数级别的，为$O(1)$。