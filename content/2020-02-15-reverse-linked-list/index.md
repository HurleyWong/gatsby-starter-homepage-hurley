---
title: 206. Reverse Linked List
tags: [ leetcode ]
date: 2020-02-15T06:25:44.226Z
path: blog/reverse-linked-list
cover: ./reverse-linked-list.png
excerpt: Reverse a singly linked list.
---

## Reverse Linked List

```
Reverse a singly linked list.

Example:

Input: 1->2->3->4->5->NULL
Output: 5->4->3->2->1->NULL
```

### 方法一：递归法

#### 思路

因为之前做了比较多树的题目，发现树的题目都是用递归遍历的方式比用迭代的方式要简单的多。但是对于这题，递归反而更难理解。

这题的递归主要是通过一个判断条件，当当前节点或者当前节点的下一个节点为`null`时，就改变节点的指向，将head的下一个节点指向head，具体是用如下一句代码

```java
head.next.next = head;
```

我们先看一个动态图来知道大概流程，然后再具体分析代码是如何实现的。

<img src="https://raw.githubusercontent.com/HurleyJames/ImageHosting/master/dacd1bf55dec5c8b38d0904f26e472e2024fc8bee4ea46e3aa676f340ba1eb9d-%E9%80%92%E5%BD%92.gif" width="100%" />

#### 代码

```java
public ListNode reverseList(ListNode head) {
    // 方法一：递归
    // 如果头结点为空，或者只有一个头结点，那么翻转过来就是头结点本身
    // 终止条件是，当前节点或者下一个节点为 null
    if (head == null || head.next == null) {
        return head;
    }
    ListNode p = reverseList(head.next);
    // 改变节点的指向
    head.next.next = head;
    head.next = null;
    return p;
}
```

我们可以根据代码和上面的动图来一步一步走一下这个程序。首先，这句代码`ListNode p = reverseList(head.next)`采用了递归的方式，假设这个链表是`1->2->3->4->5`，那么`head`最开始是1，`head.next`则是2，那么上面这句递归的代码就跳转去执行`reverseList(head.next)`即`reverseList(2)`，这样递归下去最终会执行`reverseList(4.next)`即`reverseList(5)`。因为当头结点为5时，`5.next`为`null`，所以满足第一行代码`if`语句的终止条件，就会返回`head`即返回5。

然后跳出这最后一层递归，即执行完了`reverseList(5)`，去接着执行`reverseList(4)`。这里有一句重要的**改变节点指向**的代码`head.next.next = head`。我们知道这时候的head是4，所以这句代码其实就是`4.next.next = 4`，而`4.next`在该链表中即为5，所以最终就是`5.next = 4`，即`5->4`，5的下一个节点又指向了4。

这里再注意题目的要求是翻转链表。而我们经过上面的操作后就变成了`4->5`，而且`5->4`，这就变成了双向链表了，所以我们要解除4指向5的关系，就通过这句代码`head.next =  null`，即`4.next  = null`，就把这个关系解除了（具体可以通过观察动图来理解）。

这样最终会返回原链表的头结点即1，然后头结点的下一个节点为`null`，就结束翻转了。

##### 复杂度分析

* **时间复杂度**：因为将链表从头走到尾，所以时间复杂度为$O(n)$，n为链表的长度。
* **空间复杂度**：因为这个方法使用了递归，递归会使用到**隐式栈空间**，所以递归的深度可能会达到n层，所以是$O(n)$。

### 方法二：双指针迭代法

#### 思路

我们可以申请两个指针，`prev`和`curr`。`prev`最初指向`null`，而`curr`指向`head`。然后遍历`curr`，并通过一个临时指针`temp`来储存`curr`的下一个节点即`curr.next`，然后让这个临时指针记录下一个节点`temp = curr.next`，然后让`curr`指向`prev`。最后继续遍历，让`prev`和`curr`都向前进一位，`prev = curr; curr = temp`。

具体演示效果如下所示：

<img src="https://raw.githubusercontent.com/HurleyJames/ImageHosting/master/7d8712af4fbb870537607b1dd95d66c248eb178db4319919c32d9304ee85b602-%E8%BF%AD%E4%BB%A3.gif" width="100%" />

#### 代码

```java
public ListNode reverseList(ListNode head) {

    // 方法二：迭代
    // 申请节点，pre 和 curr，pre 指向 null
    ListNode prev = null;
    ListNode curr = head;
    ListNode tmp = null;
    while (curr != null) {
        // 记录当前节点的下一个节点
        tmp = curr.next;
        // 然后将当前节点指向 pre
        curr.next = prev;
        // pre 和 curr 节点都前进一位
        prev = curr;
        curr = tmp;
    }
    return prev;
}
```

##### 复杂度分析

* **时间复杂度**：因为这个过程同样是将链表从头遍历到尾，所以时间复杂度为$O(n)$。
* **空间复杂度**：$O(1)$。