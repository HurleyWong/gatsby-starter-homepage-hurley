---
title: 100. Same Tree
tags: [ leetcode ]
date: 2020-02-02T06:25:44.226Z
path: blog/same-tree
cover: ./same-tree.png
excerpt: Give two non-empty linked lists representing two non-negative integers. The digits are stored in reverse order and each of their nodes contain a single digit. Add the two numbers and return it as a linked list.
---

## Same Tree

```
Given two binary trees, write a function to check if they are the same or not.

Two binary trees are considered the same if they are structurally identical and the nodes have the same value.

Example 1:

Input:     1         1
          / \       / \
         2   3     2   3

        [1,2,3],   [1,2,3]

Output: true
Example 2:

Input:     1         1
          /           \
         2             2

        [1,2],     [1,null,2]

Output: false
Example 3:

Input:     1         1
          / \       / \
         2   1     1   2

        [1,2,1],   [1,1,2]

Output: false
```

### 思路

一棵树要么是空树，要么有两个指针，每个指针指向一颗树。树是一种递归结构，很多树的问题都可以使用递归来处理。

判断两个数是否是相同的树的终止条件时：

1. 当两颗数的节点都为`null`时，返回`true`
2. 当两棵树的节点一个为`null`一个不为`null`时，返回`false`
3. 当两个节点都不为`null`但是值不等时，返回`false`
4. 当两个节点都不为`null`且值相等时，递推判断接下来的节点，如果全部相同，则返回`true`

### 代码

```java
public class Same_Tree {

    public boolean isSameTree1(TreeNode p, TreeNode q) {
        if (p == null && q == null) {
            return true;
        }
        if (p == null || q == null) {
            return false;
        }
        if (p.val != q.val) {
            return false;
        }
        // 递推判断
        return isSameTree1(p.right, q.right) && isSameTree1(p.left, q.left);

    }

    // 定义树结构
    class TreeNode {
        int val;
        TreeNode left;
        TreeNode right;

        TreeNode(int x) {
            val = x;
        }
    }

}
```

#### 复杂度分析

* **时间复杂度：**$O(n)$，n为节点的数量，因为每个节点都要判断是否相同。
* **空间复杂度：**在最优情况下（完全平衡二叉树）时为$O(log(n))$，最坏情况下（完全不平衡二叉树）时为$O(n)$。

#### 完全二叉树和平衡二叉树

##### 完全二叉树

只有树最下面的两层的节点度小于2，并且最下面一层的节点都集中在该层的最左边的若干位置的二叉树。

<img src="https://raw.githubusercontent.com/HurleyJames/ImageHosting/master/%E5%AE%8C%E5%85%A8%E4%BA%8C%E5%8F%89%E6%A0%91.png" width="100%" />

##### 平衡二叉树

平衡二叉树是为了保证树不至于太倾斜。所以定义如下：

平衡二叉树要么是一颗空树，要么保证左右子树的高度之差不大于1，同时子树也必须是一颗平衡二叉树。

