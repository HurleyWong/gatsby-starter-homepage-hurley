---
title: 543. Diameter of Binary Tree
tags: [ leetcode ]
date: 2020-02-03T06:25:44.226Z
path: blog/diameter-of-binary-tree
cover: ./diameter-of-binary-tree.png
excerpt: Given a binary tree, you need to compute the length of the diameter of the tree. The diameter of a binary tree is the length of the longest path between any two nodes in a tree. This path may or may not pass through the root.
---

## Diameter of Binary Tree

```
Given a binary tree, you need to compute the length of the diameter of the tree. The diameter of a binary tree is the length of the longest path between any two nodes in a tree. This path may or may not pass through the root.

Example:
Given a binary tree
          1
         / \
        2   3
       / \     
      4   5    
Return 3, which is the length of the path [4,2,1,3] or [5,2,1,3].
```

**Note:** The length of path between two nodes is represented by the number of edges between them.

### 思路

关于二叉树直径的定义是：**二叉树中从一个结点到另一个结点最长的路径**，叫做二叉树的直径。

这里存在一个陷阱，就是容易受到题目中例子的影响，认为二叉树的直径就是左子树的深度+右子树的深度。实际上，二叉树的直径**不一定经过根节点root**。有个例子如下图所示：

<img src="https://raw.githubusercontent.com/HurleyJames/ImageHosting/master/IMG_8D22A19495F3-1.jpeg" width="100%" />

所以，采用**分治**和**递归**的思想：二叉树的直径=（左子树的直径，右子树的直径，左子树的最大深度+右子树的最大深度+1）。

### 代码

```java
 public class Diameter_of_Binary_Tree {
 
     int diameter = 0;
 
     public int diameterOfBinaryTree(TreeNode root) {
         depth(root);
         return diameter;
     }
 
     private int depth(TreeNode root) {
         if (root == null) {
             return 0;
         }
         // 获得左子树的深度
         int left = depth(root.left);
         // 获得右子树的深度
         int right = depth(root.right);
         diameter = Math.max(diameter, left + right);
         return Math.max(left, right) + 1;
     }
 
     class TreeNode {
         TreeNode left;
         TreeNode right;
         int val;
 
         TreeNode(int x) {
             val = x;
         }
     }
 }
```

#### 复杂度分析

- **时间复杂度：**因为这题需要对左、右子树都进行递归遍历操作，所以每个节点都要访问一次，因为是。
- **空间复杂度：**主要是进行DFS深度优先搜索的栈开销，为。

#### 