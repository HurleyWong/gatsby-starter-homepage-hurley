---
title: 226. Invert Binary Tree
tags: [ leetcode ]
date: 2020-02-03T06:25:44.226Z
path: blog/invert-binary-tree
cover: ./invert-binary-tree.png
excerpt: Invert a binary tree.
---

## Invert Binary Tree

```
Invert a binary tree.

Example:

Input:

     4
   /   \
  2     7
 / \   / \
1   3 6   9
Output:

     4
   /   \
  7     2
 / \   / \
9   6 3   1
```

### 递归法

通过观察输入和输出，可以发现就是把所有的子树的左右结点都互换位置。所以可以使用**递归**的方法交换左节点和右节点。

```java
public class Invert_Binary_Tree {

    public TreeNode invertTree(TreeNode root) {
        if (root == null) {
            return null;
        }
        // 交换左右子树节点
        swap(root);
        // 递归当前节点的左子树
        invertTree(root.right);
        // 递归当前节点的右子树
        invertTree(root.left);
        return root;
    }

    /**
     * 交换左右子树节点
     *
     * @param root
     */
    public void swap(TreeNode root) {
        TreeNode tmp = root.left;
        root.left = root.right;
        root.right = tmp;
    }

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

关键的三行代码就是`swap(root)`和`invertTree(root.left)`以及`invertTree(root.right)`。

#### 复杂度分析

* **时间复杂度：**因为要递归所有的左子树节点和右子树节点，所有时间复杂度为$O(n)$。
* **空间复杂度：**假设树的高度为h，则最坏情况下需要$O(h)$个函数存放。

### 迭代法

递归的实现方式其实就是深度优先搜索BFS的方式，而迭代法就是广度优先搜索DFS的方式。

广度优先搜索需要额外的数据结构——队列，来存放临时遍历的元素。

首先将根节点放入到队列中，然后对当前元素调换其左右子树的位置，然后再判断其左子树是否为空，不为空就放入队列中；然后判断其右子树是否为空，不为空就放入到队列中。

```java
public class Invert_Binary_Tree {

    public TreeNode invertTree(TreeNode root) {
        if (root == null) {
            return null;
        }
        LinkedList<TreeNode> queue = new LinkedList<>();
        queue.add(root);
        while(!queue.isEmpty()) {
            // 每次从队列中拿出一个节点，并交换这个节点的左右子树
            TreeNode tmp = queue.poll();
            swap(tmp);
            // 如果当前节点的左子树不为空，则放入队列等待后续处理
            if (tmp.left != null) {
                queue.add(tmp.left);
            }
            // 如果当前节点的右子树不为空，则放入队列等待后续处理
            if (tmp.right != null) {
                queue.add(tmp.right);
            }
        }
        return root;
    }

    /**
     * 交换左右子树节点
     *
     * @param root
     */
    public void swap(TreeNode root) {
        TreeNode tmp = root.left;
        root.left = root.right;
        root.right = tmp;
    }

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

动态图如下：

<img src="https://raw.githubusercontent.com/HurleyJames/ImageHosting/master/f9e06159617cbf8372b544daee37be70286c3d9b762c016664e225044fc4d479-226_%E8%BF%AD%E4%BB%A3.gif" width="100%" />

#### 复杂度分析

- **时间复杂度：**因为每个节点都要判断是否子树为空，即每个节点都被入队出队一次，所以时间复杂度为。
- **空间复杂度：**在最坏的情况下，队列里会包含树中的所有的节点。而如果是一颗完整二叉树，那么叶子节点那一层就拥有个节点。