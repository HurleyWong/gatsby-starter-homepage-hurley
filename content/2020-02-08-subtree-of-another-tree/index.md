---
title: 572. Subtree of Another Tree
tags: [ leetcode ]
date: 2020-02-08T06:25:44.226Z
path: blog/subtree-of-another-tree
cover: ./subtree-of-another-tree.png
excerpt: Given two non-empty binary trees s and t, check whether tree t has exactly the same structure and node values with a subtree of s. A subtree of s is a tree consists of a node in s and all of this node's descendants. The tree s could also be considered as a subtree of itself.
---

## Subtree of Another Tree

```
Given two non-empty binary trees s and t, check whether tree t has exactly the same structure and node values with a subtree of s. A subtree of s is a tree consists of a node in s and all of this node's descendants. The tree s could also be considered as a subtree of itself.

Example 1:
Given tree s:

     3
    / \
   4   5
  / \
 1   2
Given tree t:
   4 
  / \
 1   2
Return true, because t has the same structure and node values with a subtree of s.
 

Example 2:
Given tree s:

     3
    / \
   4   5
  / \
 1   2
    /
   0
Given tree t:
   4
  / \
 1   2
Return false.
```

### 方法一：先序遍历

因为我们知道树的表示方式的一种就是以先序遍历的方式表示出来。所以，我们把两棵树s和t分别以先序遍历的方式表示（以字符串表示），然后判断t是否是s的子字符串即可判断出是否是其的子树。

通常，如果左节点或者右节点为空，我们就会把它以`null`的形式表示，但是在这里是要判断结构和节点是否相同，所以不能简单地用`null`来区分。

当左孩子为空时，要赋值为`lnull`；当右孩子为空时，要赋值为`rnull`。

还有一点非常重要的是，因为我们是把它转化为字符串，用字符串`contains`的方式来判断是否是子串，这样容易把更小的数字认为是更大的数字的子串。例如，会把`3`认为是`23`的子串，但是在这里显然是不同的，如果一个是`3`，一个是`23`，那么就不是子树了。所以，我们要为每个节点前加一个`#`，就可以解决这个问题了。

关于以上`contains`方法的问题，如果含有空左孩子或右孩子时，即有`lnull`或者`rnull`时也不会有问题，但是如果没有，那么当两棵树为`[12]`和`[2]`时，就会输出`true`，然而实际是`false`。**所以一定要为数字前加上一个字符**。

#### 代码

```java
/**
 * Given two non-empty binary trees s and t, check whether tree t has exactly the same structure and node values with a subtree of s. A subtree of s is a tree consists of a node in s and all of this node's descendants. The tree s could also be considered as a subtree of itself.
 */
public class Subtree_of_Another_Tree {

    public boolean isSubtree(TreeNode s, TreeNode t) {
        String tree1 = preOrder(s, true);
        String tree2 = preOrder(t, true);
        if (tree1.contains(tree2)) {
            return true;
        } else {
            return false;
        }
    }

    private String preOrder(TreeNode node, boolean left) {
        if (node == null) {
            // 如果是左孩子节点为空
            if (left) {
                // 设置为 lnull
                return "lnull";
            } else {
                // 右孩子节点为空，设置为 rnull
                return "rnull";
            }
        }
        // 给每个节点前加上 # 号
        // 如果 preOrder 的第一个参数是 left，那么就是 true
        // 如果 preOrder 的第一个参数是 right，那么就是 false
        return "#" + node.val + " " + preOrder(node.left, true) + " " + preOrder(node.right, false);
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

##### 复杂度分析

* **时间复杂度**：采用先序遍历的方式表示树，那么两棵树的花费的时间为$O(m)$和$O(n)$。最后用字符串判断是否包含的方式花费的时间为$O(mn)$，所以总的时间复杂度为$O(m+n+mn)$。
* **空间复杂度**：主要取决于哪棵树的空间更大。所以是$O(max(m,n))$。

### 方法二：比较节点

我们可以把每个给定节点t的子树都作为根，然后判断以t为根的子树是否与给定的子树相同。为了检查是否完全相同，我们就需要比较两个子树的所有节点。

首先，我们定义一个`equals(x,y)`函数去检查两个树是否相等。它先检查两个树的根是否相等，然后再递归判断左子树和右子树。

然后，使用一个函数`traverse(s,t)`，遍历给定的树s并将每个节点都当作子树的根。

#### 代码

```java
public class Subtree_of_Another_Tree {

    public boolean isSubtree(TreeNode s, TreeNode t) {
        return traverse(s, t);
    }

    private boolean traverse(TreeNode s, TreeNode t) {
        return s != null && (equals(s, t) || traverse(s.left, t) || traverse(s.right, t));
    }

    /**
     * 比较两个树是否相同
     *
     * @param t1
     * @param t2
     * @return
     */
    private boolean equals(TreeNode t1, TreeNode t2) {
        if (t1 == null && t2 == null) {
            return true;
        }
        if (t1 == null || t2 == null) {
            return false;
        }
        // 如果节点值相同，且左孩子和右孩子的节点值也相同，则返回 true
        return t1.val == t2.val && equals(t1.left, t2.left) && equals(t1.right, t2.right);
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

##### 复杂度分析

* **时间复杂度**：在最坏的情况下，即一棵树为倾斜树时，需要$O(mn)$时间。
* **空间复杂度**：如果n为树的节点数，那么空间复杂度为$O(n)$。

