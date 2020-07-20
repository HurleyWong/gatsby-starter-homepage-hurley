---
title: 437. Path Sum III
tags: [ leetcode ]
date: 2020-02-07T06:25:44.226Z
path: blog/path-sum-iii
cover: ./path-sum-iii.png
excerpt: You are given a binary tree in which each node contains an integer value. Find the number of paths that sum to a given value.
---

## Path Sum III

```
You are given a binary tree in which each node contains an integer value.

Find the number of paths that sum to a given value.

The path does not need to start or end at the root or a leaf, but it must go downwards (traveling only from parent nodes to child nodes).

The tree has no more than 1,000 nodes and the values are in the range -1,000,000 to 1,000,000.

Example:

root = [10,5,-3,3,2,null,11,3,-2,null,1], sum = 8

      10
     /  \
    5   -3
   / \    \
  3   2   11
 / \   \
3  -2   1

Return 3. The paths that sum to 8 are:

1.  5 -> 3
2.  5 -> 2 -> 1
3. -3 -> 11
```

### 思路

这道题的描述中没有要求路径的开头必须是根节点，结尾也没有要求是叶子节点，只要求了是从上往下。

所以，所有情况就分为以根节点开始的，和以根节点的左孩子和右孩子开始这三种。

具体过程同样是采取了递归的思想，当找到一条值等于sum，就让路径树加1。但需要注意的是递归的循环中，应该是`pathSum(root.left/right, sum - root.val)`这种形式。因为在递归后，后面一个参数应该从`sum`变成`sum - root.val`，因为已经经过了一个节点，需要减去这个节点的值再进行递归。

### 代码

```java
public class Path_Sum_3 {

    public int pathSum(TreeNode root, int sum) {
        if (root == null) {
            return 0;
        }
        // 找出根节点的所有路径，再找出以根节点的左孩子和右孩子开始的所有路径
        return path(root, sum) + pathSum(root.left, sum) + pathSum(root.right, sum);
    }

    public int path(TreeNode root, int sum) {

        int pathSum = 0;
        if (root == null) {
            return 0;
        }
        if (root.val == sum) {
            pathSum++;
        }

        pathSum = pathSum + path(root.left, sum - root.val);
        pathSum = pathSum + path(root.right, sum - root.val);

        return pathSum;

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

这里刚开始有些疑惑的是为什么一个方法的返回值是`return path(root, sum) + pathSum(root.left, sum) + pathSum(root.right, sum);`，而不是都调用`path`函数。

我认为实际上这是把`path(root, sum) + pathSum(root.left, sum) + pathSum(root.right, sum)`看成一个整体，分别是根节点root和它的左孩子A和右孩子B。第一个`path(root, sum)`是为了找出以根节点为路径开头的路径数量，第二个`pathSum(root.left, sum)`就是以左孩子节点作为新的根节点，然后递归，又以这个A作为根节点，找出它作为路径开头的路径数量，依次递归下去。右孩子B同理。

但是细想起来，这个做法存在着大量的重复计算，其实在效率上还是可以改进的。因为比如第一步，以根节点root作为路径的开头，去遍历可能值等于sum的路径，这时候就已经遍历过一次了，如果在这个遍历的过程中，能够发现路径中的某一部分（即不以根节点作为开头）的值正好等于sum，就明显提高了效率。