---
title: 463. Island Perimeter
tags: [ leetcode ]
date: 2020-03-12T06:25:44.226Z
path: blog/island-perimeter
cover: ./island-perimeter.png
excerpt: You are given a map in form of a two-dimensional integer grid where 1 represents land and 0 represents water.
---

## Island Perimeter

```
You are given a map in form of a two-dimensional integer grid where 1 represents land and 0 represents water.

Grid cells are connected horizontally/vertically (not diagonally). The grid is completely surrounded by water, and there is exactly one island (i.e., one or more connected land cells).

The island doesn't have "lakes" (water inside that isn't connected to the water around the island). One cell is a square with side length 1. The grid is rectangular, width and height don't exceed 100. Determine the perimeter of the island.

Input:
[[0,1,0,0],
 [1,1,1,0],
 [0,1,0,0],
 [1,1,0,0]]

Output: 16
```

这道题的思路是总的正方形块树的数量乘以4条边，然后减去重合的边树乘以2，就是岛屿的周长。所以解题的关键就是在于**有规律的找到有多少次的重叠**，这样能够避免重复的计算。

#### 代码

最开始想到的代码写法是这样的：

```java
public int islandPerimeter(int[][] grid) {
    // 如果该矩阵为空
    if (grid.length == 0 || grid[0].length == 0) {
        return 0;
    }

    // 陆地的数量
    int land = 0;
    // 重叠的数量
    int overlap = 0;
    for (int i = 0; i < grid.length; i++) {
        for (int j = 0; j < grid[0].length; j++) {
            // 规定数组元素为1时是陆地
            if (grid[i][j] == 1) {
                land++;
                // 分别找上下左右是否是陆地，如果是陆地就意味着有重合
                // 下
                // 因为判断下边，所以i要小于grid.length-1，不然就已经是最下边的元素了
                if (i < grid.length - 1 && grid[i][j] == grid[i + 1][j]) {
                    overlap++;
                }
                // 右
                // 因为判断右边，所以j要小于grid[i].length-1，不然就已经是最右边的元素了
                if (j < grid[i].length - 1 && grid[i][j] == grid[i][j + 1]) {
                    overlap++;
                }
                // 左
                // 因为判断左边，所以j要大于0，不然就已经是最左边了
                if (j > 0 && grid[i][j] == grid[i][j - 1]) {
                    overlap++;
                }
                // 上
                // 因为判断上边，所以i要大于0，不然就已经是最上边了
                if (i > 0 && grid[i][j] == grid[i - 1][j]) {
                    overlap++;
                }
            }
        }
    }
    return 4 * land - 2 * overlap;
}
```

刚开始想到的就是上下左右进行判断是否是陆地，如果是陆地，就说明了有重叠部分。但是这样去提交代码测试结果却发现是错误的。后来才想明白，其实上下左右只要判断两个方向就可以了。

例如判断左边的部分，首先条件`j > 0`本来就是恒成立的，因为`for`循环是从0开始递增的，所以这里是多余的。然后，在双层循环了里，判断`grid[i][j] == grid[i][j + 1]`和判断`grid[i][j] == grid[i][j - 1]`其实是一样的效果。所以如果在这里判断了两侧，就会造成`overlap`累加了两次，从而计算错误。

**正确的代码**：

```java
public int islandPerimeter(int[][] grid) {
    // 如果该矩阵为空
    if (grid.length == 0 || grid[0].length == 0) {
        return 0;
    }

    // 陆地的数量
    int land = 0;
    // 重叠的数量
    int overlap = 0;
    for (int i = 0; i < grid.length; i++) {
        for (int j = 0; j < grid[0].length; j++) {
            // 规定数组元素为1时是陆地
            if (grid[i][j] == 1) {
                land++;
                // 分别找上下左右是否是陆地，如果是陆地就意味着有重合
                // 纵向
                // 因为判断下边，所以i要小于grid.length-1，不然就已经是最下边的元素了
                if (i < grid.length - 1 && grid[i][j] == grid[i + 1][j]) {
                    overlap++;
                }
                // 横向
                // 因为判断右边，所以j要小于grid[i].length-1，不然就已经是最右边的元素了
                if (j < grid[i].length - 1 && grid[i][j] == grid[i][j + 1]) {
                    overlap++;
                }
            }
        }
    }
    return 4 * land - 2 * overlap;
}
```

`return 4 * land - 2 * overlap`的原因是根据规律得知，最终的周长就是陆地的数量乘以4条边然后减去重合数量的2倍（因为重复计算了2条边）。

#### 复杂度分析

* **时间复杂度**：如果这个二维矩阵的宽高分别为m和n，那么进行了双层循环，所以时间复杂度为$O(mn)$。

    