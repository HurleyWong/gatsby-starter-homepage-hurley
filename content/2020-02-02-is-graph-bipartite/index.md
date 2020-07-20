---
title: 785. Is Graph Bipartite
tags: [ leetcode ]
date: 2020-02-02T06:25:44.226Z
path: blog/is-graph-bipartite
cover: ./is-graph-bipartite.png
excerpt: Given an undirected graph, return true if and only if it is bipartite.
---

## Is Graph Bipartite

```
Given an undirected graph, return true if and only if it is bipartite.

Recall that a graph is bipartite if we can split it's set of nodes into two independent subsets A and B such that every edge in the graph has one node in A and another node in B.

The graph is given in the following form: graph[i] is a list of indexes j for which the edge between nodes i and j exists.  Each node is an integer between 0 and graph.length - 1.  There are no self edges or parallel edges: graph[i] does not contain i, and it doesn't contain any element twice.

Example 1:
Input: [[1,3], [0,2], [1,3], [0,2]]
Output: true
Explanation: 
The graph looks like this:
0----1
|    |
|    |
3----2
We can divide the vertices into two groups: {0, 2} and {1, 3}.
Example 2:
Input: [[1,2,3], [0,2], [0,1,3], [0,2]]
Output: false
Explanation: 
The graph looks like this:
0----1
| \  |
|  \ |
3----2
We cannot find a way to divide the set of nodes into two independent subsets.

```

**Note**:

- `graph` will have length in range `[1, 100]`.
- `graph[i]` will contain integers in range `[0, graph.length - 1]`.
- `graph[i]` will not contain `i` or duplicate values.
- The graph is undirected: if any element `j` is in `graph[i]`, then `i` will be in `graph[j]`.

这其实可以看做是一个着色问题，即可以转化为「**如果这个图中每个相邻的节点间的颜色都是不一样的，那么就是二分图**」。

### 邻接表表示矩阵

因为LeetCode上图的画法问题，导致我一开始没有看懂这个图是什么形状，是怎么用邻接表形式表示的。

```code
示例 1:
输入: [[1,3], [0,2], [1,3], [0,2]]
输出: true
解释: 
无向图如下:
0----1
|    |
|    |
3----2
我们可以将节点分成两组: {0, 2} 和 {1, 3}。

示例 2:
输入: [[1,2,3], [0,2], [0,1,3], [0,2]]
输出: false
解释: 
无向图如下:
0----1
| \  |
|  \ |
3----2
我们不能将节点分割成两个独立的子集。
```

如上所示，如果输入`[[1,3], [0,2], [1,3], [0,2]]`，那么这个邻接表表示的是图的节点有`1->3,0->2`，因此画出的图其实是如下所示：

<img src="https://raw.githubusercontent.com/HurleyJames/ImageHosting/master/IMG_9C75CB36F198-1.jpeg" width="50%" />

同理，如果输入`[[1,2,3], [0,2], [0,1,3], [0,2]]`，那么说明`1->2,1->3,0->2,0->1,0->3`，所以图如下所示：

<img src="https://raw.githubusercontent.com/HurleyJames/ImageHosting/master/IMG_FB44338948F8-1.jpeg" width="50%" />

### 方法：DFS搜索着色

如果节点属于第一个集合，将其设置为颜色0，否则为颜色1。当这个图为二分图时，就可以使用**贪心思想**给图着色：比如一个节点的颜色为0，则其所有的邻接点的颜色为1，其所有的邻接点的邻接点的颜色为0，以此类推。

先找到一个未着色的节点$u$，把它染上一种颜色，比如颜色1黑色，然后遍历所有与它相连的节点$v$，如果节点$v$已经被染色并且颜色和$u$是一样的，那么就不是二分图。如果这个节点$v$没有被染色，则先把它染成与节点$u$不同的颜色，例如颜色2红色，然后再遍历所有节点$v$的邻接点，依次递推。

可以使用数组或者哈希表来记录每个节点的颜色：`colors[node]`。颜色有两种，分别为1黑色和2红色，0表示未着色。

#### 代码

```java
public class Is_Graph_Bipartite {

    public static boolean isBipartite(int[][] graph) {
        if (graph == null || graph.length == 0) {
            return false;
        }
        int n = graph.length;
        // 设置 color 数组，0 表示未着色，1 黑，2 红
        int[] colors = new int[n];
        // Arrays.fill 方法将 color 数组中的所有元素的值设置为 0，表示未着色
        Arrays.fill(colors, 0);
        for (int i = 0; i < n; i++) {
            if (!dfs(graph, i, colors, 0)) {
                return false;
            }
        }

        return true;
    }

    private static boolean dfs(int[][] graph, int i, int[] colors, int preColor) {
        // 如果未被染色
        if (colors[i] == 0) {
            // 与相邻节点进行相反的染色
            colors[i] = (preColor == 1) ? 2 : 1;
            for (int j = 0; j < graph[i].length; j++) {
                // 如果不能够再往下递推
                if (!dfs(graph, graph[i][j], colors, colors[i])) {
                    return false;
                }
            }
            return true;
        } else {
            // 已染色
            // 如果颜色和邻接点颜色一致
            if (colors[i] == preColor) {
                return false;
            } else {
                return true;
            }
        }
    }
}
```