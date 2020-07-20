---
title: Two Sum
tags: [ leetcode ]
date: 2020-01-10T06:25:44.226Z
path: blog/two-sum
cover: ./two-sum.png
excerpt: Give an array of integers, return indices of the two numbers such that they add up to a specific target.
---

## Two Sum

```
Given an array of integers, return indices of the two numbers such that they add up to a specific target.

You may assume that each input would have exactly one solution, and you may not use the same element twice.

Example:

Given nums = [2, 7, 11, 15], target = 9,

Because nums[0] + nums[1] = 2 + 7 = 9,
return [0, 1].
```

<!-- more -->

### 方法一：暴力法

暴力法就是遍历每个元素x，并查找是否存在一个值与target-x相等的目标元素。

```java
public static int[] twoSum1(int[] nums, int target) {
    for (int i = 0; i < nums.length; i++) {
        for (int j = 0; j <= i; j++) {
            if (i != j && (nums[i] + nums[j] == target)) {
                System.out.println("Two numbers: " + nums[j] + " and " + nums[i]);
                System.out.println("Indices of the two numbers: " + j + " and " + i);
                return nums;
            }
        }
    }
    throw new IllegalArgumentException("No two sum solution");
}
```

#### 复杂度分析：

* **时间复杂度：**$O(n^2)$

    对于每个元素，通过遍历数组的其余部分来寻找它所对应的目标元素，这将耗费$O(n)$的时间。所以两个循环遍历的时间复杂度为$O(n^2)$。

* **空间复杂度：**$O(1)$。

### 方法二：两遍哈希表

因为该题是检查数组中是否存在目标元素满足条件。如果满足，则找出该目标元素的索引。所以可以使用**哈希表**来保持数组中的每个元素与其索引相互对应（键值对）。

使用两次迭代。在第一次迭代中，将每个元素的值和它的索引添加到表中。然后，在第二次迭代中，将检查每个元素所对应的目标元素(target-nums[i])是否存在于表中（该目标元素不能是nums[i]本身）。

```java
public static int[] twoSum2(int[] nums, int target) {
    Map<Integer, Integer> map = new HashMap<>();
    for (int i = 0; i < nums.length; i++) {
        map.put(nums[i], i);
    }
    for (int i = 0; i < nums.length; i++) {
        int complement = target - nums[i];
        if (map.containsKey(complement) && map.get(complement) != i) {
            System.out.println("Two numbers: " + nums[i] + " and " + complement);
            System.out.println("Indices of the two numbers: " + i + " and " + map.get(complement));
            return new int[]{i, map.get(complement)};
        }
    }
    throw new IllegalArgumentException("No two sum solution");
}
```

#### 复杂度分析：

* **时间复杂度：**$O(n)$

    从代码中可以看到，运行了两次```for (int i = 0; i < nums.length; i++)```代码，即将n个元素遍历的两次。但是因为哈希表将查找的时间降低到$O(1)$，所以时间复杂度是$O(n)$。

* **空间复杂度：**$O(n)$

    所需的空间是因为定义了一个哈希表存储数组的元素及其索引。所以空间大小取决于哈希表中存储的元素数量，即n个元素。所以是$O(n)$。

### 方法三：一遍哈希表

观察方法二的代码，发现其实运行了两遍```for (int i = 0; i < nums.length; i++)```代码，所以其实可以一次就完成。

首先创建一个map，然后在数组中进行循环，令complement=target-nums[i]。如果map中含有complement，就已找到目标元素。如果没有找到，那么就把这个元素的索引和值都添加到map中。

所以，其实一开始的时候是肯定找不到目标元素的，因为map中并没有蒜素。等到map中添加了两个元素和索引之后，map中就有可能含有正好等于差的complement了。

```java
public static int[] twoSum3(int[] nums, int target) {
    Map<Integer, Integer> map = new HashMap<>();
    for (int i = 0; i < nums.length; i++) {
        int complement = target - nums[i];
        if (map.containsKey(complement) && map.get(complement) != i) {
            System.out.println("Two numbers: " + complement + " and " + nums[i]);
            System.out.println("Indices of the two numbers: " + map.get(complement) + " and " + i);
            return new int[]{map.get(complement), i};
        }
        map.put(nums[i], i);
    }
    throw new IllegalArgumentException("No two sum solution");
}
```

#### 复杂度分析：

* **时间复杂度：**$O(n)$

    与方法二同理。

* **空间复杂度：**$O(n)$

    空间仍然是需要元素的数量n去用哈希表进行存储。

    