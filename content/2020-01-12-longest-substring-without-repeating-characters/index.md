---
title: Longest Substring Without Repeating Characters
tags: [ leetcode ]
date: 2020-01-12T06:25:44.226Z
path: blog/longest-substring-without-repeating-characters
cover: ./longest-substring-without-repeating-characters.png
excerpt: Give two non-empty linked lists representing two non-negative integers. The digits are stored in reverse order and each of their nodes contain a single digit. Add the two numbers and return it as a linked list.
---

## Longest Substring Without Repeating Characters

```
Given a string, find the length of the longest substring without repeating characters.

Example 1:

Input: "abcabcbb"
Output: 3 
Explanation: The answer is "abc", with the length of 3. 
Example 2:

Input: "bbbbb"
Output: 1
Explanation: The answer is "b", with the length of 1.
Example 3:

Input: "pwwkew"
Output: 3
Explanation: The answer is "wke", with the length of 3. 
             Note that the answer must be a substring, "pwke" is a subsequence and not a substring.
```

### 方法一：暴力法(Naive brute force)

可以使用暴力法逐个检查所有的子字符串，然后记录长度，最终选择长度最大的。

因为字符长度为$n$的字符串，会有$n^2$个`subString`，然后检查每一个`subString`中是否含有重复字符又得遍历该`subString`,所以又需要$O(n)$，所以总的时间复杂度就是$O(n^3)$。

首先写一个对获得不重复子字符串的方法。定义一个HashSet，然后对字符串进行遍历操作，如果HashSet中不含有该元素，就添加到HashSet中；如果有，就返回false。

然后使用两层循环，去判断是否是不重复子串并记录长度。假设开始和结束的索引分别为i和j，那么就使用i从0~n-1以及j从i+1~n这两个嵌套的循环，就可以枚举出所有的子字符串。

```kotlin
/**
 * 1.暴力法
 * 返回最长子穿的长度
 */
fun lengthOfLongestSubstring(s: String): Int {
    // n是字符串的长度
    val n = s.length
    var ans = 0
    for (i in 0 until n) {
        for (j in i + 1 until n) {
            if (allUnique(s, i, j)) {
                ans = Math.max(ans, j - i)
            }
        }
    }
    return ans
}

fun allUnique(s: String, start: Int, end: Int): Boolean {
    val hashSet: HashSet<Char> = HashSet();
    for (i in start until end) {
        val ch = s[i]
        // 如果HashSet中含有该元素，就返回false
        if (hashSet.contains(ch)) {
            return false
        }
        // 如果HashSet中不含有该元素，就添加到这个HashSet中
        hashSet.add(ch)
    }
    return true
}
```

#### 复杂度分析：

* **时间复杂度：**$O(n^3)$

    这里使用了三层循环，遍历了三次字符串。所以时间复杂度显然是$O(n^3)$。

* **空间复杂度：**$O(k)$

    因为需要$O(k)$的空间来检查子字符串中是否有重复字符，其中k表示的是HashSet的大小。

但暴力法的效率实在太低，当长度过长时可能会出现**TLE**(Time Limit Exceeded)。不推荐使用。

### 方法二：滑动窗口(Sliding Window)

在暴力法中，枚举出所有子字符串之后，第3步要**从首到尾的元素**去检查每一个子字符串是否有重复元素。

但其实没有必要遍历一个子字符串的所有元素。

例如：字符串qwekq，子字符串qwe、qwek等。

如果子字符串qwe已经检查过是没有重复元素的，那么在检查qwek时，就没有必要从头到尾，将qwe之间再检查一遍。只需要检查新添加的元素k是否与之前的字符串有重复元素即可。

即，**如果从索引i到j-1之间的子字符串$S_{ij}$已经被检查为没有重复字符，只需要检查$S[j]$对应的字符是否已经存在于子字符串$S_{ij}$中。**

**窗口**通常是在数组/字符串中由开始和结束索引定义的一系列元素的集合，即$[i,j)$。滑动窗口是可以将两个边界向某一方向**“滑动”**的窗口。滑动窗口通常用来求解数组和`String`。

例如：将$[i,j)$向右滑动1个元素$\Rightarrow[i+1,j+1)$

所以，这题可以使用HashSet将字符存储在当前窗口$[i,j)$（最初j=i）中，然后向右滑动索引j，如果它不在HashSet中，继续滑动j，直到$S[j]$已经存在于HashSet中。

#### 例子

一个字符串abcb，求最大子串。

n等于字符串长度等于4，然后令ans=0，i=0，j=0。

1. 因为初始的HashSet为空，所以肯定不含有$S[j]$即$S[0]$。所以把$S[0]$添加到HashSet中，然后j++。这时候ans=max(0, j-i)=max(0,1)=1。Set中有[a]。
2. $S[1]$为b，HashSet不含有，则把$S[1]$添加到HashSet中，然后j++。这时候ans=max(1,j-i)=max(1,2-0)=2。Set中有[a,b]。
3. $S[2]$为c，HashSet中没有，则把$S[2]$添加到HashSet中，然后j++。这时候ans=max(2,3-0)=3。Set为[a,b,c]。
4. $S[3]$为b，HashSet中已经含有b，因此`set.remove(s[0])`，然后i++。即把HashSet的第一个元素a去掉，这时候Set为[b,c]。
5. 接着进行判断，$S[3]$是b，HashSet中仍然含有b，所以`set.remove(s[1])`，然后i++。这样就是把HashSet的第二个元素b去掉了。
6. 接着判断，$S[3]$是b，这时候HashSet已经不包含b了。所以把b添加到Set中，然后j++。这时候j=4，已经不能再进行下一次循环了。这时候的Set是[c,b]，ans=max(3, 4-2)=3。
7. 所以最终得到的最长子串的长度是3。

```kotlin
/**
 * 2.滑动窗口
 * 返回最长子串的长度
 */
fun lengthOfLongestSubstring2(s: String): Int {
    val n = s.length
    val set: HashSet<Char> = HashSet()
    // 默认长度为0，i和j从0开始向右移
    var ans = 0
    var i = 0
    var j = 0
    while (i < n && j < n) {
        if (!set.contains(s[j])) {
            set.add(s[j++])
            print(set)
            ans = Math.max(ans, j - i)
        } else {
            set.remove(s[i++])
        }
    }
    return ans
}
```

#### 复杂度分析：

* **时间复杂度：**$O(n)=O(2n)$

    可以从代码中看出，使用while进行了一次遍历，长度是n。但是因为条件是`i < n && j < n`，所以最坏的情况下可能判断了两遍n，即从i遍历到n和由j遍历到n。所以最坏情况的时间复杂度是$O(2n)$。

* **空间复杂度**：$O(k)$

    滑动窗口法仍然需要$O(k)$的空间，其中k的表示Set的大小。

### 方法三：优化的滑动窗口

从方法二的例子的步骤分析上就可以看出，**步骤4-6**其实有一些冗余。当发现$S[j]$在$[i,j)$范围有重复字符时，不需要让i=0开始，使用i++逐步增加i，可以直接跳过$[i,j']$范围内的所有元素，并将i变成j'+1。

#### 图解（转载至LeetCode：灵魂画师牧码）

<img src="https://raw.githubusercontent.com/HurleyJames/ImageHosting/master/2847c2d9fb9a6326fecfcf8831ed1450046f1e10967cde9d8681c42393d745ff-frame_00001.png" width="100%" />

<img src="https://raw.githubusercontent.com/HurleyJames/ImageHosting/master/159cc7509e4a5acbfaf5c59b4b5cb1674f1a31fb87cc41528ca6e6df6132b1dc-frame_00002.png" width="100%" />

<img src="https://raw.githubusercontent.com/HurleyJames/ImageHosting/master/a62a6d9c878b4c856db1467b4282b936ee677d02a3b47ac4c67dfb4269a158f6-frame_00003.png" width="100%" />

<img src="https://raw.githubusercontent.com/HurleyJames/ImageHosting/master/7b672e389b1659d3ff2ba77101cf49de120a21732dd7aed5a707d8b33d6b2fb6-frame_00004.png" width="100%" />

<img src="https://raw.githubusercontent.com/HurleyJames/ImageHosting/master/ff8f38005f548beb5bd45a2e5e327f71acf069c8ad6e9680caeee655af71533a-frame_00005.png" width="100%" />

<img src="https://raw.githubusercontent.com/HurleyJames/ImageHosting/master/2f054f105ebcbe7a1cf3cce1a4ab8c0d85cef70fe674bb90a1c83e92dc6b1274-frame_00006.png" width="100%" />

<img src="https://raw.githubusercontent.com/HurleyJames/ImageHosting/master/018b08f276a746262cf64fa1cf0748d815f3cabe9c29c61f4973b6e6dd44e2c8-frame_00007.png" width="100%" />

```kotlin
fun lengthOfLongestSubstring3(s: String): Int {
    val n = s.length
    var ans = 0
    var i = 0
    var j = 0
    val map: HashMap<Char, Int> = HashMap()
    for (j in 0 until n) {
        if (map.containsKey(s[j])) {
            i = Math.max(map.getValue(s[j]), i)
        }
        ans = Math.max(ans, j - i + 1)
        map.put(s[j], j + 1)
    }
    return ans
}
```

#### 复杂度分析

* **时间复杂度：**$O(n)$

    可以很明显看到，j由0遍历到n，循环了n次。

* **空间复杂度：**$O(k)$

    滑动窗口法仍然需要$O(k)$的空间，其中k的表示Set的大小。

