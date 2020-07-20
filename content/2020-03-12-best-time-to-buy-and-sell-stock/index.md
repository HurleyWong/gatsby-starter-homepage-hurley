---
title: 121. Best Time to Buy and Sell Stock
tags: [ leetcode ]
date: 2020-03-12T06:25:44.226Z
path: blog/best-time-to-buy-and-sell-stock
cover: ./best-time-to-buy-and-sell-stock.png
excerpt: Say you have an array for which the ith element is the price of a given stock on day i.
---

## Best Time to Buy and Sell Stock

```
Say you have an array for which the ith element is the price of a given stock on day i.

If you were only permitted to complete at most one transaction (i.e., buy one and sell one share of the stock), design an algorithm to find the maximum profit.

Note that you cannot sell a stock before you buy one.

Example 1:

Input: [7,1,5,3,6,4]
Output: 5
Explanation: Buy on day 2 (price = 1) and sell on day 5 (price = 6), profit = 6-1 = 5.
             Not 7-1 = 6, as selling price needs to be larger than buying price.
Example 2:

Input: [7,6,4,3,1]
Output: 0
Explanation: In this case, no transaction is done, i.e. max profit = 0.
```

### 方法一：暴力法

最容易想到的暴力法就是比较每个元素与后面的元素的差值。假设数组长度为n，则第一次要比较n-1次，第二个要比较n-2次，以此类推，第n个要比较n-(n-1)=1次。所以总次数也就是时间复杂度为$O(n^2)$。

#### 代码

```java
public int maxProfit(int[] prices) {
    int maxprofit = 0;
    // 前一个元素
    for (int i = 0; i < prices.length - 1; i++) {
        // 后一个元素
        for (int j = 0; j < prices.length; j++) {
            // 记录差值
            int profit = prices[j] - prices[i];
            // 如果当前差值比maxprofit大，就将maxprofit替换成当前差值
            if (profit > maxprofit) {
                maxprofit = profit;
            }
        }
    }
    return maxprofit;
}
```

#### 复杂度分析

* **时间复杂度**：$O(n^2)$，循环进行了$(n-1)+(n-2)+...+1=\frac{n(n-1)}{2}$次。
* **空间复杂度**：$O(1)$。因此只使用了常数个变量。

### 一次遍历

遍历一次数组，在遍历每一天时，既要用一个变量判断历史最低价格`minprice`，也要在当天判断当天与历史最低价格的差值是否是最大利润。

#### 代码

```java
public int maxProfit(int[] prices) {
    int minprice = Integer.MAX_VALUE;
    int maxprofit = 0;
    for (int i = 0; i < prices.length; i++) {
        if (prices[i] < minprice) {
            minprice = prices[i];
        }
        else if (prices[i] - minprice > maxprofit) {
            maxprofit = prices[i] - minprice;
        }
    }
    return maxprofit;
}
```

首先，设置`minprice`为最大值，`maxprofit`为0。然后开始遍历数组。

如果当前数组的值比`minprice`小，就发生替换；如果更大就不替换。这样就能得到这个数组中最小的值。然后用当前数组的值减去`minprice`，如果得到的利润大于最大利润，就得到最大利润。因为这个都放在同一个`for`循环里，所以遍历的顺序是从0从n，从前往后，所以是不存在前面一个数减后面一个数的情况。

这道题评论有质疑这种做法是话如果最小值在数组的最后一位会不成立。实际上亲测是不影响的。可以通过下图说明。

<img src="https://raw.githubusercontent.com/HurleyJames/ImageHosting/master/IMG_0240%202.jpg" width="100%" />

由上图可发现，尽管最后一位才是数组的最小值，最后的`minprice`也更新为了1，但是因为之前已经保留了`maxprofit`的值为4，所以最后一位的最小值被当前元素相减，结果为1仍然小于4，所以不会更新替换`maxprofit`的值，所以结果仍然是正确的。

#### 复杂度

* **时间复杂度**：因为只使用了一次`for`循环遍历了整个数组，所以时间复杂度是$O(n)$。
* **空间复杂度**：是使用了常数个变量，所以是$O(1)$。

