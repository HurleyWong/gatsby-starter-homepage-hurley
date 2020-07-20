---
title: Add Two Numbers
tags: [ leetcode ]
date: 2020-01-11T06:25:44.226Z
path: blog/add-two-numbers
cover: ./add-two-numbers.png
excerpt: Give two non-empty linked lists representing two non-negative integers. The digits are stored in reverse order and each of their nodes contain a single digit. Add the two numbers and return it as a linked list.
---

## Add Two Numbers

```
You are given two non-empty linked lists representing two non-negative integers. The digits are stored in reverse order and each of their nodes contain a single digit. Add the two numbers and return it as a linked list.

You may assume the two numbers do not contain any leading zero, except the number 0 itself.

Example:

Input: (2 -> 4 -> 3) + (5 -> 6 -> 4)
Output: 7 -> 0 -> 8
Explanation: 342 + 465 = 807.
```

### 知识点

这道题主要考察了两个知识点。第一个就是**链表**，链表的遍历和链表的创建。第二个就是高精度加法的模拟，因为题目中数字的长度**其实可以很长**。

### 方法：初等数学

用初等数学的方法，相当于进行**加法**的计算。

上图的carry=1的意思是，前一位4+6=10进了1位，所以进位让carry从默认值0变为1。然后3+4+1=8。

#### 图解（转载至LeetCode：灵魂画师牧码）

<img src="https://raw.githubusercontent.com/HurleyJames/ImageHosting/master/2519bd7f7da0f3bd51dd0f06e6363f4f62bfb25472c5ec233cf969e5c1472e33-file_1559748028103.png" width="100%" />

<img src="https://raw.githubusercontent.com/HurleyJames/ImageHosting/master/400f2a615319c4f0f42c39eb8b8902984922d1e778ca461569ff64460eaa9757-file_1559748028117.png" width="100%" />

<img src="https://raw.githubusercontent.com/HurleyJames/ImageHosting/master/e0d3266ec83cee00c6a0ff0a8a66de8d129798b24b76a19b7883f2fd1d79c15b-file_1559748087173.png" width="100%" />

<img src="https://raw.githubusercontent.com/HurleyJames/ImageHosting/master/a5bf6bc2cc15d162bd35eb8fc467fb36887e40b36c26bdc982a11a686b34cb30-file_1559748028113.png" width="100%" />

<img src="https://raw.githubusercontent.com/HurleyJames/ImageHosting/master/fc6475aca0ec0621003f4888a59086c398ff5fc6ee2e27cbfb9bc91f107383b9-file_1559748028094.png" width="100%" />

<img src="https://raw.githubusercontent.com/HurleyJames/ImageHosting/master/743afc3cb34954e1f3a9b41924d4af5453832d23772a2e46aa4cd52a2b240bdd-file_1559748028108.png" width="100%" />

<img src="https://raw.githubusercontent.com/HurleyJames/ImageHosting/master/3323b948431675b9f2ff8b0161eee9178298cbb4403cbcd36dc857f14043cf7a-file_1559748028112.png" width="100%" />

<img src="https://raw.githubusercontent.com/HurleyJames/ImageHosting/master/508d1bb12a372e385c4052d95ca92e06c3a63a805bf12feddd0bb4e7c972f016-file_1559748028116.png" width="100%" />

#### 伪代码

* 将进位值carry设置为0
* 将p和q分别初始化为$l_1$和$l_2$的头部
* 遍历$l_1$和$l_2$直至到达它们的尾端
    * 将x设置为结点p的值。如果p已经到达$l_1$的末尾，则将其值设置为0
    * 将y设置为结点q的值。如果q已经到达$l_2$的末尾，则将其值设置为0
    * 设定sum=x+y+carry
    * carry=sum/10，将carry取整。这里的carry要么为0，要么为1
    * 创建一个数值为(sum mod 10)的新结点（mod为求余数），将其设置为当前结点的下一个结点，然后将当前结点前进到下一个结点
    * 同时，将p和q前进到下一个结点
* 检查carry=1是否成立（可以通过判断carry是否大于0），如果成立，则追加一个为1的新结点

```
dummy = tail = ListNode(0)
while l1 or l2 or carry:
	sum = l1?.val + l2?.val + carry
	tail.next = ListNode(sum % 10)
	tail = tail.next
	carry = sum /= 10
	l1, l2 = l1?.next, l2?.next
return dummy.next

Time complexity: O(max(n,m))
Space complexity: O(max(n,m))
```

#### 代码

```kotlin
package medium._002

/**
 * You are given two non-empty linked lists representing two non-negative integers. The digits are stored in reverse order and each of their nodes contain a single digit. Add the two numbers and return it as a linked list.
 *
 * You may assume the two numbers do not contain any leading zero, except the number 0 itself.
 *
 * Example:
 *
 * Input: (2 -> 4 -> 3) + (5 -> 6 -> 4)
 * Output: 7 -> 0 -> 8
 * Explanation: 342 + 465 = 807.
 */

fun addTwoNumbers(l1: ListNode?, l2: ListNode?): ListNode? {
    val head = ListNode(0)
    var p = l1
    var q = l2
    var curr: ListNode? = head
    // 进位carry初始化为0
    var carry = 0
    while (p != null || q != null) {
        // 如果p不为空，将x设为结点p的值；如果p已到达l1的末尾，则p=null，则x=0
        val x = p?.`val` ?: 0
        // q同理
        val y = q?.`val` ?: 0
        val sum = x + y + carry
        // 将carry取整
        carry = sum / 10
        curr!!.next = ListNode(sum % 10)
        curr = curr.next
        if (p != null) {
            p = p.next
        }
        if (q != null) {
            q = q.next
        }
    }
    // 检查carry是否等于1
    if (carry > 0) {
        curr!!.next = ListNode(carry)
    }
    return head.next
}

/**
 * 打印链表
 * @param last
 */
fun printList(last: ListNode?) {
    var last = last
    while (last != null) {
        // 如果是最后一位，则不输出逗号，
        if (last.next == null) {
            print(last.`val`)
        } else {
            // 如果不是最后，则输入逗号，分隔
            print(last.`val`.toString() + ",")
        }
        last = last.next
    }
}

fun main() {
    // 原测试用例：l1=[2,4,3]，l2=[5,6,4]，输出结果为[7,0,8]
    val l1 = ListNode(2)
    l1.next = ListNode(4)
    l1.next!!.next = ListNode(3)
    val l2 = ListNode(5)
    l2.next = ListNode(6)
    l2.next!!.next = ListNode(4)
// 测试用例1：l1=[0,1]，l2=[0,1,2]，输出结果应为[0,2,2]
//        ListNode l1 = new ListNode(0);
//        l1.next = new ListNode(1);
//        ListNode l2 = new ListNode(0);
//        l2.next = new ListNode(1);
//        l2.next.next = new ListNode(2);
// 测试用例2：l1=[]，l2=[0,1]，输出结果为[0,1]
//        ListNode l1 = new ListNode();
//        ListNode l2 = new ListNode(0);
//        l2.next = new ListNode(1);
// 测试用例l3：l1=[9,9]，l2=[1]，输出结果为[0,0,1]
//        ListNode l1 = new ListNode(9);
//        l1.next = new ListNode(9);
//        ListNode l2 = new ListNode(1);
    printList(addTwoNumbers(l1, l2))
}


/**
 * ListNode是自己定义的Java中的链表对象
 * 类结构如下
 */
class ListNode {
    var `val`: Int
    var next: ListNode? = null

    constructor() {
        `val` = 0
    }

    constructor(i: Int) {
        `val` = i
    }

    fun `val`(): Int {
        return `val`
    }
}
```

#### 复杂度分析：

* **时间复杂度：**$O(max(m,n))$

    该算法使用了一次whlie循环，且判断条件是`p != null || q != null`。假设链表p和q的长度分别为m和n，则该循环最多重复$max(m,n)$次。

* **空间复杂度**：$O(max(m,n))$

    新链表的长度也同样取决于p和q的长度，但由于相加后有可能产生进位，所以长度可能加1。所以长度最多为$max(m,n)+1$。

    如果仅仅是把结果打印出来，那么空间复杂度就是$O(1)$，因为不需要额外的存储。

