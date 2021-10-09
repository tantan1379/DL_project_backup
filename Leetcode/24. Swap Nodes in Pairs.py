'''
@File    :   24. Swap Nodes in Pairs.py
@Time    :   2021/10/07 16:20:53
@Author  :   Tan Wenhao 
@Version :   1.0
@Contact :   tanritian1@163.com
@License :   (C)Copyright 2021-Now, MIPAV Lab (mipav.net), Soochow University. All rights reserved.

Given a linked list, swap every two adjacent nodes and return its head.
'''

from utils.ll import *


def swapPairs(head: ListNode):
    dummy = ListNode(0, head)
    cur = dummy
    while(cur.next and cur.next.next):
        node1 = cur.next
        node2 = cur.next.next
        cur.next = node2
        node1.next = node2.next
        node2.next = node1
        cur = node1
    return dummy.next


if __name__ == "__main__":
    head = InitLinkList([1, 2, 3, 4, 5])
    res = swapPairs(head)
    ForeachLinkList(res)
