package database.leetcode_2;


/**
 * @Project: Leetcode
 * @Package: database.AddTwoNumbers_2
 * @Date: 2022/1/11 22:25
 * @Author: Wenhao Tan
 * @Version: 1.0
 * @License: (C)2022, MIPAV Lab(mipav.net), Soochow University. tanritian1@163.com All Rights Reserved.
 * 给你两个非空的链表，表示两个非负的整数。它们每位数字都是按照逆序的方式存储的，并且每个节点只能存储一位数字。
 *
 * 请你将两个数相加，并以相同形式返回一个表示和的链表。
 *
 * 你可以假设除了数字 0 之外，这两个数都不会以 0开头。

 */
class Solution {
    public ListNode addTwoNumbers(ListNode l1, ListNode l2) {
        ListNode head = new ListNode();
        ListNode cur = head;
        int carry = 0;
        while(l1!=null || l2!=null){
            int res = (l1!=null?l1.val:0)+(l2!=null?l2.val:0)+carry;
            int output = res%10;
            carry = res/10;
            cur.next = new ListNode(output);
            cur = cur.next;

            if(l1!=null){
                l1 = l1.next;
            }
            if(l2!=null){
                l2 = l2.next;
            }
        }
        if(carry!=0){
            cur.next = new ListNode(carry);
        }
        return head.next;
    }


    public static void main(String[] args) {
        Solution s = new Solution();
        Tools tools = new Tools();
        ListNode l1 = tools.createListNode(new int[]{2,4,3});
        ListNode l2 = tools.createListNode(new int[]{5,6,4});
        ListNode res = s.addTwoNumbers(l1,l2);
        tools.printListNode(res);

    }
}
