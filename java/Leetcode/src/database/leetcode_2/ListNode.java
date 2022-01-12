package database.leetcode_2;

/**
 * @Project: Leetcode
 * @Package: database.leetcode_2
 * @Date: 2022/1/11 22:27
 * @Author: Wenhao Tan
 * @Version: 1.0
 * @License: (C)2022, MIPAV Lab(mipav.net), Soochow University. tanritian1@163.com All Rights Reserved.
 */
public class ListNode {
    int val;
    ListNode next;
    ListNode() {}
    ListNode(int val) { this.val = val; }
    ListNode(int val, ListNode next) { this.val = val; this.next = next; }
}

class Tools{
    public void printListNode(ListNode head){
        ListNode cur = head;
        while(cur!=null){
            if(cur.next!=null) {
                System.out.print(cur.val + "->");
            }
            else{
                System.out.print(cur.val);
            }
            cur = cur.next;
        }
        System.out.println();
    }

    public ListNode createListNode(int[] array){
        ListNode head = new ListNode();
        ListNode cur = head;
        for (int element : array) {
            cur.next = new ListNode(element);
            cur = cur.next;
        }
        return head.next;
    }
}