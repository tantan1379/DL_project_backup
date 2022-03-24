package database.leetcode_2;

import org.junit.Test;

/**
 * @Project: Leetcode
 * @Package: database.leetcode_2
 * @Date: 2022/2/25 16:58
 * @Author: Wenhao Tan
 * @Version: 1.0
 * @License: (C)2022, MIPAV Lab(mipav.net), Soochow University. tanritian1@163.com All Rights Reserved.
 */
public class test {
    @Test
    public void testSolution() {
        Solution s = new Solution();
        Tools tools = new Tools();
        ListNode l1 = tools.createListNode(new int[]{2, 4, 3});
        ListNode l2 = tools.createListNode(new int[]{5, 6, 4});
        ListNode res = s.addTwoNumbers(l1, l2);
        tools.printListNode(res);
    }
}
