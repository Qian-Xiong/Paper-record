# Definition for singly-linked list.
class ListNode(object):
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next


class Solution(object):
    def deleteDuplicates(self, head):
        """
        :type head: ListNode
        :rtype: ListNode
        """
        q = ListNode()
        L = q
        pre = head
        while pre.next != None:
            s = 0
            t = pre
            t_v = pre.val
            while pre.val == t_v:
                s += 1
                pre = pre.next

            if s == 1 :
                print(f"s:{s},p:{pre.val},t:{t.val}")
                q.next = t
                q = q.next
        return L.next


s = Solution()

l = [1, 2, 3, 3, 3, 4, 5,5]
head = ListNode()
L = head
for i in l:
    node = ListNode()
    node.val = i
    head.next = node
    head = head.next
head = L.next
while (head != None):
    print(head.val)
    head = head.next

head = L.next
out = s.deleteDuplicates(head)
while (out != None):
    print(out.val)
    out = out.next
