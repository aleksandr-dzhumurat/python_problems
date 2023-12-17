class ListNode(object):
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def merge_sorted_lists(list1: ListNode, list2: ListNode):
    """https://leetcode.com/problems/merge-two-sorted-lists/
    """
    cur_node = ListNode(None)
    res = cur_node
    while list1 is not None and list2 is not None:
        if list1.val < list2.val:
            cur_node.next = list1
            list1 = list1.next
        else:
            cur_node.next = list2
            list2 = list2.next
        cur_node = cur_node.next
    if list1 is not None:
        cur_node.next = list1
    if list2 is not None:
        cur_node.next = list2
    return res.next

def hasCycle(head):
    """https://leetcode.com/problems/linked-list-cycle/
    """

    fast = head
    slow = head
    while fast is not None and fast.next is not None:
        slow = slow.next
        fast = fast.next.next

        if slow is fast:
            return True
    return False

def reverseList():
    pass