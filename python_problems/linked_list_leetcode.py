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

def merge_k_sorted_lists(lists):
    """https://leetcode.com/problems/merge-k-sorted-lists/submissions/1187482119/
    """
    from heapq import heappush, heappop

    heap = []
    for head in lists:
      if head:
        heappush(heap, (head.val, head))
    head = tail = None
    while heap:
      val, node = heappop(heap)  # always min element https://pythontic.com/algorithms/heapq/heappush
      if not head:
        head = tail = node
      else:
        tail.next = node
        tail = node
      if node.next:
        heappush(heap, (node.next.val, node.next))
    return head

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

def add_two_numbers(l1, l2):
    """https://leetcode.com/problems/add-two-numbers/description/"""
    res = ListNode(None)
    current = res
    carry = val_sum = 0
    while l1 is not None or l2 is not None:
        val_sum = carry
        if l1 is not None:
            val_sum += l1.val
            l1 = l1.next
        if l2 is not None:
            val_sum += l2.val
            l2 = l2.next
        carry = val_sum // 10
        current.next = ListNode(val_sum % 10)
        current = current.next
    if carry == 1:
        current.next = ListNode(carry)
    return res.next

def remove_nth_from_end(head, n):
    """https://leetcode.com/problems/remove-nth-node-from-end-of-list/"""
    tmp = ListNode(None)
    tmp.next = head

    slow = tmp
    fast = tmp.next
    for _ in range(n):
        fast = fast.next
    while fast is not None:
        fast = fast.next
        slow = slow.next
    slow.next = slow.next.next
    return tmp.next

def odd_even_list(head):
    """https://leetcode.com/problems/odd-even-linked-list/description/"""
    if head is None:
        return None
    odd = head
    even = head.next

    even_list = odd.next
    while even is not None and even.next is not None:
        odd.next = even.next
        odd = odd.next

        even.next = odd.next
        even = even.next
    odd.next = even_list
    return head


def reverseList(head):
    """https://leetcode.com/problems/reverse-linked-list/"""
    prev = None
    current = head
    while current is not None:
        next = current.next
        current.next = prev

        prev = current
        current = next
    head = prev
    return head