from typing import Any, Optional


class LinkedListIterator:
    def __init__( self, iterable_linked_list):
        self._linked_list_head = iterable_linked_list
        self._curItem = iterable_linked_list._head
    
    def __iter__(self):
        """Метод будет вызван при создании нового итератора"""
        return self

    def __next__(self):
        """Вызывается при каждой итерации"""
        if self._curItem != None:
            item = self._curItem.item
            self._curItem = self._curItem.next_node
            return item
        else:
            raise StopIteration


class ListNode:
    def __init__(self, item: int, next_node: Optional['ListNode'] = None):
        self.item = item
        self.next_node = next_node


class LinkedList:
    """Элементы, добавленные последними, встают в хвост списка"""
    def __init__(self):
        self._head = None
        self._tail = None
        self._size = 0

    def __len__(self):
        return self._size

    def __iter__(self):
        return LinkedListIterator(self)
    
    def add(self, item):
        new_node = ListNode(item)
        if self._head is None:
            # случай пустого списка
            self._head = new_node
        else:
            # "перевешиваем" указатель next "хвоста" на новую ноду
            self._tail.next_node = new_node
        self._tail = new_node  # новая нода становится "хвостом списка"
        self._size += 1

    @property
    def head(self):
        """Иногда полезно получить указатель на голову"""
        return self._head
    
    def set_head(self, new_head: ListNode):
        """Для случаев, когда нужно откинуть dummy head"""
        self._head = new_head


def create_linked_list(input_list: list):
    result = LinkedList()
    for item in input_list:
        result.add(item)
    return result 


def print_linked_list(l: LinkedList):
    print([i for i in l])


def merge_sorted_lists(l1: LinkedList, l2: LinkedList):
    """https://leetcode.com/problems/merge-two-sorted-lists/
    """
    result_list = LinkedList()
    i = l1._head
    j = l2._head
    while i is not None and j is not None:
        if i.item < j.item:
            result_list.add(i.item)
            i = i.next_node
        else:
            result_list.add(j.item)
            j = j.next_node
    if i is not None:
        current_node = i
        while current_node is not None:
            result_list.add(current_node.item)
            current_node = current_node.next_node
    if j is not None:
        current_node = j
        while current_node is not None:
            result_list.add(current_node.item)
            current_node = current_node.next_node
    return result_list


def add_two_numbers(l1: LinkedList, l2: LinkedList):
    """
        (2 -> 4 -> 3) +
        (5 -> 6 -> 4) --->
        (7 -> 0 -> 8)

        (9 -> 9) +
        (1)           ---> 
        (0 -> 0 -> 1)
    Алгоритм:
        l1_curr, l2_curr - указатели, которые синхронно перемещаем по массиву
    """
    relult = LinkedList()
    n1 = l1.head
    n2 = l2.head
    overflow = 0  # тут храним переполнение, когда сумма цифр на позиции больше чем 10
    while n1 is not None or n2 is not None:
        l1_position = 0 if n1 is None else n1.item
        l2_position = 0 if n2 is None else n2.item
        sum_on_position = overflow + l1_position + l2_position
        # сохраняем переполнение для последующих итераций
        overflow = sum_on_position // 10
        relult.add(sum_on_position % 10)
        if n1 is not None:
            n1 = n1.next_node
        if n2 is not None:
            n2 = n2.next_node
    if overflow > 0:
        relult.add(overflow)
    return relult


def swap_nodes_in_pairs(l1: LinkedList):
    """
     { p -> q -> r -> s } to { q -> p -> r -> s } 
    """
    # этот узел - фейковый, он в обменах не участвует
    dummy_head = ListNode(0, next_node=l1.head)
    cur_node = l1.head
    pre_node = dummy_head
    while (cur_node is not None) and (cur_node.next_node is not None):
        next_node = cur_node.next_node
        next_next_node = cur_node.next_node.next_node
        # перевешиваем указатели
        pre_node.next_node = next_node
        next_node.next_node = cur_node
        cur_node.next_node = next_next_node
        # финальный обмен - сдвигаем итератор
        pre_node = cur_node
        cur_node = next_next_node
    # подменяем указатель на голову списка (отбрасываем dummy)
    l1.set_head(dummy_head.next_node)
    return l1


## LeetCode realisation
class MyListNode:
    def __init__(self, val):
        self.val = val
        self.next = None


class MyLinkedList:
    def __init__(self):
        self.head = None
        self.tail = None
        self.len = 0

    def get(self, index: int) -> int:
        cur_node = self.head
        cur_index = 0
        while cur_index != index:
            if cur_node is None:
                return -1
            cur_index += 1
            cur_node = cur_node.next
        if cur_node is None:
            return -1
        return cur_node.val

    def addAtHead(self, val: int) -> None:
        new_node = MyListNode(val)
        if self.head is None:
            self.tail = new_node
        else:
            new_node.next = self.head
        self.len += 1
        self.head = new_node

    def addAtTail(self, val: int) -> None:
        # print('add tail')
        new_node = MyListNode(val)
        if self.tail is None:
            self.head = new_node
        else:
            self.tail.next = new_node
        self.len += 1
        self.tail = new_node

    def addAtIndex(self, index: int, val: int) -> None:
        if index > self.len:
            return
        new_node = MyListNode(val)
        # print(index, val)
        if index == 0:
            self.addAtHead(val)
            return
        cur_index = 0
        cur_node = self.head
        while cur_index != index - 1:
            if cur_node is None:
                raise ValueError('Incorrect index: out of range')
            cur_index += 1
            cur_node = cur_node.next
        self.len += 1
        if cur_node is None:  # at the end of the list
            self.addAtTail(val)
            return
        # for example zero
        # print('cur_node.val ', cur_node.val)
        new_node.next = cur_node.next
        if cur_node.next is None:
            self.tail = new_node
        cur_node.next = new_node
        return cur_node.val

    def deleteAtIndex(self, index: int) -> None:
        if index >= self.len:
            return
        self.len -= 1
        if index == 0:
            self.head = self.head.next
            return
        cur_index = 0
        cur_node = self.head
        prev_node = self.head
        while cur_index != index - 1:
            if cur_node is None:
                raise ValueError('Incorrect index: out of range')
            cur_index += 1
            cur_node = cur_node.next
            prev_node = cur_node
        if cur_node is None:  # at the end of the list
            self.len += 1
            return
        if cur_node.next is None:
            prev_node.next = None
            return
        cur_node.next = cur_node.next.next
        if cur_node.next is None:
            self.tail = cur_node
        return cur_node.val


def print_my_linked_list(input_list: MyLinkedList):
    cur_node = input_list.head
    print('head: %d, tail: %d, len %d' % (input_list.head.val, input_list.tail.val, input_list.len))
    cnt = 0
    while cur_node is not None:
        print(f'{cnt}: {cur_node.val}')
        cur_node = cur_node.next
        cnt += 1


def detectCycle(head: Optional[MyListNode]) -> Optional[MyListNode]:
    if head is None:
        return
    slow = head
    fast = head.next
    while slow != fast:
        if fast is None or fast.next is None:
            return
        if fast == slow:
            break
        fast = fast.next.next
        slow = slow.next
    if slow is None or fast is None:
        return
    while head != slow:
        head = head.next
        slow = slow.next
    return head


if __name__ == '__main__':
    # l1 = create_linked_list([3, 4, 1, 2])
    # swap_nodes_in_pairs(l1)
    # ---------------------------------------------------------
    # Your MyLinkedList object will be instantiated and called as such:
    linked_list = MyLinkedList()
    res = '\n'.join([f'linked_list.{i}({",".join([str(k) for k in j]) if len(j) > 0 else None})' for i, j in zip(
        ["MyLinkedList", "addAtHead", "addAtIndex", "addAtTail", "addAtTail", "addAtTail", "addAtIndex", "addAtTail",
         "addAtHead", "deleteAtIndex", "deleteAtIndex", "deleteAtIndex", "addAtIndex", "addAtTail", "get", "get",
         "addAtHead", "addAtTail", "addAtTail", "get", "addAtTail", "addAtTail", "deleteAtIndex", "deleteAtIndex",
         "addAtHead", "addAtTail", "addAtIndex", "get", "addAtTail", "addAtIndex", "addAtHead", "addAtTail",
         "addAtIndex", "get", "addAtHead", "addAtTail", "addAtIndex", "addAtHead", "addAtIndex", "addAtTail",
         "addAtHead", "addAtIndex", "addAtTail", "addAtHead", "deleteAtIndex", "get", "addAtIndex", "get", "addAtIndex",
         "addAtTail", "addAtTail", "get", "deleteAtIndex", "get", "addAtHead", "addAtTail", "addAtIndex", "addAtIndex",
         "addAtIndex", "addAtHead", "addAtTail", "addAtIndex", "deleteAtIndex", "addAtHead", "addAtHead", "addAtTail",
         "get", "addAtTail", "addAtIndex", "addAtHead", "deleteAtIndex", "addAtHead", "deleteAtIndex", "get", "get",
         "addAtTail", "addAtIndex", "get", "deleteAtIndex", "deleteAtIndex", "addAtHead", "addAtHead", "addAtIndex",
         "get", "addAtTail", "addAtHead", "addAtIndex", "get", "addAtHead", "deleteAtIndex", "deleteAtIndex",
         "deleteAtIndex", "addAtHead", "addAtTail", "get", "addAtHead", "addAtTail", "addAtHead", "addAtHead",
         "deleteAtIndex", "get", "addAtHead"],
        [[], [55], [1, 90], [51], [91], [12], [2, 72], [17], [82], [4], [7], [7], [5, 75], [54], [6], [2], [8], [35], [
            36], [10], [40], [43], [12], [3], [78], [89], [3, 41], [10], [96], [5, 37], [51], [26], [16, 91], [18], [
             11], [66], [22, 20], [44], [17, 16], [95], [2], [14, 2], [99], [51], [1], [11], [22, 99], [20], [25, 42], [
             72], [45], [2], [4], [32], [55], [84], [32, 64], [26, 14], [30, 80], [88], [51], [27, 71], [15], [8], [
             60], [37], [25], [96], [25, 53], [36], [8], [85], [42], [20], [34], [78], [42, 76], [26], [30], [39], [
             27], [93], [19, 75], [8], [24], [32], [25, 98], [21], [95], [18], [45], [24], [38], [8], [20], [83], [
             71], [78], [55], [29], [11], [84]]
    )])
    # print(res)
    linked_list.addAtHead(55)
    linked_list.addAtIndex(1, 90)
    linked_list.addAtTail(51)
    linked_list.addAtTail(91)
    linked_list.addAtTail(12)
    linked_list.addAtIndex(2, 72)
    linked_list.addAtTail(17)
    linked_list.addAtHead(82)
    linked_list.deleteAtIndex(4)
    linked_list.deleteAtIndex(7)
    linked_list.deleteAtIndex(7)
    # linked_list.addAtIndex(5, 75)
    # linked_list.addAtTail(54)
    # linked_list.get(6)
    # linked_list.get(2)
    # linked_list.addAtHead(8)
    # linked_list.addAtTail(35)
    # linked_list.addAtTail(36)
    # linked_list.get(10)
    # linked_list.addAtTail(40)
    # linked_list.addAtTail(43)
    # linked_list.deleteAtIndex(12)
    # linked_list.deleteAtIndex(3)
    # linked_list.addAtHead(78)
    # linked_list.addAtTail(89)
    # linked_list.addAtIndex(3, 41)
    # linked_list.get(10)
    # linked_list.addAtTail(96)
    # linked_list.addAtIndex(5, 37)
    # linked_list.addAtHead(51)
    # linked_list.addAtTail(26)
    # linked_list.addAtIndex(16, 91)
    # linked_list.get(18)
    # linked_list.addAtHead(11)
    # linked_list.addAtTail(66)
    # linked_list.addAtIndex(22, 20)
    # linked_list.addAtHead(44)
    # linked_list.addAtIndex(17, 16)
    # linked_list.addAtTail(95)
    # linked_list.addAtHead(2)
    # linked_list.addAtIndex(14, 2)
    # linked_list.addAtTail(99)
    # linked_list.addAtHead(51)
    # linked_list.deleteAtIndex(1)
    # linked_list.get(11)
    # linked_list.addAtIndex(22, 99)
    # linked_list.get(20)
    # linked_list.addAtIndex(25, 42)
    # linked_list.addAtTail(72)
    # linked_list.addAtTail(45)
    # linked_list.get(2)
    # linked_list.deleteAtIndex(4)
    # linked_list.get(32)
    # linked_list.addAtHead(55)
    # linked_list.addAtTail(84)
    # linked_list.addAtIndex(32, 64)
    # linked_list.addAtIndex(26, 14)
    # linked_list.addAtIndex(30, 80)
    # linked_list.addAtHead(88)
    # linked_list.addAtTail(51)
    # linked_list.addAtIndex(27, 71)
    # linked_list.deleteAtIndex(15)
    # linked_list.addAtHead(8)
    # linked_list.addAtHead(60)
    # linked_list.addAtTail(37)
    # linked_list.get(25)
    # linked_list.addAtTail(96)
    # linked_list.addAtIndex(25, 53)
    # linked_list.addAtHead(36)
    # linked_list.deleteAtIndex(8)
    # linked_list.addAtHead(85)
    # linked_list.deleteAtIndex(42)
    # linked_list.get(20)
    # linked_list.get(34)
    # linked_list.addAtTail(78)
    # linked_list.addAtIndex(42, 76)
    # linked_list.get(26)
    # linked_list.deleteAtIndex(30)
    # linked_list.deleteAtIndex(39)
    # linked_list.addAtHead(27)
    # linked_list.addAtHead(93)
    # linked_list.addAtIndex(19, 75)
    # linked_list.get(8)
    # linked_list.addAtTail(24)
    # linked_list.addAtHead(32)
    # linked_list.addAtIndex(25, 98)
    # linked_list.get(21)
    # linked_list.addAtHead(95)
    # linked_list.deleteAtIndex(18)
    # linked_list.deleteAtIndex(45)
    # linked_list.deleteAtIndex(24)
    # linked_list.addAtHead(38)
    # linked_list.addAtTail(8)
    # linked_list.get(20)
    # linked_list.addAtHead(83)
    # linked_list.addAtTail(71)
    # linked_list.addAtHead(78)
    # linked_list.addAtHead(55)
    # linked_list.deleteAtIndex(29)
    # linked_list.get(11)
    # linked_list.addAtHead(84)

    # print('item: ', linked_list.get(4))
    print_my_linked_list(linked_list)

    # linked_list.deleteAtIndex(2)
    # linked_list.addAtHead(6)
    # linked_list.addAtTail(4)
    # print('----------------------------\n', linked_list.get(4))
    # linked_list.addAtHead(4)
    # linked_list.addAtIndex(5, 0)
    # linked_list.addAtHead(6)
    # ----------------------------------------
    # ----------------------------------------
    # ----------------------------------------
    # index = 0
    # linked_list.addAtTail(3)
    # print_my_linked_list(linked_list)
    # print('----------\n', linked_list.get(0), '\n----------\n', linked_list.get(2), '\n----------\n')
    #
    # print_my_linked_list(linked_list)
    # print('----------------------------')
    # print_my_linked_list(linked_list)
    # obj.addAtHead(val)
    # obj.addAtTail(val)
    # obj.addAtIndex(index,val)
    # obj.deleteAtIndex(index)
