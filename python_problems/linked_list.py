from typing import Any, Optional

class LinkedListIterator :
    def __init__( self, iterable_linked_list):
        self._linked_list_head = iterable_linked_list
        self._curItem = iterable_linked_list._head
    
    def __iter__( self ):
        """Метод будет вызван при создании нового итератора"""
        return self

    def __next__( self ):
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
    """
        
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


if __name__ == '__main__':
    l1 = create_linked_list([3, 4, 1, 2])
    swap_nodes_in_pairs(l1)