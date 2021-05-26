import unittest

from python_problems.linked_list import (
    LinkedList,
    merge_sorted_lists,
    add_two_numbers,
    create_linked_list,
    swap_nodes_in_pairs,
)


class TestStringMethods(unittest.TestCase):

    def assertLinkedListEqual(self, l1: LinkedList, l2: LinkedList):
        current_l1 = l1.head
        current_l2 = l2.head
        while (current_l1 is not None and current_l2 is not None):
            if current_l1.item != current_l2.item:
                raise ValueError(f'{current_l1.item} != {current_l2.item}')
            current_l1 = current_l1.next_node
            current_l2 = current_l2.next_node
        if current_l1 is not None:
            raise ValueError(f'l1 is longer then l2')
        if current_l2 is not None:
            raise ValueError(f'l2 is longer then l1')

    def test_merge_sorted_lists(self):
        l1 = create_linked_list([5, 15, 32, 34])
        l2 = create_linked_list([8, 18, 22])

        origin = create_linked_list([5, 8, 15, 18, 22, 32, 34])
        self.assertLinkedListEqual(merge_sorted_lists(l1, l2), origin)
    
    def test_add_two_numbers(self):
        # test 1
        l1 = create_linked_list([2, 4, 3])
        l2 = create_linked_list([5, 6, 4])

        origin = create_linked_list([7, 0, 8])
        self.assertLinkedListEqual(add_two_numbers(l1, l2), origin)

        # test 2
        l1 = create_linked_list([9, 9])
        l2 = create_linked_list([1])

        origin = create_linked_list([0, 0, 1])
        self.assertLinkedListEqual(add_two_numbers(l1, l2), origin)

    def test_swap_nodes_in_pairs(self):
        l1 = create_linked_list([3, 4, 1, 2])

        origin = create_linked_list([4, 3, 2, 1])
        self.assertLinkedListEqual(swap_nodes_in_pairs(l1), origin)


if __name__=='__main__':
    unittest.main()
