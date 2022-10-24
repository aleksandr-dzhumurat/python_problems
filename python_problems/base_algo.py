#!/bin/python3

import math
import os
import random
import re
import sys
from collections import defaultdict


# Complete the staircase function below.
def staircase(n):
    buf = ''
    for i in range(n):
        for j in range(n):
            if j+1 < n-i:
                buf += ' '
            else:
                buf += '#'
        buf += '\n'
    buf += '\n'
    print(buf)


def miniMaxSum(arr):
    min_sum, max_sum = sum(arr), -sum(arr)
    for i in range(len(arr)):
        partial_sum = sum(arr[0:i]) + sum(arr[i+1:])
        if partial_sum < min_sum:
            min_sum = partial_sum
        if partial_sum > max_sum:
            max_sum = partial_sum
    print(f'{min_sum} {max_sum}')


def two_sum(nums, target):
    """
    Given an array of integers, return indices of the two numbers such that they add up to a specific target.

    You may assume that each input would have exactly one solution, and you may not use the same element twice.

    Важно: в массиве не должно быть повторений
    """
    num_items = len(nums)
    arr_hash_map = dict(
        zip(
            nums,
            range(num_items)
        )
    )
    for i in range(num_items):
        if target - nums[i] in arr_hash_map:
            return (i, arr_hash_map[target - nums[i]])


def timeConversion(s):
    #
    # Write your code here.
    #
    shift = 0
    if s[-2:]=='PM':
        shift = 12
    #print(s[-2:], s[:-2])
    h, m, s = s[:-2].split(':')
    h = int(h)
    h = h + shift
    if h < 10:
        h = f'0{h}'
    return f'{h}:{m}:{s}'


def bin_search(arr: list, target: int):
    sorted_arr = sorted(arr)
    N = len(sorted_arr)
    low = 0
    high = N - 1
    while low <= high:
        mid_index = (high + low) // 2
        mid_elem = sorted_arr[mid_index] # для целей дебаггинга сохраняем в отдельную переменную
        if target < mid_elem:
            high = mid_index - 1
        elif target > mid_elem:
            low = mid_index + 1
        else:
            return mid_index


def two_sum_sorted(nums, target_num):
    """
    For each element x, we could look up if target – x exists in O(log n) time by applying
    binary search over the sorted array. Total runtime complexity is O(n log n).
    """
    N = len(nums)
    for current_index in range(N):
        second_index = bin_search(nums[current_index:], target=target_num - nums[current_index])
        if second_index is not None:
            return (current_index, current_index + second_index)


def two_sum_sorted_pointers(nums, target_num):
    high = len(nums) - 1
    low = 0
    while low <= high:
        if nums[low] + nums[high] > target_num:
            high -= 1
        elif nums[low] + nums[high] < target_num:
            low += 1
        else:
            return (low, high)


class TwoSumClass:
    def __init__(self):
        self.storage = dict()

    def add(self, item):
        if self.storage.get(item, 0) == 0:
            self.storage[item] = 1
        else:
            self.storage[item] += 1

    def find(self, target) -> bool:
        for entry in self.storage:
            second_entry = target - entry
            if second_entry == entry and self.storage[entry] == 2:
                return True
            elif second_entry in self.storage:
                return True
        return False


def valid_palindrome(input_str):
    begin = 0
    end = len(input_str) - 1
    while begin < end:
        while begin < end and not (input_str[begin].isdigit() or input_str[begin].isalpha()):
            begin += 1
        while begin < end and not (input_str[end].isdigit() or input_str[end].isalpha()):
            end -= 1
        if input_str[begin].lower() != input_str[end].lower():
            return False
        begin += 1
        end -= 1
    return True


def strstr_bruteforce(needle, haystack: str):
    """
        Returns the index of the first occurrence of needle in haystack, or –1
        if needle is not part of haystack.
    """
    i = 0
    j = 0
    str_len = len(haystack)
    pattern_len = len(needle)
    for i in range(str_len + 1):
        # проверяем вхождение паттерна в внутреннем цикле
        for j in range(pattern_len + 1):
            if j == pattern_len:
                # если прошлись по всей длин паттерна и не выпали из цикла - это матч 
                return i
            elif i + j == str_len:
                return -1
            elif haystack[i + j] != needle[j]:
                str_char = haystack[i + j]
                pattern_char = needle[j]
                break


def longest_substr_with_no_repeated(s):
    if len(s) == 0: return 0
    start = maxLength = 0
    usedChars = {}
    for i in range(len(s)):
        if s[i] in usedChars and start <= usedChars[s[i]]:
            start = usedChars[s[i]] + 1
        else:
            maxLength = max(maxLength, i - start + 1)
        usedChars[s[i]] = i
    return maxLength


def reverse_str(str_list: list, begin: int, end: int):
    for i in range((end - begin) // 2):
        swap = str_list[begin + i]
        str_list[begin + i] = str_list[end - i -1]
        str_list[end - i -1] = swap
    return ''.join(str_list)


def reverse_sentence(input_str: str):
    reversed_words = list(input_str)
    i = 0
    for j in range(len(input_str) + 1):
        if j == len(reversed_words) or reversed_words[j] == ' ':
            reverse_str(reversed_words, i, j)
            i = j + 1
    return ''.join(reversed_words)


def find_missing_ranges(values, start, end):
    """
        ['2->2', '4->49', '51->74', '76->99']

        find_missing_ranges([0, 1, 3, 50, 75], 0, 99))
    """
    ranges = []
    prev = start - 1
    for i in range(len(values)+1):
        if i == len(values):
            current = end + 1
        else:
            current = values[i]
        if current - prev >= 2:
            ranges.append(f'{prev+1}->{current - 1}')
        prev = current
    return ranges


def expand_around_center(s, l, r):
    while l >= 0 and r < len(s) and s[l] == s[r]:
        l -= 1
        r += 1
    return s[l+1:r]


def longest_palindrome(s):
    """Given a string S, find the longest palindromic substring in S.
    You may assume that the maximum length of S is 1000, and there exists one unique longest palindromic substring.
    """
    longest = ''
    for i in range(len(s)):
        # нечётный палиндром: aba
        tmp = expand_around_center(s, i, i)
        if len(tmp) > len(longest):
            longest = tmp
        # чётный палиндром: abba
        tmp = expand_around_center(s, i, i+1)
        if len(tmp) > len(longest):
            longest = tmp
    return longest


def is_one_edit_distance(s: str, t: str) -> bool:
    """Given two strings S and T, determine if they are both one edit distance apart."""
    small_str_len, long_str_len = len(s), len(t)
    if (small_str_len > long_str_len):
        return is_one_edit_distance(t, s)
    s = list(s)
    t = list(t)
    if long_str_len - small_str_len > 1:
        return False
    i = 0
    shift = long_str_len - small_str_len  # тут либо 1, либо 0
    while i < small_str_len and s[i]==t[i]:
        i += 1
    if i == small_str_len:
        return shift > 0
    if shift == 0:
        i += 1
    while i < small_str_len and s[i] == s[i + shift]:
        i += 1
    return i==small_str_len


def integer_reverse(x: int):
    reversed = 0
    neg = 1
    if x < 0:
        x = x * (-1)
        neg = -1
    while x != 0:
        reversed = reversed * 10 + x % 10
        x = x // 10
    if neg < 0:
        reversed = reversed * -1
    return reversed


def plus_one(digits: list):
    num_digits = len(digits)
    in_mind = 1
    for i in range(num_digits):
        digits[-(i + 1)] = digits[-(i + 1)] + in_mind
        in_mind = digits[-(i + 1)] // 10
        digits[-(i + 1)] = digits[-(i + 1)] % 10
    if in_mind == 1:
        digits = [1] + digits
    return digits


def is_integer_palindrome(x: int):
    if x < 0:
        return False
    div = 1
    while x // div >= 10:
        div = div * 10
    while x != 0:
        l = x // div
        r = x % 10
        if l != r:
            return False
        x = (x % div) // 10
        div = div // 100
    return True


if __name__ == '__main__':
    # print(timeConversion('07:05:45PM'))
    # print(timeConversion('17:05:45PM'))
    # print(timeConversion('07:05:45AM'))
    # print(timeConversion('17:05:45AM'))
    # print(two_sum([2, 7, 11, 15, 3, 6], 9))
    # print(bin_search([2, 7, 11, 15, 3, 6], target=15))
    # print(bin_search([2, 7, 11, 15, 3, 6], target=2))
    # print(two_sum_sorted([2, 3, 6, 7, 11, 15], target_num=9))
    # print(two_sum_sorted_pointers([2, 3, 6, 7, 11, 15], target_num=9))
    # two_sum = base_algo.TwoSumClass(); two_sum.add(1); two_sum.add(3); two_sum.add(5); two_sum.find(4)
    # valid_palindrome('race a car'); valid_palindrome('rac a     car')
    # print(strstr_bruteforce('an', 'banan')); print(strstr_bruteforce('na', 'banan'))
    # print(find_missing_ranges([0, 1, 3, 50, 75], 0, 99))
    # print(longest_palindrome("hgtabgbaggg"))
    # print(is_one_edit_distance("baggg", "bagg"))
    # ------- MATH ------- #
    # print(integer_reverse(-1234)
    # print(plus_one([9,9,9,9]))'
    # is_integer_palindrome(101)
    # ------- LIMKED LIST ------- #
    pass
