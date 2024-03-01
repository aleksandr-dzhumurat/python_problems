#!/bin/python3

import math
from collections import defaultdict


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


def twoSum(nums, target):
    """https://leetcode.com/problems/two-sum/
    """
    num_items = len(nums)
    arr_hash_map = dict(
        zip(
            nums,
            range(num_items)
        )
    )
    for i in range(num_items):
        diff = target - nums[i]
        # special case where target == num // 2 for some num
        if diff in arr_hash_map:
            if i!=arr_hash_map[target - nums[i]]:
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
    """https://leetcode.com/problems/binary-search"""
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
    return -1


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
    """https://leetcode.com/problems/longest-substring-without-repeating-characters/"""
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

def move_zeroes(nums):
    """
    :type nums: List[int]
    :rtype: None Do not return anything, modify nums in-place instead.
    """
    j = 0
    for num in nums:
        if num != 0:
            nums[j] = num
            j += 1
    while j < len(nums):
        nums[j] = 0
        j += 1
    return nums

def numRescueBoats(people, limit):
    """
    :type people: List[int]
    :type limit: int
    :rtype: int
    """
    people = sorted(people)
    light_p = 0
    heavy_p = len(people) - 1
    boats = 0
    while light_p <= heavy_p:
        if people[light_p] + people[heavy_p] <= limit:
            light_p += 1
            heavy_p -= 1
            boats +=1
        else:
            heavy_p -= 1
            boats +=1
    return boats

def validMountainArray(arr):
    """
    https://leetcode.com/problems/valid-mountain-array/

    :type arr: List[int]
    :rtype: bool
    """
    i = 1
    while i < len(arr) and  arr[i] > arr[i-1]:
        i += 1
    if i==1 or i == len(arr):
        return False
    while i < len(arr) and arr[i] < arr[i-1]:  # check len at first!!
        i += 1
    if i == len(arr):
        return True
    else:
        return False

def maxArea(height):
    """
    https://leetcode.com/problems/container-with-most-water/
    :type height: List[int]
    :rtype: int
    """
    def eval_area(i, j):
        return min(height[i], height[j])*(j - i)
    left = 0
    right = len(height) - 1
    max_vol = 0
    while left < right:
        vol = eval_area(left, right)
        max_vol = max(max_vol, vol)
        if height[left] < height[right]:
            left += 1
        else:
            right -= 1
    return max_vol

def searchRange(nums, target):
    """
    https://leetcode.com/problems/find-first-and-last-position-of-element-in-sorted-array/
    :type nums: List[int]
    :type target: int
    :rtype: List[int]
    """
    first_entry = -1
    last_entry = -1
    for i in range(len(nums)):
        if nums[i] == target:
            first_entry = i
            break
    last_entry = first_entry
    if first_entry >=0:
        while last_entry < len(nums) and nums[last_entry] == target:
            last_entry += 1
        last_entry -= 1
    return [first_entry, last_entry]

def isBadVersion(version, ground_truth):
    if version >= ground_truth:
        return True
    else:
        return False

def firstBadVersion_array(n: int, ground_truth: int):
    """ https://leetcode.com/problems/first-bad-version/
    :type n: int
    :rtype: int
    """
    sorted_arr = list(range(1,n+1))
    N = len(sorted_arr)
    low = 0
    high = N - 1
    while low <= high:
        if low == high:
            break
        mid_index = (high + low) // 2
        mid_elem = sorted_arr[mid_index]
        if isBadVersion(mid_elem):
            high = mid_index
        else:
            low = mid_index + 1
    if not isBadVersion(sorted_arr[mid_index]):
        mid_index += 1  # 2, 2
    return sorted_arr[mid_index]

def firstBadVersion(n: int, ground_truth: int):
    """ https://leetcode.com/problems/first-bad-version/
    :type n: int
    :rtype: int
    """
    low = 1
    high = n
    while low <= high:
        if low == high:
            break
        mid_elem = (high + low) // 2
        if isBadVersion(mid_elem, ground_truth):
            high = mid_elem
        else:
            low = mid_elem + 1
    if not isBadVersion(mid_elem, ground_truth):
        mid_elem += 1  # 2, 2
    return mid_elem

def missingNumber(self, nums):
    """ https://leetcode.com/problems/missing-number
    TODO: improve, based on Gauss formula progression)
    :type nums: List[int]
    :rtype: int
    """
    n = len(nums)
    num_exists ={}
    for i in nums:
        num_exists[i] = True
    for i in range(n+1):
        if not i in num_exists:
            return i

def countPrimes(n: int):
    """https://leetcode.com/problems/count-primes/
    sieve of eratosthenes
    :type n: int
    :rtype: int
    """
    if n <=2: return 0
    is_prime = [True] * n
    is_prime[0] = is_prime[1] = False
    for i in range(2, int(math.ceil(math.sqrt(n)))):
        if is_prime[i]:
            for k in range(i * i, n, i):
                is_prime[k] = False
    return sum(is_prime)

def singleNumber(nums):
    """ https://leetcode.com/problems/single-number/"""
    uniq_nums = set(nums)
    check_sum = sum(uniq_nums) * 2
    actual_sum = sum(nums)

    return check_sum - actual_sum

def singleNumberMemoryLimit(nums):
    """ XORing a number with itself results in 0, and XORing any number with 0 results in the number itself.
    By XORing all the elements in the array, the duplicates will cancel each other out, leaving only the single element.
    """
    result = 0
    for num in nums:
        result ^= num
    return result

def judgeCircle(moves):
    """https://leetcode.com/problems/robot-return-to-origin/
    :type moves: str
    :rtype: bool
    """
    x, y = (0, 0)
    for m in moves:
        if m == 'U':
            y +=1 
        elif m == 'R':
            x +=1
        elif m == 'D':
            y -= 1
        elif m == 'L':
            x -=1
    if x == 0 and y ==0:
        return True
    else:
        return False

def addBinary(a, b):
    """ https://leetcode.com/problems/add-binary/
    :type a: str
    :type b: str
    :rtype: str
    """
    i = len(a) - 1
    j = len(b) - 1
    res = ""
    carry = 0
    while (i>=0 or j>=0 or carry==1):
        cur_sum = carry
        if i>=0:
            cur_sum  += int(a[i])
            i -= 1
        if j>=0:
            cur_sum  += int(b[j])
            j -= 1
        res = str(cur_sum % 2) + res
        carry = cur_sum // 2
    return res

def containsDuplicate(nums):
    """https://leetcode.com/problems/contains-duplicate/
    :type nums: List[int]
    :rtype: bool
    """
    entries = {}
    for i in nums:
        if i in entries:
            return True
        entries[i] = True

def majorityElement(nums):
    """https://leetcode.com/problems/majority-element/submissions/
    :type nums: List[int]
    :rtype: int
    """
    majority_element_count = int(math.floor(len(nums) / 2))
    res = {}
    for num in nums:
        res[num] = res.get(num, 0) + 1
        if res[num] > majority_element_count:
            return num

def majorityElementBoyerMoor(nums):
    """https://leetcode.com/problems/majority-element/submissions/
    Only if you can garatee that majority element exists

    :type nums: List[int]
    :rtype: int
    """
    candidate = nums[0]
    cnt = 0
    for num in nums:
        if cnt == 0:
            candidate = num
        if candidate == num:
            cnt += 1
        else:
            cnt -= 1
    return candidate

def groupAnagrams(strs):
    """https://leetcode.com/problems/group-anagrams/"""
    res = {}
    for token in strs:
        key = ''.join(sorted(token))
        res[key] = res.get(key, []) + [token]
    return [res[k] for k in res]

def is_anagram(s, t):
    """https://leetcode.com/problems/valid-anagram/"""
    if len(s) != len(t):
        return False
    char_count = {}
    for char in s:
        char_count[char] = char_count.get(char, 0) + 1
    for char in t:
        if char not in char_count or char_count[char] == 0:
            return False
        char_count[char] -= 1
    return all(count == 0 for count in char_count.values())

def findAnagrams(s, p):
    """https://leetcode.com/problems/find-all-anagrams-in-a-string"""
    if len(p) > len(s):
        return []
    result = []
    window_size = len(p)
    p_count = [0] * 26
    window_count = [0] * 26
    for char in p:
        p_count[ord(char) - ord('a')] += 1
    for i in range(window_size):
        window_count[ord(s[i]) - ord('a')] += 1
    if window_count == p_count:
        result.append(0)
    for i in range(1, len(s) - window_size + 1):
        window_count[ord(s[i - 1]) - ord('a')] -= 1  # Remove the leftmost character
        window_count[ord(s[i + window_size - 1]) - ord('a')] += 1  # Add the rightmost character
        if window_count == p_count:
            result.append(i)
    return result

def fourSumCount(nums1, nums2, nums3, nums4):
    """https://leetcode.com/problems/4sum-ii/
    :type nums1: List[int]
    :type nums2: List[int]
    :type nums3: List[int]
    :type nums4: List[int]
    :rtype: int
    """
    input_len = len(nums1)
    sums_12 = {}
    sums_34 = {}
    for i in range(input_len):
        for j in range(input_len):
            k1 = nums1[i] + nums2[j]
            # res = sums_12.get(k1, [])
            res = sums_12.get(k1, 0)
            # sums_12[k1] = res + [(i, j)]
            sums_12[k1] = res + 1

            k2 = nums3[i] + nums4[j]
            # res = sums_34.get(k2, [])
            res = sums_34.get(k2, 0)
            # sums_34[k2] = res + [(i, j)]
            sums_34[k2] = res + 1
    cnt = 0
    for k in sums_12:
        # cnt += len(sums_34.get(-1 * k, [])) * len(sums_12[k])
        if -k in sums_34:
            cnt += sums_12[k] * sums_34[-k]
    return cnt

def find_min_cycle_shifted(arr: list):
    """https://leetcode.com/problems/find-minimum-in-rotated-sorted-array/
    """
    left = 0
    right = len(arr) - 1
    while left < right:
        mid = left + (right - left) // 2 
        if arr[mid] > arr[right]:
            left = mid + 1
        else:
            right = mid
    return arr[left]


def nextGreatestLetter(letters, target):
    """https://leetcode.com/problems/find-smallest-letter-greater-than-target"""
    n = len(letters)
    if letters[-1] <= target:
        return letters[0]
    low = 0
    high = n - 1
    while low < high:
        mid = (low + high) // 2
        if letters[mid] <= target:
            low = mid + 1
        else:
            high = mid
    return letters[low]


def minWindow(s, t):
    """https://leetcode.com/problems/minimum-window-substring/
    :type s: str
    :type t: str
    :rtype: str
    """
    res = ""
    if len(s) < len(t):
        return res
    s_hashmap, pattern_hashmap = defaultdict(lambda: 0), defaultdict(lambda: 0)
    for i in t:  # same as Counter BTW
        pattern_hashmap[i] += 1
    l, r, cnt, start_index = 0, 0, 0, -1
    min_len = float('inf')
    for r in range(len(s)): # window exanding
        cur_char  = s[r]
        s_hashmap[cur_char] += 1
        if cur_char in pattern_hashmap and s_hashmap[cur_char] <= pattern_hashmap[cur_char]:
            cnt += 1
        if cnt == len(t):  #window collapsing
            while s[l] not in pattern_hashmap or s_hashmap[s[l]] > pattern_hashmap[s[l]]:
                left_char = s[l]
                if left_char in pattern_hashmap and s_hashmap[left_char] > pattern_hashmap[left_char]:
                    s_hashmap[left_char] -=  1
                l += 1  # more then enough characters still in  window
            window_len = r - l + 1
            if min_len > window_len:
                min_len = window_len
                start_index = l
    if start_index != -1:
        res = s[start_index:start_index + min_len]
    return res


if __name__ == '__main__':
    # print(timeConversion('07:05:45PM'))
    # print(timeConversion('17:05:45PM'))
    # print(timeConversion('07:05:45AM'))
    # print(timeConversion('17:05:45AM'))
    # print(twoSum([3,2,4], 6))# print(two_sum([2, 7, 11, 15, 3, 6], 9))
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
    # print(find_min_cycle_shifted([3, 2])) # [2,4,5,0,1]
    # ------- MATH ------- #
    # print(integer_reverse(-1234)
    # print(plus_one([9,9,9,9]))'
    # is_integer_palindrome(101)
    # ------- LIMKED LIST ------- #
    pass
    #----------------
    # print(move_zeroes([0, 1, 0, 3, 12]))
    # print(numRescueBoats([1,2], 3))
    # print(validMountainArray([0,5,3,1]))
    # print(maxArea([5, 9, 2, 1, 4]))
    # print(searchRange([10,11,11,11,14,15], 11))
    # print(firstBadVersion(2, 2)) # print(firstBadVersion(5, 4)) # print(firstBadVersion(10, 3))
    # print(countPrimes(10))
    # print(singleNumber([4,1,2,1,2]))
    # print(judgeCircle("UD"))
    # print(addBinary(a="1010", b = "1011")) # print(addBinary(a="11", b = "1"))
    # print(containsDuplicate([2, 1, 3, 1]))
    # print(majorityElement([0]))
    # print(groupAnagrams(["eat","tea","tan","ate","nat","bat"]))
    # assert fourSumCount(nums1 = [-1,1,1,1,-1], nums2 =  [0,-1,-1,0,1] , nums3 = [-1,-1,1,-1,-1] , nums4 = [0,1,0,-1,-1]) == 132
    print(minWindow(s="ADOBECODEBANC", t="ABC"))
    assert minWindow(s="ADOBECODEBANC", t="ABC") == "BANC"
