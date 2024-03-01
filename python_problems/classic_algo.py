def fib(n):
    """https://leetcode.com/problems/fibonacci-number/description/"""
    def fibonacci_memoization(n, memo={}):
        if n in memo:
            return memo[n]
        if n <= 1:
            return n
        memo[n] = fibonacci_memoization(n - 1, memo) + fibonacci_memoization(n - 2, memo)
        return memo[n]

    result = fibonacci_memoization(n)
    return result

def fib_plain_recursion(n):
    """https://leetcode.com/problems/fibonacci-number/description/"""
    if n <= 1:
        return n
    else:
        return fib_plain_recursion(n-1) + fib_plain_recursion(n-2)

def guess(num):
    pick = 6
    if num == pick:
        return 0
    elif num < pick:
        return 1
    else:
        return -1
  
def guessNumber(n):
    """https://leetcode.com/problems/guess-number-higher-or-lower/"""
    left, right = 1, n
    while left <= right:
        mid = left + (right - left) // 2
        result = guess(mid)
        if result == 0:
            return mid
        elif result == 1:
            left = mid + 1
        else:
            right = mid - 1
    return -1

def searchMatrix(matrix, target):
    """https://leetcode.com/problems/search-a-2d-matrix/"""
    if not matrix or not matrix[0]:
        return False
    
    rows, cols = len(matrix), len(matrix[0])
    left, right = 0, rows * cols - 1
    
    while left <= right:
        mid = left + (right - left) // 2
        mid_elem = matrix[mid // cols][mid % cols]
        
        if mid_elem == target:
            return True
        elif mid_elem < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return False
    
def search_rotated_sorted_array(nums, target):
    """https://leetcode.com/problems/search-in-rotated-sorted-array/"""
    left, right = 0, len(nums) - 1
    while left <= right:
        mid = (left + right) // 2
        if nums[mid] == target:
            return mid
        if nums[left] <= nums[mid]:
            if nums[left] <= target < nums[mid]:
                right = mid - 1
            else:
                left = mid + 1
        else:
            if nums[mid] < target <= nums[right]:
                left = mid + 1
            else:
                right = mid - 1
    return -1

def search_rotated_sorted_array_ii(nums, target):
        """https://leetcode.com/problems/search-in-rotated-sorted-array-ii"""
        left, right = 0, len(nums) - 1
        while left <= right:
            mid = (left + right) // 2
            if nums[mid] == target:
                return True
            while left < mid and nums[left] == nums[mid]:
                left += 1
            while right > mid and nums[right] == nums[mid]:
                right -= 1
            if nums[left] <= nums[mid]:
                if nums[left] <= target < nums[mid]:
                    right = mid - 1
                else:
                    left = mid + 1
            else:
                if nums[mid] < target <= nums[right]:
                    left = mid + 1
                else:
                    right = mid - 1
        return False


def find_min(nums):
    """https://leetcode.com/problems/find-minimum-in-rotated-sorted-array/"""
    left, right = 0, len(nums) - 1
    while left < right:
        mid = (left + right) // 2
        if nums[mid] > nums[right]:
            left = mid + 1
        else:
            right = mid
    return nums[left]

def fourSum(nums, target):
    """https://leetcode.com/problems/4sum/"""
    nums.sort()
    result = []
    for i in range(len(nums) - 3):
        if i > 0 and nums[i] == nums[i - 1]:
            continue
        for j in range(i + 1, len(nums) - 2):
            if j > i + 1 and nums[j] == nums[j - 1]:
                continue
            left, right = j + 1, len(nums) - 1
            while left < right:
                current_sum = nums[i] + nums[j] + nums[left] + nums[right]    
                if current_sum == target:
                    result.append([nums[i], nums[j], nums[left], nums[right]])
                    
                    while left < right and nums[left] == nums[left + 1]:
                        left += 1
                    while left < right and nums[right] == nums[right - 1]:
                        right -= 1
                    left += 1
                    right -= 1
                elif current_sum < target:
                    left += 1
                else:
                    right -= 1
    return result

def merge_intervals(intervals):
    """https://leetcode.com/problems/merge-intervals/"""
    if not intervals:
        return []
    intervals.sort(key=lambda x: x[0])
    merged_intervals = [intervals[0]]
    for i in range(1, len(intervals)):
        current_start, current_end = intervals[i]
        previous_start, previous_end = merged_intervals[-1]
        if current_start <= previous_end:
            merged_intervals[-1] = [previous_start, max(current_end, previous_end)]
        else:
            merged_intervals.append([current_start, current_end])
    return merged_intervals

def top_k_frequent_words(words, k):
    """https://leetcode.com/problems/top-k-frequent-words/"""
    import heapq
    from collections import Counter

    word_count = Counter(words)
    max_heap = [(-freq, word) for word, freq in word_count.items()]
    heapq.heapify(max_heap)
    result = []
    for _ in range(k):
        result.append(heapq.heappop(max_heap)[1])
    return result

def mean_sliding_window(nums, k):
    """https://leetcode.com/problems/sliding-window-median/"""
    import heapq
    
    result = []
    for i in range(len(nums) - k + 1):
        window = sorted(nums[i:i + k])  # Sort the current window
        mid = len(window) // 2
        if k % 2 == 0:  # Even size window
            median = (window[mid - 1] + window[mid]) / 2.0
        else:  # Odd size window
            median = window[mid]
        result.append(median)
    return result

def maximum_sliding_window(nums, k):
    """https://leetcode.com/problems/sliding-window-maximum"""
    from collections import deque

    result = []
    window = deque()
    # Fill the initial window
    for i in range(k):
        while window and nums[i] >= window[-1]:
            window.pop()
        window.append(nums[i])
    for i in range(k, len(nums)):
        result.append(window[0])
        # Remove elements outside the window
        if window[0] == nums[i - k]:
            window.popleft()
        # Update the window
        while window and nums[i] >= window[-1]:
            window.pop()
        window.append(nums[i])
    result.append(window[0])  # Append the last window's maximum
    return result

if __name__ == '__main__':
    # print(fib_plain_recursion(10))
