# nums = [3,2,4]
# target = 6
#
# num_items = len(nums)
# arr_hash_map = dict(
#     zip(
#         nums,
#         range(num_items)
#     )
# )
#
# print(arr_hash_map)
# print([i for i in range(num_items)])
#
# for i in range(num_items):
#     if ((target - nums[i]) in arr_hash_map) and (target != 2*nums[i]):
#         print (i, arr_hash_map[target - nums[i]])


# nums = [2,7,11,15]
# target_num = 9
# high = len(nums) - 1
# low = 0
# while low <= high:
#     if nums[low] + nums[high] > target_num:
#         high -= 1
#     elif nums[low] + nums[high] < target_num:
#         low += 1
#     else:
#         print(low, high)

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

input_str = "abcabcbb"
print(longest_substr_with_no_repeated(input_str))