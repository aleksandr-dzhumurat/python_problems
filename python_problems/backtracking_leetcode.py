def subsets(nums):
    """https://leetcode.com/problems/subsets/
    https://medium.com/@koray.kara98.kk/daily-leetcode-problems-exploring-the-coin-change-problem-subsets-6957b279c92d
    """
    def backtrack_subsets(nums, res, cur, index):
        if index > len(nums):
            return
        res.append(cur[:])
        for i in range(index, len(nums)):
            if nums[i] not in cur:
                cur.append(nums[i])
                backtrack_subsets(nums, res, cur, i)
                cur.pop() # backtrack
        return
    for _ in range(len(nums)):
        res = []
        curr = []
        backtrack_subsets(nums, res, curr, 0)
        return res

def letterCombinations(digits):
    """https://leetcode.com/problems/letter-combinations-of-a-phone-number/
    https://medium.com/nerd-for-tech/leetcode-letter-combinations-of-a-phone-number-f711ab47dfb1
    """
    def backtrack_letters(digits, digits_to_string, curr_str, res, item_index):
        if len(curr_str) == len(digits):
            res.append(curr_str.lower())
            return
        current_digit = digits[item_index]
        digit_chars = digit_to_string[current_digit]

        for _, i in enumerate(digit_chars):
            next_str = curr_str + digit_chars[i]
            backtrack_letters(digits, digits_to_string, next_str, res, item_index + 1)
        

    if len(digits) == 0:
        return []
    digit_to_string = {
        "2": "ABC",
        "3": "DEF",
        "4": "GHI",
        "5": "JKL",
        "6": "MNO",
        "7": "PQRS",
        "8": "TUV",
        "9": "WXYZ"
    }
    res = []
    backtrack_letters(digits, digit_to_string, "", res, 0)
    return res


def word_search(board, word):
    """https://leetcode.com/problems/word-search/
    """
    def backtrack_word(board, word, row_ind, col_ind, cur_str):
        NUM_NEIGBOURS = 4  # because of flat board
        dx = [0, 0, -1, 1]
        dy = [1, -1, 0, 0]

        if row_ind <0 or row_ind >= len(board) or col_ind<0 or col_ind >= len(board[row_ind]) or board[row_ind][col_ind] == ' ':
            return False
        cur_str += board[row_ind][col_ind]

        if len(cur_str) > len(word):
            return False
        if cur_str[len(cur_str) - 1] != word[len(cur_str) - 1]:
            return False
        if cur_str == word:
            return True
        
        tmp = board[row_ind][col_ind]
        board[row_ind][col_ind] = ' '
        for i in range(NUM_NEIGBOURS):
            if backtrack_word(board, word, row_ind+dx[i], col_ind+dy[i], cur_str):
                return True
        board[row_ind][col_ind] = tmp
        return False

    if len(word) == 0:
        return True
    n = len(board)
    for i in range(n):
        m = len(board[i])
        for j in range(m):
            if word[0] == board[i][j] and backtrack_word(board, word, i, j, ""):
                return True
    return False

def combination_sum(candidates, target):
    """https://leetcode.com/problems/combination-sum/"""
    def backtrack_sums(candidates, res, cur_comb, target, index, sum):
        if sum == target:
            res.append(cur_comb[:])
        elif sum < target:
            n = len(candidates)
            for i in range(index, n):
                cur_comb.append(candidates[i])
                backtrack_sums(candidates, res, cur_comb, target, i, sum+candidates[i])
                cur_comb.pop()
        return res
    return backtrack_sums(candidates, [], [], target, 0, 0)

def palindrome_partitioning(s: str):
    """https://leetcode.com/problems/palindrome-partitioning/

    partition is a collection of substrings that gives original string wjhen added together
    """
    def is_palindrome(s):
        l = 0
        r = len(s) - 1
        while(l < r):
            if s[l] != s[r]:
                return False
            l +=1
            r -= 1
        return True
    
    def palindrome_backtrack(s, current_substr, res):
        if len(s) == 0:
            res.append(current_substr[:])
        for i in range(1, len(s) + 1):
            cur_str = s[0:i]
            if is_palindrome(cur_str):
                current_substr.append(cur_str)
                palindrome_backtrack(s[i:], current_substr, res)
                current_substr.pop()

    res = []

    palindrome_backtrack(s, [], res)

    return res

def number_of_islands(grid):
    """https://leetcode.com/problems/number-of-islands/description/"""
    if not grid:
        return 0
    rows, cols = len(grid), len(grid[0])
    islands_count = 0

    def dfs(row, col):
        if row < 0 or row >= rows or col < 0 or col >= cols or grid[row][col] == '0':
            return
        grid[row][col] = '0'
        # Explore adjacent land cells
        dfs(row - 1, col)  # Up
        dfs(row + 1, col)  # Down
        dfs(row, col - 1)  # Left
        dfs(row, col + 1)  # Right
    for row in range(rows):
        for col in range(cols):
            if grid[row][col] == '1':
                islands_count += 1
                dfs(row, col)
    return islands_count

def remove_invalid_parenthesis(s):
    """https://leetcode.com/problems/remove-invalid-parentheses/"""
    from collections import deque

    def isValid(string):
        count = 0
        for char in string:
            if char == '(':
                count += 1
            elif char == ')':
                count -= 1
            if count < 0:
                return False
        return count == 0
        
    result = []
    visited = set()
    queue = deque([s])
    found = False
    while queue:
        current_str = queue.popleft()
        if isValid(current_str):
            result.append(current_str)
            found = True
        if found:
            continue
        for i, char in enumerate(current_str):
            if char.isalpha():
                continue
            next_str = current_str[:i] + current_str[i + 1:]
            if next_str not in visited:
                visited.add(next_str)
                queue.append(next_str)
    return result