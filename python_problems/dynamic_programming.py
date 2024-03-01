def house_robber(nums):
    """https://leetcode.com/problems/house-robber/description/"""
    if not nums:
        return 0
    n = len(nums)
    if n == 1:
        return nums[0]
    dp = [0] * n
    dp[0] = nums[0]
    dp[1] = max(nums[0], nums[1])
    for i in range(2, n):
        dp[i] = max(dp[i-1], dp[i-2] + nums[i])
    return dp[-1]

def best_time_to_buy_and_sell_stock(prices):
    """https://leetcode.com/problems/best-time-to-buy-and-sell-stock"""
    if not prices or len(prices) < 2:
        return 0
    min_price = prices[0]
    max_profit = 0
    for price in prices:
        min_price = min(min_price, price)
        current_profit = price - min_price
        max_profit = max(max_profit, current_profit)
    return max_profit

def best_time_to_buy_and_sell_stock_ii(prices):
    """https://leetcode.com/problems/best-time-to-buy-and-sell-stock-ii"""
    max_profit = 0
    for i in range(1, len(prices)):
        price_diff = prices[i] - prices[i - 1]
        if price_diff > 0:
            max_profit += price_diff
    return max_profit

def max_profit_transaction_fee(prices, fee):
    """https://leetcode.com/problems/best-time-to-buy-and-sell-stock-with-transaction-fee/"""
    n = len(prices)
    hold, cash = -prices[0], 0
    for i in range(1, n):
        hold = max(hold, cash - prices[i])
        cash = max(cash, hold + prices[i] - fee)
    return cash

def max_prfit_sell_cooldown(prices):
    """https://leetcode.com/problems/best-time-to-buy-and-sell-stock-with-cooldown/"""
    n = len(prices)

    # Define variables for holding stock, having no stock, and cooldown
    hold, cash, cooldown = float('-inf'), 0, 0

    for i in range(n):
        prev_hold = hold
        hold = max(hold, cooldown - prices[i])
        cooldown = cash
        cash = max(cash, prev_hold + prices[i])

    return max(cash, cooldown)


def climb_stairs(n):
    """https://leetcode.com/problems/climbing-stairs/description/"""
    if n == 1:
        return 1
    if n == 2:
        return 2
    dp = [0] * n
    dp[0] = 1
    dp[1] = 2
    for i in range(2, n):
        dp[i] = dp[i-1] + dp[i-2]
    return dp[-1]

def coin_change(coins, amount):
    """https://leetcode.com/problems/coin-change/"""
    dp = [float('inf')] * (amount + 1)
    dp[0] = 0
    for coin in coins:
        for i in range(coin, amount + 1):
            dp[i] = min(dp[i], dp[i - coin] + 1)
    if dp[amount] != float('inf'):
        res= dp[amount]
    else:
        res = -1
    return res

def unique_paths(m, n):
    """https://leetcode.com/problems/unique-paths/description/"""
    dp = [[0] * n for _ in range(m)]
    for i in range(m):
        dp[i][0] = 1
    for j in range(n):
        dp[0][j] = 1
    for i in range(1, m):
        for j in range(1, n):
            dp[i][j] = dp[i-1][j] + dp[i][j-1]
    return dp[-1][-1]

def longest_palindrome(s):
    """https://leetcode.com/problems/longest-palindromic-substring/"""
    n = len(s)
    dp = [[False] * n for _ in range(n)]

    # All substrings of length 1 are palindromes
    for i in range(n):
        dp[i][i] = True
    start, max_len = 0, 1
    for i in range(n - 1):
        if s[i] == s[i + 1]:
            dp[i][i + 1] = True
            start = i
            max_len = 2
    for length in range(3, n + 1):
        for i in range(n - length + 1):
            j = i + length - 1
            if dp[i + 1][j - 1] and s[i] == s[j]:
                dp[i][j] = True
                start = i
                max_len = length
    return s[start:start + max_len]

def trapping_rain_water(height):
    """https://leetcode.com/problems/trapping-rain-water/"""
    n = len(height)
    if n <= 2:
        return 0
    left, right = 0, n - 1
    left_max, right_max = 0, 0
    result = 0
    while left < right:
        if height[left] < height[right]:
            # Process left side
            if height[left] >= left_max:
                left_max = height[left]
            else:
                result += left_max - height[left]
            left += 1
        else:
            if height[right] >= right_max:
                right_max = height[right]
            else:
                result += right_max - height[right]
            right -= 1
    return result