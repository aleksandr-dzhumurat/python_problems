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

print(fib_plain_recursion(10))
