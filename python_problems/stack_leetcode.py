class MinStack:
    """https://leetcode.com/problems/min-stack/description/
    Input
    ["MinStack","push","push","push","getMin","pop","top","getMin"]
    [[],[-2],[0],[-3],[],[],[],[]]

    Output
    [null,null,null,null,-3,null,0,-2]

    complexity: o[1]
    """

    def __init__(self):
        self.stack = []
        self.min_stack = []

    def push(self, val):
        self.stack.append(val)
        if not self.min_stack or val <= self.min_stack[-1]:
            self.min_stack.append(val)

    def pop(self):
        if self.stack:
            popped = self.stack.pop()
            if popped == self.min_stack[-1]:
                self.min_stack.pop()

    def top(self):
        if self.stack:
            return self.stack[-1]

    def getMin(self):
        if self.min_stack:
            return self.min_stack[-1]


def isValid(s):
    """https://leetcode.com/problems/valid-parentheses/description/"""
    stack = []
    brackets = {'(': ')', '{': '}', '[': ']'}
    
    for char in s:
        if char in brackets.keys():
            stack.append(char)
        else:
            if not stack or brackets[stack.pop()] != char:
                return False
    return not stack

