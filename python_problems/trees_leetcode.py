from collections import deque

# Definition for a binary tree node.
class TreeNode(object):
    def __init__(self, val=0, left=None, right=None):
        self.val = val
        self.left = left
        self.right = right

def is_symmetric(root):
    """https://leetcode.com/problems/symmetric-tree/"""
    def is_mirror(t1, t2):
        if t1 is None and t2 is None:
            return True
        if t1 is None or t2 is None:
            return False
        
        return t1.val == t2.val and is_mirror(t1.right, t2.left) and is_mirror(t1.left, t2.right)

    if root is None:
        return True
    return is_mirror(root.left, root.right)


def maxDepth(root):
    """https://leetcode.com/problems/maximum-depth-of-binary-tree/"""
    if root is None:
        return 0
    left = 1 + maxDepth(root.left)
    right = 1 + maxDepth(root.right)
    return max(left, right)

def has_path_sum(root, target_sum):
    """https://leetcode.com/problems/path-sum/
    Given the root of a binary tree and an integer targetSum, return true if the tree has a root-to-leaf path such that adding up all the values equals targetSum.
    """
    def has_sum(root, target, cur_sum):
        cur_sum += root.val
        if cur_sum == target and root.left is None and root.right is None:
            return True
        if root.left is not None:
            if has_sum(root.left, target, cur_sum):
                return True
        if root.right is not None:
            if has_sum(root.right, target, cur_sum):
                return True
        return False
    if root is None:
        return False
    return has_sum(root, target_sum, 0)

def has_path_sum_ii(root, targetSum):
    """https://leetcode.com/problems/path-sum-ii/"""
    result = []
    def dfs(node, current_sum, path):
        if not node:
            return
        current_sum += node.val
        path.append(node.val)
        if not node.left and not node.right and current_sum == targetSum:
            result.append(path.copy())  # Append a copy to avoid modifying the original path
        dfs(node.left, current_sum, path)
        dfs(node.right, current_sum, path)
        path.pop()  # Backtrack by removing the current node from the path
    dfs(root, 0, [])
    return result


def lowest_common_ancestor(root, p, q):
    """https://leetcode.com/problems/lowest-common-ancestor-of-a-binary-tree/
    """
    if root is None:
        return None
    if root.val==p.val or root.val==q.val:
        return root
    left = lowest_common_ancestor(root.left, p, q)
    right = lowest_common_ancestor(root.right, p, q)

    if left is None and right is None:
        return None
    if left is not None and right is not None:
        return root
    if left is None:
        return right
    
    return left


def kthSmallest(root, k):
    """https://leetcode.com/problems/kth-smallest-element-in-a-bst/"""
    def inorder_traversal(node):
        if not node:
            return []
        return inorder_traversal(node.left) + [node.val] + inorder_traversal(node.right)

    values = inorder_traversal(root)

    return values[k - 1] if 0 < k <= len(values) else None

def serialize(root):
    """https://leetcode.com/problems/serialize-and-deserialize-binary-tree/"""
    def dfs(node):
        if not node:
            return 'None'
        return str(node.val) + ',' + dfs(node.left) + ',' + dfs(node.right)

    return dfs(root)

def deserialize(data):
    """https://leetcode.com/problems/serialize-and-deserialize-binary-tree/"""
    def dfs(values):
        if values[0] == 'None':
            values.pop(0)
            return None
        root = TreeNode(int(values.pop(0)))
        root.left = dfs(values)
        root.right = dfs(values)
        return root

    values = data.split(',')
    return dfs(values)


def maxPathSum(root):
    """https://leetcode.com/problems/binary-tree-maximum-path-sum/"""
    max_sum = float('-inf')

    def max_gain(node):
        global max_sum

        if not node:
            return 0

        left_gain = max(max_gain(node.left), 0)
        right_gain = max(max_gain(node.right), 0)

        path_sum = node.val + left_gain + right_gain

        max_sum = max(max_sum, path_sum)

        # Return the maximum gain for the current subtree
        return node.val + max(left_gain, right_gain)

    max_gain(root)
    return max_sum


def tree_level_order_traversal(root):
    """https://leetcode.com/problems/binary-tree-level-order-traversal/
    Breadth-First Search (BFS) explores a tree level by level. Starting from the root node,
    it explores all the nodes at the current level before moving on to the next level.
    It uses a queue data structure to keep track of the nodes to be visited next.
    """
    if not root:
        return []
    result = []
    queue = [root]
    while queue:
        level_size = len(queue)
        current_level = []
        for _ in range(level_size):
            current_node = queue.pop(0)
            current_level.append(current_node.val)

            if current_node.left:
                queue.append(current_node.left)
            if current_node.right:
                queue.append(current_node.right)
        result.append(current_level)
    return result


def tree_zig_zag_level_order(root):
        """https://leetcode.com/problems/binary-tree-zigzag-level-order-traversal/description/"""
        if not root:
            return []
        result = []
        queue = [root]
        level_index = 1

        while len(queue) > 0:
            level_size = len(queue)
            current_level = []
            for _ in range(level_size):
                current_node = queue.pop(0)
                current_level.append(current_node.val)
                if current_node.left:
                    queue.append(current_node.left)
                if current_node.right:
                    queue.append(current_node.right)
            if level_index % 2 ==0:
                result.append(current_level[::-1])
            else:
                result.append(current_level)
            level_index += 1
        return result

def postorderTraversal(root):
    """https://leetcode.com/problems/binary-tree-postorder-traversal/description/"""
    result = []
    if root:
        result.extend(postorderTraversal(root.left))
        result.extend(postorderTraversal(root.right))
        result.append(root.val)
    return result

def isSameTree(p, q):
    """https://leetcode.com/problems/same-tree/description/"""
    if not p and not q:
        return True
    if not p or not q:
        return False
    if p.val != q.val:
        return False
    return isSameTree(p.left, q.left) and isSameTree(p.right, q.right)

def isBalanced(root):
    """https://leetcode.com/problems/balanced-binary-tree/"""
    def height(node):
        if not node:
            return 0
        return 1 + max(height(node.left), height(node.right))

    def isBalancedHelper(node):
        if not node:
            return True
        left_height = height(node.left)
        right_height = height(node.right)

        return abs(left_height - right_height) <= 1 and \
               isBalancedHelper(node.left) and \
               isBalancedHelper(node.right)

    return isBalancedHelper(root)