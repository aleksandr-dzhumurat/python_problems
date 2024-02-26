"""
https://leetcode.com/problems/lru-cache/submissions/
"""
from collections import OrderedDict


class LRUCache(object):

    def __init__(self, capacity):
        """
        :type capacity: int
        """
        self.cache = OrderedDict()
        self.max_size = capacity

    def get(self, key):
        """
        :type key: int
        :rtype: int
        """
        if key in self.cache:
            # Move the accessed item to the end
            value = self.cache.pop(key)
            self.cache[key] = value
            return value
        else:
            return -1

    def put(self, key, value):
        """
        :type key: int
        :type value: int
        :rtype: None
        """
        if key in self.cache:
            self.cache.pop(key)
        elif len(self.cache) >= self.max_size:
            # Remove the first item (the least recently used)
            self.cache.popitem(last=False)
        self.cache[key] = value


if __name__ == '__main__':
    # Your LRUCache object will be instantiated and called as such:
    commands = ["LRUCache", "put", "put", "get", "put", "get", "put", "get", "get", "get"]
    values = [[2], [1, 1], [2, 2], [1], [3, 3], [2], [4, 4], [1], [3], [4]]

    lru_cache = None
    results = []

    for i, val in enumerate(commands):
        command = commands[i]
        value = values[i]

        if command == "LRUCache":
            lru_cache = LRUCache(value[0])
            results.append(None)
        elif command == "put":
            lru_cache.put(value[0], value[1])
            results.append(None)
        elif command == "get":
            results.append(lru_cache.get(value[0]))

    print(results)