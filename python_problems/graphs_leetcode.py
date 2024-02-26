def findCheapestPrice(n, flights, src, dst, k):
    """https://leetcode.com/problems/cheapest-flights-within-k-stops/"""
    from heapq import heappush, heappop
    from collections import defaultdict

    graph = defaultdict(list)
    for u, v, w in flights:
        graph[u].append((v, w))
    heap = [(0, 0, src)]
    visited = set()
    while heap:
        cost, stops, node = heappop(heap)
        if node == dst:
            return cost
        if stops <= k and (node, stops) not in visited:
            visited.add((node, stops))
            for neighbor, edge_cost in graph[node]:
                heappush(heap, (cost + edge_cost, stops + 1, neighbor))
    return -1

# Example usage
n = 4
flights = [[0, 1, 100], [1, 2, 100], [2, 0, 100], [1, 3, 600], [2, 3, 200]]
src = 0
dst = 3
k = 1

cheapest_price = findCheapestPrice(n, flights, src, dst, k)
print(f"Cheapest price: {cheapest_price}")