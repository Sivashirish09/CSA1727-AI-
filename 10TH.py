import heapq

def a_star(graph, start, goal, h):
    open_set = []
    heapq.heappush(open_set, (0 + h[start], 0, start, [start]))  # (f, g, node, path)
    visited = set()

    while open_set:
        f, g, current, path = heapq.heappop(open_set)

        if current == goal:
            return path, g

        if current in visited:
            continue
        visited.add(current)

        for neighbor, cost in graph[current]:
            if neighbor not in visited:
                new_g = g + cost
                new_f = new_g + h[neighbor]
                heapq.heappush(open_set, (new_f, new_g, neighbor, path + [neighbor]))

    return None, float('inf')
