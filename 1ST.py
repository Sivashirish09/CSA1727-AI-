import heapq

goal = [[1,2,3],[4,5,6],[7,8,0]]
moves = [(-1,0), (1,0), (0,-1), (0,1)]

def manhattan(puzzle):
    dist = 0
    for i in range(3):
        for j in range(3):
            val = puzzle[i][j]
            if val != 0:
                x, y = divmod(val-1, 3)
                dist += abs(i - x) + abs(j - y)
    return dist

def get_neighbors(puzzle):
    neighbors = []
    for i in range(3):
        for j in range(3):
            if puzzle[i][j] == 0:
                x, y = i, j
    for dx, dy in moves:
        nx, ny = x+dx, y+dy
        if 0 <= nx < 3 and 0 <= ny < 3:
            new = [row[:] for row in puzzle]
            new[x][y], new[nx][ny] = new[nx][ny], new[x][y]
            neighbors.append(new)
    return neighbors

def solve(puzzle):
    heap = [(manhattan(puzzle), 0, puzzle, [])]
    visited = set()
    while heap:
        est, cost, state, path = heapq.heappop(heap)
        if state == goal:
            return path + [state]
        visited.add(str(state))
        for neighbor in get_neighbors(state):
            if str(neighbor) not in visited:
                heapq.heappush(heap, (cost+1+manhattan(neighbor), cost+1, neighbor, path + [state]))
    return None

# Example input
start = [[1,2,3],[4,0,6],[7,5,8]]
solution = solve(start)

for step in solution:
    for row in step:
        print(row)
    print("----")
