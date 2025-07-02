graph = {
    'WA': ['NT', 'SA'], 'NT': ['WA', 'SA', 'Q'],
    'SA': ['WA', 'NT', 'Q', 'NSW', 'V'],
    'Q': ['NT', 'SA', 'NSW'], 'NSW': ['Q', 'SA', 'V'],
    'V': ['SA', 'NSW'], 'T': []
}

colors = ['Red', 'Green', 'Blue']

def is_valid(state, node, color):
    return all(state.get(n) != color for n in graph[node])

def solve(state):
    if len(state) == len(graph): return state
    node = next(v for v in graph if v not in state)
    for color in colors:
        if is_valid(state, node, color):
            state[node] = color
            result = solve(state)
            if result: return result
            del state[node]
    return None

solution = solve({})
for k in sorted(solution): print(f"{k} => {solution[k]}")
