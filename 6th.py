def vacuum(env, loc):
    print(f"Start: {env}, Vacuum at {loc}")
    if env[loc] == 'Dirty':
        print("Action: Suck")
        env[loc] = 'Clean'
    move = 'B' if loc == 'A' else 'A'
    print(f"Action: Move to {move}")
    if env[move] == 'Dirty':
        print("Action: Suck")
        env[move] = 'Clean'
    print(f"End: {env}")

# Example
vacuum({'A': 'Dirty', 'B': 'Dirty'}, 'A')
