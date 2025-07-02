from itertools import permutations

def solve_cryptarithmetic():
    letters = 'SENDMORY'
    digits = '0123456789'

    for perm in permutations(digits, len(letters)):
        mapping = dict(zip(letters, perm))

        if mapping['S'] == '0' or mapping['M'] == '0':
            continue  # Leading digits cannot be 0

        send = int(''.join(mapping[c] for c in 'SEND'))
        more = int(''.join(mapping[c] for c in 'MORE'))
        money = int(''.join(mapping[c] for c in 'MONEY'))

        if send + more == money:
            print("Solution Found:")
            for k, v in mapping.items():
                print(f"{k} = {v}")
            print(f"\n{send} + {more} = {money}")
            return

    print("No solution found.")

# Run the solver
solve_cryptarithmetic()
