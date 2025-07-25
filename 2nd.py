N = 8

def print_board(board):
    for row in board:
        print(" ".join("Q" if c else "." for c in row))
    print()

def is_safe(board, row, col):
    for i in range(row):
        if board[i][col] or \
           (col >= row - i and board[i][col - (row - i)]) or \
           (col + row - i < N and board[i][col + (row - i)]):
            return False
    return True

def solve(board, row):
    if row == N:
        print_board(board)
        return True  # Change to False to find all solutions
    for col in range(N):
        if is_safe(board, row, col):
            board[row][col] = 1
            if solve(board, row + 1):
                return True
            board[row][col] = 0
    return False

board = [[0]*N for _ in range(N)]
solve(board, 0)
