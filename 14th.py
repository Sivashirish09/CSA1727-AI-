import math

board = [" " for _ in range(9)]

def print_board():
    print()
    for i in range(3):
        print(f"{board[3*i]}|{board[3*i+1]}|{board[3*i+2]}")
        if i < 2:
            print("-+-+-")
    print()

def check_winner(brd, player):
    wins = [(0,1,2),(3,4,5),(6,7,8),
            (0,3,6),(1,4,7),(2,5,8),
            (0,4,8),(2,4,6)]
    return any(brd[a] == brd[b] == brd[c] == player for a,b,c in wins)

def is_draw(brd):
    return " " not in brd

def alpha_beta(brd, depth, alpha, beta, is_max):
    if check_winner(brd, 'O'): return 1
    if check_winner(brd, 'X'): return -1
    if is_draw(brd): return 0

    if is_max:
        max_eval = -math.inf
        for i in range(9):
            if brd[i] == " ":
                brd[i] = 'O'
                eval = alpha_beta(brd, depth+1, alpha, beta, False)
                brd[i] = " "
                max_eval = max(max_eval, eval)
                alpha = max(alpha, eval)
                if beta <= alpha:
                    break  # Beta cut-off
        return max_eval
    else:
        min_eval = math.inf
        for i in range(9):
            if brd[i] == " ":
                brd[i] = 'X'
                eval = alpha_beta(brd, depth+1, alpha, beta, True)
                brd[i] = " "
                min_eval = min(min_eval, eval)
                beta = min(beta, eval)
                if beta <= alpha:
                    break  # Alpha cut-off
        return min_eval

def best_move():
    best_val = -math.inf
    move = None
    for i in range(9):
        if board[i] == " ":
            board[i] = 'O'
            move_val = alpha_beta(board, 0, -math.inf, math.inf, False)
            board[i] = " "
            if move_val > best_val:
                best_val = move_val
                move = i
    board[move] = 'O'

def play_game():
    print("Tic Tac Toe with Alpha-Beta AI (You: X, AI: O)")
    print_board()

    while True:
        # Player move
        try:
            pos = int(input("Enter your move (1-9): ")) - 1
            if not 0 <= pos <= 8 or board[pos] != " ":
                print("Invalid move.")
                continue
        except:
            print("Enter a number from 1 to 9.")
            continue

        board[pos] = 'X'
        print_board()

        if check_winner(board, 'X'):
            print("🎉 You win!")
            break
        if is_draw(board):
            print("It's a draw!")
            break

        print("AI is thinking...")
        best_move()
        print_board()

        if check_winner(board, 'O'):
            print("💻 AI wins!")
            break
        if is_draw(board):
            print("It's a draw!")
            break

play_game()
