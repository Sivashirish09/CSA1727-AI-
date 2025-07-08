import math

# Initialize board
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

def minimax(brd, depth, is_maximizing):
    if check_winner(brd, 'O'): return 1
    if check_winner(brd, 'X'): return -1
    if is_draw(brd): return 0

    if is_maximizing:
        best_score = -math.inf
        for i in range(9):
            if brd[i] == " ":
                brd[i] = 'O'
                score = minimax(brd, depth + 1, False)
                brd[i] = " "
                best_score = max(score, best_score)
        return best_score
    else:
        best_score = math.inf
        for i in range(9):
            if brd[i] == " ":
                brd[i] = 'X'
                score = minimax(brd, depth + 1, True)
                brd[i] = " "
                best_score = min(score, best_score)
        return best_score

def best_move():
    best_score = -math.inf
    move = None
    for i in range(9):
        if board[i] == " ":
            board[i] = 'O'
            score = minimax(board, 0, False)
            board[i] = " "
            if score > best_score:
                best_score = score
                move = i
    board[move] = 'O'

def play_game():
    print("Tic Tac Toe (You: X, AI: O)")
    print_board()
    
    while True:
        # Player Move
        try:
            move = int(input("Enter your move (1-9): ")) - 1
            if board[move] != " " or not 0 <= move <= 8:
                print("Invalid move.")
                continue
        except:
            print("Invalid input.")
            continue

        board[move] = 'X'
        print_board()

        if check_winner(board, 'X'):
            print("🎉 You win!")
            break
        if is_draw(board):
            print("It's a draw!")
            break

        # AI Move
        print("AI is making a move...")
        best_move()
        print_board()

        if check_winner(board, 'O'):
            print("💻 AI wins!")
            break
        if is_draw(board):
            print("It's a draw!")
            break

play_game()
