# Player = X (User), AI = O (Decision Tree Logic)

board = [" " for _ in range(9)]

def print_board():
    print()
    for i in range(3):
        print(f"{board[3*i]}|{board[3*i+1]}|{board[3*i+2]}")
        if i < 2:
            print("-+-+-")
    print()

def check_winner(player):
    wins = [(0,1,2), (3,4,5), (6,7,8),
            (0,3,6), (1,4,7), (2,5,8),
            (0,4,8), (2,4,6)]
    return any(board[a] == board[b] == board[c] == player for a, b, c in wins)

def is_draw():
    return " " not in board

# Simulated Decision Tree Rules
def ai_move():
    # Rule 1: Win if possible
    for i in range(9):
        if board[i] == " ":
            board[i] = 'O'
            if check_winner('O'):
                return
            board[i] = " "

    # Rule 2: Block opponent win
    for i in range(9):
        if board[i] == " ":
            board[i] = 'X'
            if check_winner('X'):
                board[i] = 'O'
                return
            board[i] = " "

    # Rule 3: Take center if free
    if board[4] == " ":
        board[4] = 'O'
        return

    # Rule 4: Take a corner if free
    for i in [0, 2, 6, 8]:
        if board[i] == " ":
            board[i] = 'O'
            return

    # Rule 5: Take any side
    for i in [1, 3, 5, 7]:
        if board[i] == " ":
            board[i] = 'O'
            return

def play_game():
    print("Tic Tac Toe (Decision Tree AI)")
    print_board()

    while True:
        try:
            move = int(input("Enter your move (1-9): ")) - 1
            if not (0 <= move <= 8) or board[move] != " ":
                print("Invalid move.")
                continue
        except:
            print("Please enter a valid number.")
            continue

        board[move] = 'X'
        print_board()

        if check_winner('X'):
            print("🎉 You win!")
            break
        if is_draw():
            print("It's a draw!")
            break

        print("AI is thinking...")
        ai_move()
        print_board()

        if check_winner('O'):
            print("💻 AI wins!")
            break
        if is_draw():
            print("It's a draw!")
            break

play_game()
