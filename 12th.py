# Tic Tac Toe Game - 2 Player Version

def print_board(board):
    print("\n")
    print(f"{board[0]} | {board[1]} | {board[2]}")
    print("--+---+--")
    print(f"{board[3]} | {board[4]} | {board[5]}")
    print("--+---+--")
    print(f"{board[6]} | {board[7]} | {board[8]}")
    print("\n")

def check_winner(board, player):
    win_cond = [
        [0,1,2], [3,4,5], [6,7,8],  # Rows
        [0,3,6], [1,4,7], [2,5,8],  # Columns
        [0,4,8], [2,4,6]            # Diagonals
    ]
    for cond in win_cond:
        if board[cond[0]] == board[cond[1]] == board[cond[2]] == player:
            return True
    return False

def is_draw(board):
    return all(cell != " " for cell in board)

def play_game():
    board = [" "] * 9
    current_player = "X"

    print("Welcome to Tic Tac Toe!")
    print_board(board)

    while True:
        try:
            move = int(input(f"Player {current_player}, choose a position (1-9): ")) - 1
            if board[move] != " " or not 0 <= move <= 8:
                print("Invalid move. Try again.")
                continue
        except (ValueError, IndexError):
            print("Please enter a number from 1 to 9.")
            continue

        board[move] = current_player
        print_board(board)

        if check_winner(board, current_player):
            print(f"🎉 Player {current_player} wins!")
            break
        elif is_draw(board):
            print("It's a draw!")
            break

        current_player = "O" if current_player == "X" else "X"

if __name__ == "__main__":
    play_game()
