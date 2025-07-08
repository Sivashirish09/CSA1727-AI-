import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np

# FNN Model Definition
class TicTacToeNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(9, 27)
        self.fc2 = nn.Linear(27, 18)
        self.fc3 = nn.Linear(18, 9)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)  # Raw scores for each move

# Encode board: X=-1, O=1, empty=0
def encode_board(board):
    return torch.tensor([1 if c == 'O' else -1 if c == 'X' else 0 for c in board], dtype=torch.float32)

# Create dummy training data (example: center > corner > edge)
def generate_data():
    data = []
    for _ in range(1000):
        board = [' '] * 9
        move = random.choice([4] + [0, 2, 6, 8] + [1, 3, 5, 7])
        if board[move] == ' ':
            board[move] = 'O'
            input_vector = encode_board(board)
            target = torch.tensor(move)
            data.append((input_vector, target))
    return data

# Train model
def train(model, data):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    for epoch in range(10):  # Train for few epochs
        total_loss = 0
        for x, y in data:
            optimizer.zero_grad()
            output = model(x)
            loss = criterion(output.unsqueeze(0), y.unsqueeze(0))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}")

# Make move based on model prediction
def get_model_move(model, board):
    with torch.no_grad():
        input_tensor = encode_board(board)
        output = model(input_tensor)
        for idx in output.argsort(descending=True):  # Try best to worst
            if board[idx] == ' ':
                return idx.item()
    return None

# Play game with neural net AI
def play_game():
    model = TicTacToeNN()
    data = generate_data()
    train(model, data)

    board = [' '] * 9
    print("You are X, Neural Net is O")

    while True:
        # User move
        print_board(board)
        try:
            move = int(input("Enter your move (1-9): ")) - 1
            if board[move] != ' ':
                print("Invalid.")
                continue
        except:
            print("Error.")
            continue
        board[move] = 'X'
        if check_winner(board, 'X'):
            print_board(board)
            print("You win!")
            return

        # NN move
        ai_move = get_model_move(model, board)
        if ai_move is not None:
            board[ai_move] = 'O'
            print_board(board)
            if check_winner(board, 'O'):
                print("Neural Net wins!")
                return
        if ' ' not in board:
            print("It's a draw!")
            return

# Supporting functions
def print_board(board):
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

# Run the game
play_game()
