import Connect4Interface
import random
import copy

#A=1+2

#print('test')




class GameWithAI(Connect4Interface.Connect4Game):
    """This class inherits the Connect4Game class in Connect4Interface.py
    it simply overrides the p2_next_move()"""
                 
    def p2_next_move(self,currentBoard):
        """This overrides the p2_next_move() function in the original class
        to impelement an AI. The "AI" here simply picks a random move between
        column 0 and 6"""
        print "Move by random AI"
        print
        column = random.randint(0,6)
        return column
        
    def add_board(self):
        boards = []
        for x in range(7):
            boards.append(copy.deepcopy(self.board))
            if(Connect4Interface.is_move_valid(x, boards[x])):
                Connect4Interface.place_disc(boards[x], x, 2)
        return boards
    
    def add_value(self, player):
        bvs = []
        boards = self.add_board()
        for b in boards:
           bv = {}
           bv['b'] = b
           bv['v'] = eval(b, player)
           bvs.append(bv)
        return bvs
    
def eval(board, player):
    four_streak = get_streak(board, 4, player)
    three_streak = get_streak(board, 3, player)
    two_streak = get_streak(board, 2, player)
    one_streak = get_streak(board, 1, player)
    """evaluation function"""
    """one_streak / 4 for clear repeat count"""
    value = 5000 * four_streak + 500 * three_streak + 50 * two_streak + one_streak / 4
    return value
    
def get_streak(board, number, player):
    return get_col_streak(board, number, player) + get_row_streak(board, number, player) + get_dia_streak(board, number, player)

def get_col_streak(board, number, player):
    """return number of column streak given number of steak and player"""
    count = 0
    max_row = 6 - number
    for c in range(7):
        for r in range(max_row + 1):
            if number == 1:
                if board[r][c] == player:
                    count += 1
            elif number == 2:
                if board[r][c] == player and board[r+1][c] == player:
                    count += 1
            elif number == 3:
                if board[r][c] == player and board[r+1][c] == player and board[r+2][c] == player:
                    count += 1
            elif number == 4:
                if board[r][c] == player and board[r+1][c] == player and board[r+2][c] == player and board[r+3][c] == player:
                    count += 1
    return count
    
def get_row_streak(board, number, player):
    """return number of row streak given number of steak and player"""
    count = 0
    max_col = 7 - number
    for r in range(6):
        for c in range(max_col + 1):
            if number == 1:
                if board[r][c] == player:
                    count += 1
            elif number == 2:
                if board[r][c] == player and board[r][c+1] == player:
                    count += 1
            elif number == 3:
                if board[r][c] == player and board[r][c+1] == player and board[r][c+2] == player:
                    count += 1
            elif number == 4:
                if board[r][c] == player and board[r][c+1] == player and board[r][c+2] == player and board[r][c+3] == player:
                    count += 1
    return count

def get_dia_streak(board, number, player):
    """return number of diagonal streak given number of steak and player"""
    count = 0
    max_row = 6 - number
    max_col = 7 - number
    """check top left to bottom right (135 degree) diagonal"""
    for r in range(max_row + 1):
        for c in range(max_col + 1):
            if number == 1:
                if board[r][c] == player:
                    count += 1
            elif number == 2:
                if board[r][c] == player and board[r+1][c+1] == player:
                    count += 1
            elif number == 3:
                if board[r][c] == player and board[r+1][c+1] == player and board[r+2][c+2] == player:
                    count += 1
            elif number == 4:
                if board[r][c] == player and board[r+1][c+1] == player and board[r+2][c+2] == player and board[r+3][c+3] == player:
                    count += 1
    """check top right to bottom left (45 degree) diagonal"""
    for r in range(max_row + 1):
        for c in range(number - 1, 7):
            if number == 1:
                if board[r][c] == player:
                    count += 1
            elif number == 2:
                if board[r][c] == player and board[r+1][c-1] == player:
                    count += 1
            elif number == 3:
                if board[r][c] == player and board[r+1][c-1] == player and board[r+2][c-2] == player:
                    count += 1
            elif number == 4:
                if board[r][c] == player and board[r+1][c-1] == player and board[r+2][c-2] == player and board[r+3][c-3] == player:
                    count += 1
    return count
    
    
#print "New Game with AI"
gameWithAI_1 = GameWithAI()
gameWithAI_1.run_game()
        
        
        