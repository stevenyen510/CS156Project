import Connect4Interface
#import random
import copy


class GameWithAI(Connect4Interface.Connect4Game):
    """This class inherits the Connect4Game class in Connect4Interface.py
    it simply overrides the p2_next_move()"""
                 
    def p2_next_move(self,currentBoard):
        """This overrides the p2_next_move() function in the original class
        to impelement an AI. The "AI" here simply picks a random move between
        column 0 and 6"""
        print "Move by Minimax AI"
        print
        #column = random.randint(0,6)
        #return column
        """depth only works with 2 for now"""
        depth = 2
        player = 2
        minimax_value = minimax(self.board, player, depth)
        board_dict = add_value(self.board, player)
        return next(index for (index, d) in enumerate(board_dict) if d['v'] == minimax_value)
        
def add_board(board, player):
    """add leaf boards"""
    copy_boards = []
    for x in range(7):
        copy_boards.append(copy.deepcopy(board))
        if(Connect4Interface.is_move_valid(x, copy_boards[x])):
            Connect4Interface.place_disc(copy_boards[x], x, player)
    return copy_boards

def add_value(board, player):
    """create list of dictionary for leaf boards"""
    bvs = []
    boards = add_board(board, player)
    for b in boards:
        bv = {}
        bv['b'] = b
        bv['v'] = eval(b, player)
        bvs.append(bv)
    return bvs
    
def minimax(board, player, depth):
    """return minimax value"""
    return maxV(board, player, depth - 1)
    
def maxV(board, player, depth):
    """return max value from leaf boards"""
    if(depth == 0 or Connect4Interface.board_full(board)):
        return eval(board, player)
    v = -99999999
    bs = add_board(board, player)
    for b in bs:
        v = max(v, minV(b, player, depth - 1))
    return v

def minV(board, player, depth):
    """return min value from leaf boards"""
    if(depth == 0 or Connect4Interface.board_full(board)):
        return eval(board, player)
    v = 99999999
    bs = add_board(board, player)
    for b in bs:
        v = min(v, minV(b, player, depth - 1))
    return v
    
def eval(board, player):
    """evaluation the current board for given player"""
    four_factor = 1000
    three_factor = 100
    two_factor = 10
    one_factor = 1
    four_streak = get_streak(board, 4, player)
    three_streak = get_streak(board, 3, player)
    two_streak = get_streak(board, 2, player)
    one_streak = get_streak(board, 1, player)
    """evaluation function"""
    """one_streak / 4 for clear repeat count"""
    value = four_factor * four_streak + three_factor * three_streak + two_factor * two_streak + one_factor * one_streak / 4
    return value
    
def get_streak(board, number, player):
    """get total streak"""
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
AI = GameWithAI()
AI.run_game()
        
        
        