#SJSU CS156 -Spring 2017
#Connect 4 Project

class Connect4Game:

    def __init__(self):
        """Constructor that initializes an empty board"""
        self.board = [[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0]]

    def run_game(self):
        """This function launches the game. Function to check for winner or a
        full game board still needs to be implemented"""
        
        quit = False
        print "Game begins:"
        while (not quit):
        
        ## Player 1  
            show_board(self.board)
            
            #check if board is full
            if(is_full(self.board)):
                quit = True;
                print "The board is full!"
                break;
            
            #confirm the move is valid. Re-ask user for input until valid entry.
            input_valid = False
            while(not input_valid):
                print "Player 1's turn:"
                col = self.p1_next_move(self.board)
                input_valid=is_move_valid(col,self.board)
                if(not input_valid): print "Invalid move, try again!"
            
            #place the disc in the board
            i=0
            while i<6:
                if self.board[i][col]!=0:
                    break
                i=i+1
            self.board[i-1][col]=1
                
        ## Player 2    
            show_board(self.board)
            
            #check if board is full
            if(is_full(self.board)):
                quit = True;
                print "The board is full!"
                break;
        
            #confirm the move is valid. Re-ask user for input until valid entry.
            input_valid = False
            while(not input_valid):
                print "Player 2's turn:"
                col = self.p2_next_move(self.board)
                input_valid=is_move_valid(col,self.board)
                if(not input_valid): print "Invalid move, try again!"
            
            #place the disc in the board
            i=0
            while i<6:
                if self.board[i][col]!=0:
                    break
                i=i+1                    
            self.board[i-1][col]= 2
                                            
    def p1_next_move(self,currentBoard):
        """This function decides what the next move should be based on currentBoard
        We need to implement the logic for player 1 here. If human player, leave as is.
        @param currentBoard: current board as a list of lists
        @return the next move as an integer representing column number"""
        col = input("Enter the column (0-indexed) to place disc in:")
        return col
    
    def p2_next_move(self,currentBoard):
        """This function decides what the next move should be based on currentBoard
        We need to implement the logic for player 2 (****THE AI *****) here.
        @param currentBoard: current board as a list of lists
        @return the next move as an integer representing column number"""
        col = input("Enter the column (0-indexed) to place disc in:")
        return col
        

def is_full(currentBoard):
    """This function returns true if board is full. False if not full"""
    for i in range(6):
        if(0 in currentBoard[i]):
            return False
    return True
         
def is_move_valid(proposedMove,currentBoard):
    """This function checks whether the proposedMove is valid for currentBoard
    @param proposedMove: an integer indicating column to place the disc in [0,1,..6].
    @param currentBoard: a list of list representing the board.
    @return True if proposedMove is valid. False if proposedMove is invalid."""
    
    if proposedMove<0: return False
    
    #if proposedMove>6: return False
    #NOTE: I left this check out, so we can use it as a means to quit the game while testing
    #by simply entering a number greater than 6. It'll cause error and terminate program.
    #in final submission we'll uncomment the line above.

    i=5
    while i>=0:
        if currentBoard[i][proposedMove]==0:
            return True #returns breaks us out of while loop and terminates.
        i=i-1
    return False #if it reaches this point this column is full.

def show_board(currentBoard):
    """Prints the board to console"""
    for i in range(6):
        print currentBoard[i]
    print "======================"
    print " 0  1  2  3  4  5  6 <--Col Number"
    print


game1 = Connect4Game()
game1.run_game()
