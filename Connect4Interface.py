#SJSU CS156 -Spring 2017
#Connect 4 Project

class Connect4Game:
    """This class administers the game. Game currently asks users for input
    for each move. To implement AI, redefine the method p2_next_move(). p1_next_move()
    could stay the same, prompting human player to select move."""

    def __init__(self):
        """Constructor that initializes an empty board"""
        self.board = [[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0]]

    def run_game(self):
        """This function launches the game."""
        
        quit = False
        print "Game begins:"
        while (not quit):
        
        ## Player 1  
            show_board(self.board)

            #check if there's a winner
            if(player_won(self.board)!=0):
                print "Game ended."
                break;  
                                        
            #check if board is full
            if(board_full(self.board)):
                print "The board is full!"
                break;
            
            #confirm the move is valid. Re-ask user for input until valid entry.
            input_valid = False
            while(not input_valid):
                print "Player 1's turn:"
                col = self.p1_next_move(self.board)
                if col == 9: # 9 to quit
                    return
                input_valid=is_move_valid(col,self.board)
                if(not input_valid): print "Invalid move, try again!"
            
            #place the disc in the board
            """
            i=0
            while i<6:
                if self.board[i][col]!=0:
                    break
                i=i+1
            self.board[i-1][col]=1
            """
            place_disc(self.board, col, 1)
                            
        ## Player 2    
            show_board(self.board)
            
            #check if there's a winner
            if(player_won(self.board)!=0):
                print "Game ended."
                break;            
            
            #check if board is full
            if(board_full(self.board)):
                print "The board is full!"
                break;
        
            #confirm the move is valid. Re-ask user for input until valid entry.
            input_valid = False
            while(not input_valid):
                print "Player 2's turn:"
                col = self.p2_next_move(self.board)
                if col == 9: # 9 to quit
                    return
                input_valid=is_move_valid(col,self.board)
                if(not input_valid): print "Invalid move, try again!"
            
            #place the disc in the board
            """
            i=0
            while i<6:
                if self.board[i][col]!=0:
                    break
                i=i+1                    
            self.board[i-1][col]= 2
            """
            place_disc(self.board, col, 2)
                                            
    def p1_next_move(self,currentBoard):
        """This function decides what the next move should be based on currentBoard
        We need to implement the logic for player 1 here. If human player, leave as is.
        @param currentBoard: current board as a list of lists
        @return the next move as an integer representing column number"""
        col = input("Enter the column (0-indexed, 9 to quit) to place disc in:")
        return col
    
    def p2_next_move(self,currentBoard):
        """This function decides what the next move should be based on currentBoard
        We need to implement the logic for player 2 (****THE AI *****) here.
        @param currentBoard: current board as a list of lists
        @return the next move as an integer representing column number"""
        col = input("Enter the column (0-indexed, 9 to quit) to place disc in:")
        return col

#below functions are global so they can be used in other classes too.

def place_disc(board, col, player):
    """place the disc in the board"""
    i=0
    while i<6:
        if board[i][col]!=0:
            break
        i=i+1                    
    board[i-1][col]= player

def board_full(currentBoard):
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

def player_won(currentBoard):
    """Checks if there's a winner in the current board
    @param currentBoard a list of list of 0,1,2 representing the board state.
    @returns an integer 1, if player #1 won; 2, if player #2 won; 0, if no one won"""
    
    colWinner = check_columns(currentBoard) #returns 0 if no winner (caching output in variable)
    if(colWinner!=0): 
        print "Player %s won, vertical 4 in a row" % colWinner
        return colWinner

    rowWinner = check_rows(currentBoard) #returns 0 if no winner
    if(rowWinner!=0): 
        print "Player %s won, horizontal 4 in a row" % rowWinner
        return rowWinner

    diagWinner = check_diagonal(currentBoard) #return 0 if no winner 
    if(diagWinner!=0): 
        print "Player %s won, diagonal 4 in a row" % diagWinner
        return diagWinner
    
    #if it reaches this point, then no col, row, or diag winner
    return 0
    
def check_columns(currentBoard):
    """Check columns for vertical 4 in a row
    @return integer 1, if player #1 won; 2, if player #2 won; 0, if no one won"""   
    for c in range(7):
        if (currentBoard[5][c]!=0): #indicating the column is not empty. No point checking empty col
            currNumb = currentBoard[5][c] #store current location disc type "1" or "2"
            consecCount = 1 
            for r in range(4,-1,-1): #r in 4, 3, 2, 1, 0
                if(currentBoard[r][c]==0):
                    break #rest of the column is empty, no point continue checking if no 4-in-a-row yet
                elif(currentBoard[r][c]==currNumb):
                    consecCount+=1
                    if(consecCount==4):
                        #print "Vertical 4 in a row. Player %s has won!" % currNumb
                        return currNumb
                elif(currentBoard[r][c]!=currNumb):
                    consecCount=1 #reset count if a different player's disc encountered.
                    currNumb=currentBoard[r][c]
    return 0
    
def check_rows(currentBoard):
    """Check rows for horizontal 4 in a row
    @return integer 1, if player #1 won; 2, if player #2 won; 0, if no one won"""
    for row in range(6):
        for col in range(4): #col in 0,1,2,3
            consec4s = [currentBoard[row][col],currentBoard[row][col+1],currentBoard[row][col+2],currentBoard[row][col+3]]
            if((0 not in consec4s) and (sum(consec4s)==4)):
                #4 1's in a row should sum to 4 if there are no 0's
                #indicating that there were four (1) in a row
                #print "Horizontal 4 in a row. Player 1 has won!"
                return 1
            elif((0 not in consec4s) and (sum(consec4s)==8)):
                #4 2's in a row should sum to 8 if there are no 0's
                #print "Horizontal 4 in a row. Player 2 has won!"
                return 2
    return 0
    
def check_diagonal(currentBoard):
    """Check diagonals (both ascending and descending) for 4 in a row
    @return integer 1, if player #1 won; 2, if player #2 won; 0, if no one won"""
    #Check diagonals going down to right.
    for row in [0,1,2]:
        for col in range(4):
            if(currentBoard[row][col]!=0): #no point checking consecutive 0's
                consec4disc = True  
                for i in range(1,4):
                    if(currentBoard[row][col]!=currentBoard[row+i][col+i]):
                        consec4disc = False
                        break
                        
                #if after looping through the 3 discs after current and still haven't
                #found one of a different type or 0 (both of which would set consec4disc to False)
                #then we have 4 in a row.
                if(consec4disc!=False):
                    #print "Player %s has won!" % currentBoard[row][col]
                    return currentBoard[row][col] 
    
    #Check diagonals going up to right.
    for row in [3,4,5]:
        for col in range(4):
            if(currentBoard[row][col]!=0): #no point checking consecutive 0's
                consec4disc = True
                for i in range(1,4):
                    if(currentBoard[row][col]!=currentBoard[row-i][col+i]):
                        consec4disc = False
                        break
    
                if(consec4disc!=False):
                    #print "Player %s has won!" % currentBoard[row][col]
                    return currentBoard[row][col]
    
    #if by this point no diagonal 4 in a row found, then there are none.            
    return 0
            
#game1 = Connect4Game()
#game1.run_game()
