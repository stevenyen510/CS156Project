#SJSU CS156 -Spring 2017
#Connect 4 Project

#testing git 040317
#testing git edit from browser 040317 1834
#testing git push from desktop 040317 1842

#initialize the game board
board=[
        [0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0],
        [0,0,0,0,0,0,0]
      ]

#this function launches the game
#the game continues until terminated by user (enter number >6) 
#Or until the game board is full.
#Function to check for full game board still need to be implemented.
def run_game():
     
    quit = False
    print "Game begins:"
    while (not quit):
      
    ## Player 1  
        show_board(board)
        
        print "Player 1's turn:"
        col = next_move(board)
        
        if col>6:
            print "invalid column number selected, quit game"
            break

        i=0
        while i<6:
            if board[i][col]!=0:
                break
            i=i+1
                
        board[i-1][col] =1
    
    ## Player 2    
        show_board(board)
    
        print "Player 2's turn:"
        col = next_move(board)
        
        if col>6:
            print "invalid column number selected, quit game"
            break
        
        i=0
        while i<6:
            if board[i][col]!=0:
                break
            i=i+1
                
        board[i-1][col]= 2
         
           
#This move takes in the current board as a list of lists
#the logic to decide next move is to be included in this function
#here, the logic is simply to ask for user input.
def next_move(currentBoard):
    col = input("Enter the column (0-indexed) to place disc in:")
    return col


#Prints the list of list with each row seperated by a line break
def show_board(currentBoard):
    for i in range(6):
        print currentBoard[i]
    print "======================"
    print " 0  1  2  3  4  5  6 <--Col Number"
    print



run_game()
