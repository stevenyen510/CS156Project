#CS 156 Spring 2017
#Connect 4 Project 
#05-14-17

import Connect4Interface
#import random
import copy

from sklearn import tree #need to install Scikit Learn to import this module

##Training data used are from http://archive.ics.uci.edu/ml/datasets/Connect-4
##See the website and our project report/presentation for details of data transformation.

############## Loading training sample from file, added 051417 ##############
def load_data_array(fileName, count):
    """This function loads data from a file represented in the uci format
    @param fileName: String = representing the full name of file with extension
    @param count: int = number of data points to be loaded
    @return returnArray: array of strings."""
    
    returnArray =[]

    f = open(fileName,'r')
    for j in range(count):
        t_line=f.readline()
        returnArray.append(t_line)
        
    f.close() #close the file to free up system resources.
    
    return returnArray
    
#############################################################################
#Changed the following to a setupt where human goes first. 050817 1653
#flipped the signs. Now loss=1, win=-1 (because in our game, human starts first)
#AI "win" necessarily mean human "loss"
def seperateVariabes(dataStrArr):
    """Takes data as a string of arrays and return two arrays in form for DTree
    @param dataStrArr: array of strings econding board state with {x,b,o} plus cl label
    @return tuple (xx,yy) where xx is arrary of arrays representing board state as -1,+1,0
    yy is an arrary of -1,+1, 0 representing {loss, win, draw}
    """

    xx=[]
    yy=[]
    
    for each_line in dataStrArr:
        
        line_as_arr = each_line.split(',')
        board_rep = line_as_arr[:42]  #get's string of board rep
        
        board_rep_int = [None]*42
        for i in range(42):
            if board_rep[i]=='x':
                board_rep_int[i] = -1 #first player, the human
            elif board_rep[i]=='o':
                board_rep_int[i] = 1 #second player, the AI.
            elif board_rep[i]=='b':
                board_rep_int[i] = 0
        
        class_label = line_as_arr[42] #get class label "draw","loss", "win "
        
        class_label_int =0
        if class_label=="win " or class_label=="win" or class_label =="win\n":
            class_label_int =-1
        elif class_label =='loss':
            class_label_int = 1  #signs inverted since human loss = AI win 
        elif class_label =='draw':
            class_label_int = 0
        
        xx.append(board_rep_int)
        yy.append(class_label_int)
        
    return (xx,yy)

####################Instantiating and Traiing the DTree########################
training_data = load_data_array("training_data.txt",500) #note this data from UCI
#only 1st 10000 points were included to reduce size and allow upload to github
X,Y = seperateVariabes(training_data)
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X,Y)
###############################################################################

def OurBoard2TreeInput_TF(currentBoard):
    """This function converts our board representation into the same representation
    used to train the Decision Tree Classifier Model
    @param currentBoard the current board state where 1 is human player, 2 is AI, 0 is blank
    @return a list of length 42 {-1,0,+1}, corresponding to the 42 slots of the board"""
    
    TreeInputArr=[None]*42
    k=0
    for col in range(7):
        for row in [5,4,3,2,1,0]:
            
            if(currentBoard[row][col]==2):
                slot_val = 1   #since ai is o
            elif(currentBoard[row][col]==1):
                slot_val =-1
            elif(currentBoard[row][col]==0):
                slot_val =0
            
            
            TreeInputArr[k]= slot_val
            k+=1
    
    return TreeInputArr

def OurBoard2TrainData_TF(currentBoard,cl):
    """Converts a board represented by our convention into a string of characters
    in the same format used by the online training data base
    @param currentBoard: the board as a list of lists
    @param cl: the class label to be assigned to that state as a string
    @return a line of string in the format used by online database.
    """
    outputStr =""
    for col in range(7):
        for row in [5,4,3,2,1,0]:
            if(currentBoard[row][col]==2):
                outputStr+='o,'   #since ai is o
            elif(currentBoard[row][col]==1):
                outputStr+='x,'
            elif(currentBoard[row][col]==0):
                outputStr+='b,'
    outputStr+=cl
    return outputStr
    
def TrainData2OurRep_TF(oxbString):
    """Takes a board encoded in o,x,b and creates a list of list in our rep"""
    oxbArr = oxbString.split(',')
    indx=0
    
    boardAsList = [[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0],
                    [0,0,0,0,0,0,0],[0,0,0,0,0,0,0],[0,0,0,0,0,0,0]]
    
    for col in range(7):
        for row in [5,4,3,2,1,0]:
            if oxbArr[indx]=='b':    
                boardAsList[row][col]=0
            elif oxbArr[indx]=='x':
                boardAsList[row][col]=1
            elif oxbArr[indx]=='o':
                boardAsList[row][col]=2
            indx+=1
            
    return boardAsList
                               
###############################################################################                                
#################stuff above for decision tree.                                        
###############################################################################                                                                                                                                                             
                                                                                                                                
class GameWithDTreeAI(Connect4Interface.Connect4Game):
    """Derived class of the Connect4Game class on Connect4Interface.py module
    simply overrides the single function p2_next_move() so it uses the Minimax
    algorithm to pick the best move"""
                                                            
    def p2_next_move(self,currentBoard):
        """Overrides this corresponding function in the parent class
        now this method takes an a board state and initiates a Minimax algorithm
        to help determine the move (a col number) that maximizes p2's utility value"""
        
        player = 2
        depth = 4
        
        next_boards_utility={}
        
        for move in range(7):
            if Connect4Interface.is_move_valid(move, currentBoard):
                copy_of_board = copy.deepcopy(currentBoard) #creates copy of currentBoard
                Connect4Interface.place_disc(copy_of_board,move,player)
                next_boards_utility[move] = min_val(copy_of_board,1,depth-1)  #change this to opponent as player.
        
                
        #now next_boards_utilty is a dictionary {0: min_val, 1: min_va,.....}        
        util_max = -100000
        move_max = None #the movethat maximizes the utility value
        moves_and_util = next_boards_utility.items() #items() returns a list of dict's (key, value) tuple pairs
        
        for move, util_val in moves_and_util:   
            if util_val >= util_max:
                util_max = util_val
                move_max = move    
        
        print "AI picked:", move_max
        print "from:", next_boards_utility
        return move_max
           

def min_val(boardX,player,depth):
    """Implementation of the MIN-VALUE(state) function, AIMA 3rd Ed, Fig 5.3"""
    
    if(player==1):
        opponent = 2
    else:
        opponent =1
        
    next_boards = []  #a list of boards
    
    for move in range(7):
        if Connect4Interface.is_move_valid(move, boardX):
            new_board = copy.deepcopy(boardX) #creates copy of currentBoard
            Connect4Interface.place_disc(new_board,move,player)
            next_boards.append(new_board)

    if depth ==0 or len(next_boards)==0 or Connect4Interface.player_won(boardX):
        return heuristic_function(boardX,player,depth-1)
            
    #now find the board with lowest beta value
    beta = 10000 
    for board_i in next_boards:
        beta = min(beta, max_val(board_i,opponent, depth-1))
        
    return beta
    
def max_val(boardX,player,depth):
    """Implementation of the MAX_VALUE(state) function, AIMA 3rd Ed, Fig 5.3"""
    
    if(player==1):
        opponent = 2
    else:
        opponent =1
        
    next_boards =[] #a list of boards
    
    for move in range(7):
        if Connect4Interface.is_move_valid(move, boardX):
            new_board = copy.deepcopy(boardX) #creates copy of currentBoard
            Connect4Interface.place_disc(new_board,move,player)
            next_boards.append(new_board)

    ###if TERMINA-TEST(state) then return Utility(state)
    if depth ==0 or len(next_boards)==0 or Connect4Interface.player_won(boardX):
        return heuristic_function(boardX, player,depth-1)
                                                                            
    #now find the board with max alpha value
    
    ###v <-- -infinity
    alpha = -10000
    
    ###for each a in ACTIONS(state) do
    for board_i in next_boards:
        ###v<--MAX(v,MIN-VALUE(board_i)
        alpha = max(alpha, min_val(board_i,opponent, depth-1))
    
    return alpha
        
def heuristic_function(boardX,player,depth):
    """New heuristic_function(), simply call uses the Decision Tree Classifier
    Model trained in the begining of this code to predict the theoretical outcome
    of the board win=+1, loss = -1, draw = 0."""

    if Connect4Interface.player_won(boardX)==1:
        return -10000 - depth #essentially assigning it to negative infinity.
    elif Connect4Interface.player_won(boardX)==2:
        return +10000 #essentially assining it to positive infinity. 051417
    else:
        
        #Using Linear Heuristic Based on Threats
        return threat_heuristic(boardX,player)
        
        #Using DTree Heuristic
        #dtree_output = clf.predict([OurBoard2TreeInput_TF(boardX)])
        #return dtree_output[0]


###############################################################################    
#######################Added Threat Heuristic 051517 ##########################           
print "Threat Heuristic Functions Added"
    
def is_a_threat(boardX,row,col,player):
    """Check if the cell at current row,col is a threat favoring the player
    (e.g., if a row has [0,1,1,1,2,1,2] then first cell is a 'threat' for 1)
    See comments in code body for which cells are being checked 
    @param boardX, the board being evaluated
    @param row, the current cell's row
    @param col, the current cell's col
    @param player {1,2} the player we're evaluating
    @return True if current cell is a threat favoring the player --051517"""
    
    if player==2:
        opp = 1
    elif player ==1:
        opp =2
    
    if boardX[row][col]==0:
        
        for offset in [-3,-2,-1,0]:
            #check 4 to the left from start
            start_col = col+offset
            horiz4s = []
            if start_col in [0,1,2,3]:
                horiz4s = []
                horiz4s.append(boardX[row][start_col])
                horiz4s.append(boardX[row][start_col+1])
                horiz4s.append(boardX[row][start_col+2])
                horiz4s.append(boardX[row][start_col+3])
            else:
                continue
            
            if (sum(horiz4s)==(3*player)) and (opp not in horiz4s):
                return True
            
        for offset in [3,2,1,0]:
            #check 4 above the start row
            start_row = row+offset
            
            vert4s=[]
            if start_row in [5,4,3]:
                vert4s = []
                vert4s.append(boardX[start_row][col]) 
                vert4s.append(boardX[start_row-1][col])
                vert4s.append(boardX[start_row-2][col])
                vert4s.append(boardX[start_row-3][col])
            else:
                continue
            
            if (sum(vert4s)==(3*player)) and (opp not in vert4s):
                return True
                
        for offset in [3,2,1,0]:
            #check diag ascend going up to right
            start_col = col+offset
            start_row = row-offset
            diag4s = []
            if (start_col in [0,1,2,3]) and (start_row in [5,4,3]):
                diag4s = []
                diag4s.append(boardX[start_row][start_col])
                diag4s.append(boardX[start_row-1][start_col+1])
                diag4s.append(boardX[start_row-2][start_col+2])
                diag4s.append(boardX[start_row-3][start_col+3])
            else:
                continue

            if (sum(diag4s)==(3*player)) and (opp not in diag4s):
                return True
                
        for offset in [3,2,1,0]:
            #check diag desc going down to right
            start_col = col-offset
            start_row = row-offset
            diag4s = []
            if (start_col in [0,1,2,3]) and (start_row in [0,1,2]):
                diag4s = []
                diag4s.append(boardX[start_row][start_col])
                diag4s.append(boardX[start_row+1][start_col+1])
                diag4s.append(boardX[start_row+2][start_col+2])
                diag4s.append(boardX[start_row+3][start_col+3])                
            else:
                continue
                
            if (sum(diag4s)==(3*player)) and (opp not in diag4s):
                return True
            
    return False
                    

def count_even_threats(boardX,player):
    """Return the number of even threats established by player --051517"""
    
    even_threats =0
    
    for row in [0,2,4]:
        for col in range(7):
            if(is_a_threat(boardX,row,col,player)):
                even_threats+=1
    return even_threats
            
def count_odd_threats(boardX,player):
    """Return the number of odd threats established by player --051517"""
    
    odd_threats =0
    
    for row in [1,3,5]:
        for col in range(7):
            if(is_a_threat(boardX,row,col,player)):
                odd_threats+=1
                                                                                                      
    return odd_threats                                                                    
                                                                                                                                                                                                                                                                                                                                                                                                                                              
def threat_array(boardX,player):
    """Return an array of threat counts: [my_even_threats, my_odd_threats,
    opp_even_threats, opp_odd_threats]"""
    
    if (player==2):
        opp =1
    elif (player==1):
        opp =2
    
    #my threats
    my_even_threats = count_even_threats(boardX,player)
    my_odd_threats = count_odd_threats(boardX,player)
    #opp threats
    op_even_threats = count_even_threats(boardX,opp)
    op_odd_threats = count_odd_threats(boardX,opp)
    
    return [my_even_threats, my_odd_threats, op_even_threats, op_odd_threats]
                                                                                                                                                                                        
my_even_threat_coef=1
my_odd_threat_coef =1
op_even_threat_coef=-1
op_odd_threat_coef=-1

def threat_heuristic(boardX,player):
    """Linear combination function based on features"""
    threatArr = threat_array(boardX,player)
    
    board_val = my_even_threat_coef*threatArr[0]
    board_val+=my_odd_threat_coef*threatArr[1]
    board_val+=my_even_threat_coef*threatArr[2]
    board_val+=my_odd_threat_coef*threatArr[3]
    
    return  board_val
                                
###############################################################################    
#######################Added Threat Heuristic above  ##########################

                                                                                                                                                                                
print "DTree Module Loaded"
game3 = GameWithDTreeAI()
game3.run_game()

keep_playing = True
while(keep_playing):
    user_input = raw_input("Would you like to player another game? (y/n): ")
    if(user_input =='n'):
        break
    game3.run_game()




    