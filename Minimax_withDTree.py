#CS 156 Spring 2017
#Connect 4 Project 
#05-16-17

import Connect4Interface
import random
import copy
import time

from sklearn import tree #need to install Scikit Learn to import this module

#####added 051617 to save board states encountered during game play############
#filename1= "gameplay_"+time.strftime("%m%d%Y_%H%M")+".txt"
#f3Glob = open(filename1,'w')

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
        elif class_label =='loss' or class_label=="loss\n":
            class_label_int = 1  #signs inverted since human loss = AI win 
        elif class_label =='draw' or class_label=="draw\n":
            class_label_int = 0
        
        xx.append(board_rep_int)
        yy.append(class_label_int)
        
    return (xx,yy)

def trainWithFeatures(td_StrArr):
    xx =[]
    yy =[]
    
    for each_line in td_StrArr:
        board_obx = each_line[0:83]
        indpVar_entry = []
        indpVar_entry.extend(wr_thrts(all_winning_combinations(TrainData2OurRep_TF(board_obx))))
        #indpVar_entry.extend(threat_array(TrainData2OurRep_TF(board_obx),2))
        #indpVar_entry is an array consisting the value of 6 features -051617
        
        xx.append(indpVar_entry)
        
        cli = each_line[84:]
        if(cli=="win") or (cli=="win\n") or (cli=="win "):
            yy.append(-1)
        elif (cli=="loss") or (cli=="loss\n"):
            yy.append(+1)
        elif (cli=="draw") or (cli=="draw\n"):
            yy.append(0)
            
    return (xx,yy)
        


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

def OurBoard2TrainData_TF(currentBoard):
    """Converts a board represented by our convention into a string of characters
    in the same format used by the online training data base
    @param currentBoard: the board as a list of lists
    @return a line of string in the format used by online database (w/o cl)
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
    
    return outputStr[:-1]

def TrainData2OurRep_TF(oxbString):
    """Takes a board encoded in o,x,b and creates a list of list in our rep
    @param oxbString: the state of the board encoded as a string of o,x,b no CL
    """
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
        depth = 5
        
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

print "DTree Model Trained"        
def heuristic_function(boardX,player,depth):
    """New heuristic_function(), simply call uses the Decision Tree Classifier
    Model trained in the begining of this code to predict the theoretical outcome
    of the board win=+1, loss = -1, draw = 0."""

        
    
    if Connect4Interface.player_won(boardX)==1:
        return -10000 - depth #essentially assigning it to negative infinity.
    elif Connect4Interface.player_won(boardX)==2:
        return +10000 #essentially assining it to positive infinity. 051417
    else:
        allcomb = all_winning_combinations(boardX)
        features = wr_thrts(allcomb)
        #features.extend(threat_array(boardX,2))
        #this creates an array of 6 features
        
        util_val = clf2.predict([features])[0]
        
        #f3Glob.write(OurBoard2TrainData_TF(boardX)+","+str(util_val)+","+str(features)+"\n")
        
        return util_val
        
        #tally = all_winning_comb_tally(allcomb)
        #return sum(tally)        
        
        #dtree_output = clf.predict([OurBoard2TreeInput_TF(boardX)])
        #return dtree_output[0]
        
        #based on thrats
        #return threat_heuristic(boardX,2)       
                             

###############################################################################
##############################ADDED THIS 051517 12:25 #########################
def all_winning_combinations(boardX):
    """Check all 69 possible 4-in-row's in the board added 051517
    @player {1,2}, the player that's evaluating the board
    @return and array of {+1,-1,0} if claimed by {player,opp, empty}"""
    
    all_comb =[]  #list of list. 69 lists each of size 4
    
    #enumerate winning rows
    for row in [5,4,3,2,1,0]:
        for col in [0,1,2,3]:    
            consec4s =[]
            consec4s.append(boardX[row][col])
            consec4s.append(boardX[row][col+1])
            consec4s.append(boardX[row][col+2])
            consec4s.append(boardX[row][col+3])
            all_comb.append(consec4s)
            
    #enumerate winning cols
    for col in range(7):
        for row in [5,4,3]:
            consec4s =[]
            consec4s.append(boardX[row][col])
            consec4s.append(boardX[row-1][col])
            consec4s.append(boardX[row-2][col])
            consec4s.append(boardX[row-3][col])
            all_comb.append(consec4s)
            
    #enumerate all ascending winning diags
    for row in [5,4,3]:
        for col in [0,1,2,3]:
            consec4s =[]
            consec4s.append(boardX[row][col])
            consec4s.append(boardX[row-1][col+1])
            consec4s.append(boardX[row-2][col+2])
            consec4s.append(boardX[row-3][col+3])
            all_comb.append(consec4s)
            
    #enumerate all descending wining diags
    for row in [2,1,0]:
        for col in [0,1,2,3]:
            consec4s =[]
            consec4s.append(boardX[row][col])
            consec4s.append(boardX[row+1][col+1])
            consec4s.append(boardX[row+2][col+2])
            consec4s.append(boardX[row+3][col+3])
            all_comb.append(consec4s)
            
    return all_comb

def all_winning_comb_tally(all_comb):
    """added 051517"""
    tally_arr =[None]*69
    for i in range(69):
        consec4 = all_comb[i]
        if (2 in consec4) and (1 not in consec4):
            tally_arr[i] = +1
        elif (1 in consec4) and (2 not in consec4):
            tally_arr[i] = -1
        else:
            tally_arr[i] = 0
    return tally_arr
    
def winning_rows(all_comb):
    p1_winning_rows = 0
    p2_winning_rows = 0
    for i in range(69):
        consec4 = all_comb[i]
        if (2 in consec4) and (1 not in consec4):
            p2_winning_rows+=1
        elif (1 in consec4) and (2 not in consec4):
            p1_winning_rows+=1
    return [p1_winning_rows,p2_winning_rows]

    
def wr_thrts(all_comb):
    """Returns winning rows and threats as array of integers
    Output array used as indp var for DTree for training and predicting
    The length of the array can be changed to add features as needed -051717"""
    p1_winning_rows = 0
    p2_winning_rows = 0
    p1_threats =0
    p2_threats =0
    
    for i in range(69):
        consec4 = all_comb[i]
        if (2 in consec4) and (1 not in consec4):
            if(sum(consec4)==6):
                p2_threats+=1
            else:
                p2_winning_rows+=1
        elif (1 in consec4) and (2 not in consec4):
            if(sum(consec4)==3):
                p1_threats+=1
            else:
                p1_winning_rows+=1
                
    return [p1_winning_rows,p2_winning_rows,p1_threats,p2_threats]    
                            

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
                                                                                                                                                                                                                                                                                                                                                
####################Instantiating and Traiing the DTree########################

##Training data used are from http://archive.ics.uci.edu/ml/datasets/Connect-4
##See the website and our project report/presentation for details of data transformation.
##training_data.txt contains data from this database.

training_data = load_data_array("training_data.txt",500) #note this data from UCI
#random.shuffle(training_data)
#only 1st 10000 points were included to reduce size and allow upload to github
#X,Y = seperateVariabes(training_data)
#clf = tree.DecisionTreeClassifier()
#clf = clf.fit(X,Y)

X2,Y2 = trainWithFeatures(training_data)
clf2 = tree.DecisionTreeClassifier()
clf2 = clf2.fit(X2,Y2)


###############################################################################                
                                
#########################Model Verification####################################

verificationData = load_data_array("verificationData.txt",200)

vx1, vy1 = trainWithFeatures(verificationData)

vdatasize=len(verificationData)

misses =0 

for i in range(vdatasize):
    
    #print verificationData[i]
    #print "   ", clf2.predict([vx1[i]])[0], vy1[i]
    
    
    if(clf2.predict([vx1[i]])[0]!=vy1[i]): 
        misses=misses+1

        
correctPredictions = vdatasize - misses
accuracy = float(correctPredictions)/vdatasize

print "training sample size:", len(training_data)        
print "correct:", correctPredictions
print "out of:", vdatasize
print "accuracy:", accuracy

###################################Play Game###################################

print "DTree Module Loaded"
game3 = GameWithDTreeAI()
game3.run_game()

keep_playing = True
while(keep_playing):
    user_input = raw_input("Would you like to player another game? (y/n): ")
    if(user_input =='n'):
        break
    game3.run_game()

####close file 051617
#f3Glob.close()    