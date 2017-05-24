#CS 156 Spring 2017
#Connect 4 Project 

import Connect4Interface
import random
import copy
from sklearn import tree #need to install Scikit Learn to import this module

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
    """Reduce the training data into independent variables containing features"""
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
################functions above for decision tree #############################                                        
###############################################################################                                                                                                                                                             
                                                                                                                                
class Connect4_AI(Connect4Interface.Connect4Game):
    """Derived class of the Connect4Game class on Connect4Interface.py module
    simply overrides the single function p2_next_move() so it uses the Minimax
    algorithm to pick the best move"""
    def set(self, gp, gt, p1, p2):
        self.games_played = gp
        self.games_tied = gt
        self.p1_wins = p1
        self.p2_wins = p2
                                                            
    def p2_next_move(self,currentBoard):
        """Overrides this corresponding function in the parent class
        now this method takes an a board state and initiates a Minimax algorithm
        to help determine the move (a col number) that maximizes p2's utility value
        Based on AIMA text 3ed Fig. 5.3 function MINIMAX-DECISION(state)"""
        
        player = 2
        depth = SEARCH_DEPTH
        
        #Generate ACTIONS(s), which contains all the next moves.
        actions_set =[]
        
        for action in range(7):
            if Connect4Interface.is_move_valid(action, currentBoard):
                actions_set.append(action)
        
        next_boards_utility={}
        
        #calculate the util_valuty associated with each move.
        for move in actions_set:
            copy_of_board = copy.deepcopy(currentBoard) #creates copy of currentBoard
            Connect4Interface.place_disc(copy_of_board,move,player)
            next_boards_utility[move] = min_val(copy_of_board,1,depth-1) #MIN-VALUE(RESULT(state,a))
        
                
        #now next_boards_utilty is a dictionary {0: min_val, 1: min_va,.....}        
        move_max = None #the movethat maximizes the utility value
        util_max = -100000  #v <-- -infinity
        
        #find the move from the set ACTIONS(s) that maximizes the util value
        for key in list(next_boards_utility.keys()):   
            if next_boards_utility[key]>=util_max:
                util_max = next_boards_utility[key]
                move_max = key    
        
        print "AI picked:", move_max
        print "From:", next_boards_utility
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
    
    #if TERMINAL-TEST(state) then return UTILITY(state)
    if depth ==0 or Connect4Interface.player_won(boardX) or len(next_boards)==0:
        return heuristic_function(boardX,player,depth-1)
            
    #now find the board with lowest util value v
    v_min = 10000 # v<--  +infinity
    #for each a in ACTIONS(state) do
    for board_i in next_boards:
        v_min = min(v_min, max_val(board_i,opponent, depth-1))
        
    return v_min
    
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

    ###if TERMINAL-TEST(state) then return UTILITY(state)
    if depth ==0 or Connect4Interface.player_won(boardX) or len(next_boards)==0:
        return heuristic_function(boardX, player,depth-1)
                                                                            
    #now find the board with max util value v
    
    ###v <-- -infinity
    v_max = -10000
    
    ###for each a in ACTIONS(state) do
    for board_i in next_boards:
        ###v<--MAX(v,MIN-VALUE(board_i)
        v_max = max(v_max, min_val(board_i,opponent, depth-1))
    
    return v_max

        
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
        if (util_val) == -1:
            ##its theoretical loosing board. But how bad is it? some losing boards are "better"
            sumAll = sum(all_winning_comb_tally(allcomb))
            ##but, only use it if it is consistent with DTree prediction. Should still be losing board.
            if(sumAll<util_val):
                util_val=sumAll
        
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
    """Detect both players winning rows feature, added 051517"""
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
    """Counting the number of winning rows by each player"""
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

def load_records(game):
    """Loads the saved state from file"""
    lr = open('records.txt', 'r')
    sd = int(lr.readline())
    tc = int(lr.readline())
    gp = int(lr.readline())
    gt = int(lr.readline())
    p1 = int(lr.readline())
    p2 = int(lr.readline())
    game.set(gp,gt, p1, p2)
    lr.close()
    return sd,tc

def save_records(game):
    """Save the current states to text file"""
    sr = open('records.txt', 'w')
    g = game.__dict__
    sr.write(str(SEARCH_DEPTH)+'\n')
    sr.write(str(TRAINING_COUNT)+'\n')
    #gp = g['games_tied'] + g['p1_wins'] + g['p2_wins']
    #sr.write(str(gp)+'\n')
    sr.write(str(g['games_played'])+'\n')
    sr.write(str(g['games_tied'])+'\n')
    sr.write(str(g['p1_wins'])+'\n')
    sr.write(str(g['p2_wins'])+'\n')
    sr.close()
    
game3 = Connect4_AI()

while(True):
    
    rds = load_records(game3)

    SEARCH_DEPTH = rds[0]
    TRAINING_COUNT = rds[1]

    stats = game3.__dict__
    if stats['games_played'] != 0:
        winning_rate = float(stats['p2_wins']) / stats['games_played']
        if winning_rate < .5 and stats['games_played'] >= 3:
            game3.set(0,0,0,0)
            if TRAINING_COUNT <= 6000:
                TRAINING_COUNT *= 10
                print "AI is losing badly! Tranning data increases!"
            else:
                SEARCH_DEPTH += 1
                TRAINING_COUNT = 10
                print "AI is losing bigly! Tranning data resets and increase depth!"
                                                                                                                                                                                                                                                                                                                                                
    ####################Instantiating and Traiing the DTree########################

    ##Training data used are from http://archive.ics.uci.edu/ml/datasets/Connect-4
    ##See the website and our project report/presentation for details of data transformation.
    ##training_data.txt contains data from this database.

    training_data = load_data_array("training_data.txt",TRAINING_COUNT) #note this data from UCI
    #X,Y = seperateVariabes(training_data)
    #clf = tree.DecisionTreeClassifier()
    #clf = clf.fit(X,Y)

    X2,Y2 = trainWithFeatures(training_data)
    clf2 = tree.DecisionTreeClassifier()
    clf2 = clf2.fit(X2,Y2)
                                
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
    
    print "**********************************************************"
    print "Decision Tree Model:"
    print "   -Training sample size:", len(training_data)
    print "   -Verifying model on data from verificationData.txt:"        
    #print "     -correct:", correctPredictions
    #print "     -out of:", vdatasize
    print "     -Accuracy:", accuracy
    print "   -DTree Trained and Verified."
    print "**********************************************************"
    print

    game3.run_game()
    save_records(game3)

    user_input = raw_input("Would you like to player another game? (y/n): ")
    print
    if(user_input =='n'):
        break
    



####close file 051617
#f3Glob.close()    