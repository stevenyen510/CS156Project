import Connect4Interface
import random

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
        
#print "New Game with AI"        
#gameWithAI_1 = GameWithAI()
#gameWithAI_1.run_game()

        
        
        