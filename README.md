# CS156Project

Connect 4 Game AI with MinMax

Before running project, install scikit_learn from canopy package manager or other equivalent ones, such as using pip install.

To run the program, open Canopy code editor, and open the Minimax_withDTree.py file in the project folder. 

Before click run, remember to change the working directory to the project folder because the program need to reload and save records to records.txt, load training data for AI with training_data.txt, as well as calculate and verify accuracy with verificationData.txt.

Run Minimax_withDTree.py, and the game will start for you, and follow the instruction to play the game.

Input 0 to 6 to put piece in board, and 9 to quit the program. After each game is finished, choose y/n to have another game or quit.

AI will automatically adjust its own SEARCH_DEPTH and number of training_data it grabs, which is TRAINING_COUNT, and each time when played 3 games or more and winning rate is less than 50%, AI will either increase TRAINING_COUNT by 10 times bigger, or increase SEARCH_DEPTH by 1 if TRAINING_COUNT is very large. This will result in increasing the difficulty of beating the AI.

p.s.
content for records.txt
"""
SEARCH_DEPTH
TRAINING_COUNT
games_played
games_tied
p1_wins
p2_wins
"""
starting point for AI:
2
10
0
0
0
0