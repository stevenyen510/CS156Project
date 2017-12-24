# Connect 4 Game AI with MinMax
# Synopsis
The user can play Connect 4 with an AI by running this program. The AI uses the Minimax algorithm to determine the moves to make. We also incorporated a machine learning component by using a Decision Tree Classifier from the SciKit Learn library. The Decision Tree Classifier is used to determine the heuristic value used in the algorithm (as the Min and Max values). The DTree model is trained using data from repository: http://archive.ics.uci.edu/ml/datasets/Connect-4.

# Running the Program

1. Before running the project, install Scikit Learn. This can be done through command line with the following command: <br>
```
pip install -U scikit-learn
```
2. Download all the files of this repository and save to the same directory. <br>
3. Through command line, change directory to the folder containing all the files. <br>
4. Run the program by running the Python file Minimax_withDTree.py <br>
```
python Minimax_withDTree.py
```
<br>

Alternatively, the program can also be ran using an IDE such as Canoyp. Below are the instructions for running the program in Canopy: <br>
1. To run the program, open Canopy code editor, and open the Minimax_withDTree.py file in the project folder. <br>
2. Before clicking run, remember to change the working directory to the project folder because the program need to reload and save records to records.txt, load training data for AI with training_data.txt, as well as calculate and verify accuracy with verificationData.txt. <br>
3. Run Minimax_withDTree.py, and the game will start for you, and follow the instruction to play the game. <br>

The game will now start, printing the board state and prompting the user to enter a move. Input 0 to 6 to put piece in board, and 9 to quit the program. After each game is finished, choose y/n to have another game or quit.<br>
