import Connect4Interface
#import random
#import copy

from sklearn import tree
#since "from" is used, the classes ABCMeta and abstractmethod can be used w/o qualifying


CONNECT_FOUR_GRID_WIDTH = 7
CONNECT_FOUR_GRID_HEIGHT = 6
CONNECT_FOUR_COLORS = ["x", "o"]




training_data = ["b,b,b,b,b,b,b,b,b,b,b,b,x,o,b,b,b,b,x,o,x,o,x,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,win ",
"b,b,b,b,b,b,b,b,b,b,b,b,x,b,b,b,b,b,x,o,x,o,x,o,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,win ",
"b,b,b,b,b,b,o,b,b,b,b,b,x,b,b,b,b,b,x,o,x,o,x,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,win ",
"b,b,b,b,b,b,b,b,b,b,b,b,x,b,b,b,b,b,x,o,x,o,x,o,b,b,b,b,b,b,o,b,b,b,b,b,b,b,b,b,b,b,win ",
"o,b,b,b,b,b,b,b,b,b,b,b,x,b,b,b,b,b,x,o,x,o,x,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,win ",
"b,b,b,b,b,b,b,b,b,b,b,b,x,b,b,b,b,b,x,o,x,o,x,o,b,b,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,win ",
"b,b,b,b,b,b,x,b,b,b,b,b,o,b,b,b,b,b,x,o,x,o,x,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,draw",
"b,b,b,b,b,b,x,b,b,b,b,b,b,b,b,b,b,b,x,o,x,o,x,o,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,win ",
"b,b,b,b,b,b,x,o,b,b,b,b,b,b,b,b,b,b,x,o,x,o,x,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,win ",
"b,b,b,b,b,b,x,b,b,b,b,b,b,b,b,b,b,b,x,o,x,o,x,o,b,b,b,b,b,b,o,b,b,b,b,b,b,b,b,b,b,b,win ",
"o,b,b,b,b,b,x,b,b,b,b,b,b,b,b,b,b,b,x,o,x,o,x,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,win ",
"b,b,b,b,b,b,x,b,b,b,b,b,b,b,b,b,b,b,x,o,x,o,x,o,b,b,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,win ",
"x,b,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,x,o,x,o,x,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,win ",
"x,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,x,o,x,o,x,o,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,win ",
"x,b,b,b,b,b,o,b,b,b,b,b,b,b,b,b,b,b,x,o,x,o,x,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,loss",
"x,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,x,o,x,o,x,o,b,b,b,b,b,b,o,b,b,b,b,b,b,b,b,b,b,b,win ",
"x,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,x,o,x,o,x,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,win ",
"x,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,x,o,x,o,x,o,b,b,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,win ",
"b,b,b,b,b,b,b,b,b,b,b,b,o,o,b,b,b,b,x,o,x,o,x,x,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,loss",
"b,b,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,x,o,x,o,x,x,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,win ",
"b,b,b,b,b,b,o,b,b,b,b,b,o,b,b,b,b,b,x,o,x,o,x,x,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,win ",
"b,b,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,x,o,x,o,x,x,b,b,b,b,b,b,o,b,b,b,b,b,b,b,b,b,b,b,win ",
"o,b,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,x,o,x,o,x,x,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,win ",
"b,b,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,x,o,x,o,x,x,b,b,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,draw",
"b,b,b,b,b,b,b,b,b,b,b,b,o,x,b,b,b,b,x,o,x,o,x,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,win ",
"b,b,b,b,b,b,b,b,b,b,b,b,o,x,o,b,b,b,x,o,x,o,x,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,win ",
"b,b,b,b,b,b,b,b,b,b,b,b,o,x,b,b,b,b,x,o,x,o,x,b,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,win ",
"b,b,b,b,b,b,o,b,b,b,b,b,o,x,b,b,b,b,x,o,x,o,x,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,win ",
"b,b,b,b,b,b,b,b,b,b,b,b,o,x,b,b,b,b,x,o,x,o,x,b,b,b,b,b,b,b,o,b,b,b,b,b,b,b,b,b,b,b,win ",
"o,b,b,b,b,b,b,b,b,b,b,b,o,x,b,b,b,b,x,o,x,o,x,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,win ",
"b,b,b,b,b,b,b,b,b,b,b,b,o,x,b,b,b,b,x,o,x,o,x,b,b,b,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,win ",
"b,b,b,b,b,b,b,b,b,b,b,b,o,o,b,b,b,b,x,o,x,o,x,b,x,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,loss",
"b,b,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,x,o,x,o,x,b,x,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,draw",
"b,b,b,b,b,b,o,b,b,b,b,b,o,b,b,b,b,b,x,o,x,o,x,b,x,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,win ",
"b,b,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,x,o,x,o,x,b,x,b,b,b,b,b,o,b,b,b,b,b,b,b,b,b,b,b,loss",
"o,b,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,x,o,x,o,x,b,x,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,win ",
"b,b,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,x,o,x,o,x,b,x,b,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,win ",
"b,b,b,b,b,b,x,b,b,b,b,b,o,o,b,b,b,b,x,o,x,o,x,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,loss",
"b,b,b,b,b,b,x,b,b,b,b,b,o,b,b,b,b,b,x,o,x,o,x,b,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,win ",
"b,b,b,b,b,b,x,o,b,b,b,b,o,b,b,b,b,b,x,o,x,o,x,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,loss",
"b,b,b,b,b,b,x,b,b,b,b,b,o,b,b,b,b,b,x,o,x,o,x,b,b,b,b,b,b,b,o,b,b,b,b,b,b,b,b,b,b,b,draw",
"o,b,b,b,b,b,x,b,b,b,b,b,o,b,b,b,b,b,x,o,x,o,x,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,win ",
"b,b,b,b,b,b,x,b,b,b,b,b,o,b,b,b,b,b,x,o,x,o,x,b,b,b,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,draw",
"b,b,b,b,b,b,b,b,b,b,b,b,o,o,b,b,b,b,x,o,x,o,x,b,b,b,b,b,b,b,x,b,b,b,b,b,b,b,b,b,b,b,win ",
"b,b,b,b,b,b,o,b,b,b,b,b,o,b,b,b,b,b,x,o,x,o,x,b,b,b,b,b,b,b,x,b,b,b,b,b,b,b,b,b,b,b,win ",
"b,b,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,x,o,x,o,x,b,b,b,b,b,b,b,x,o,b,b,b,b,b,b,b,b,b,b,win ",
"o,b,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,x,o,x,o,x,b,b,b,b,b,b,b,x,b,b,b,b,b,b,b,b,b,b,b,win ",
"b,b,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,x,o,x,o,x,b,b,b,b,b,b,b,x,b,b,b,b,b,o,b,b,b,b,b,win ",
"x,b,b,b,b,b,b,b,b,b,b,b,o,o,b,b,b,b,x,o,x,o,x,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,loss",
"x,b,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,x,o,x,o,x,b,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,win ",
"x,b,b,b,b,b,o,b,b,b,b,b,o,b,b,b,b,b,x,o,x,o,x,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,win ",
"x,b,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,x,o,x,o,x,b,b,b,b,b,b,b,o,b,b,b,b,b,b,b,b,b,b,b,draw",
"x,o,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,x,o,x,o,x,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,win ",
"x,b,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,x,o,x,o,x,b,b,b,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,win ",
"b,b,b,b,b,b,b,b,b,b,b,b,o,o,b,b,b,b,x,o,x,o,x,b,b,b,b,b,b,b,b,b,b,b,b,b,x,b,b,b,b,b,win ",
"b,b,b,b,b,b,o,b,b,b,b,b,o,b,b,b,b,b,x,o,x,o,x,b,b,b,b,b,b,b,b,b,b,b,b,b,x,b,b,b,b,b,win ",
"b,b,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,x,o,x,o,x,b,b,b,b,b,b,b,o,b,b,b,b,b,x,b,b,b,b,b,loss",
"o,b,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,x,o,x,o,x,b,b,b,b,b,b,b,b,b,b,b,b,b,x,b,b,b,b,b,win ",
"b,b,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,x,o,x,o,x,b,b,b,b,b,b,b,b,b,b,b,b,b,x,o,b,b,b,b,win ",
"b,b,b,b,b,b,o,o,b,b,b,b,b,b,b,b,b,b,x,o,x,o,x,x,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,loss",
"b,b,b,b,b,b,o,b,b,b,b,b,b,b,b,b,b,b,x,o,x,o,x,x,b,b,b,b,b,b,o,b,b,b,b,b,b,b,b,b,b,b,loss",
"o,b,b,b,b,b,o,b,b,b,b,b,b,b,b,b,b,b,x,o,x,o,x,x,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,win ",
"b,b,b,b,b,b,o,b,b,b,b,b,b,b,b,b,b,b,x,o,x,o,x,x,b,b,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,draw",
"b,b,b,b,b,b,o,b,b,b,b,b,x,o,b,b,b,b,x,o,x,o,x,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,win ",
"b,b,b,b,b,b,o,o,b,b,b,b,x,b,b,b,b,b,x,o,x,o,x,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,loss",
"b,b,b,b,b,b,o,b,b,b,b,b,x,b,b,b,b,b,x,o,x,o,x,b,b,b,b,b,b,b,o,b,b,b,b,b,b,b,b,b,b,b,draw",
"o,b,b,b,b,b,o,b,b,b,b,b,x,b,b,b,b,b,x,o,x,o,x,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,win ",
"b,b,b,b,b,b,o,b,b,b,b,b,x,b,b,b,b,b,x,o,x,o,x,b,b,b,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,win ",
"b,b,b,b,b,b,o,b,b,b,b,b,b,b,b,b,b,b,x,o,x,o,x,b,x,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,win ",
"b,b,b,b,b,b,o,o,b,b,b,b,b,b,b,b,b,b,x,o,x,o,x,b,x,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,win ",
"o,b,b,b,b,b,o,b,b,b,b,b,b,b,b,b,b,b,x,o,x,o,x,b,x,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,win ",
"b,b,b,b,b,b,o,b,b,b,b,b,b,b,b,b,b,b,x,o,x,o,x,b,x,b,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,win ",
"b,b,b,b,b,b,o,x,b,b,b,b,b,b,b,b,b,b,x,o,x,o,x,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,win ",
"b,b,b,b,b,b,o,x,b,b,b,b,o,b,b,b,b,b,x,o,x,o,x,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,win ",
"b,b,b,b,b,b,o,x,b,b,b,b,b,b,b,b,b,b,x,o,x,o,x,b,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,win ",
"b,b,b,b,b,b,o,x,o,b,b,b,b,b,b,b,b,b,x,o,x,o,x,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,win ",
"b,b,b,b,b,b,o,x,b,b,b,b,b,b,b,b,b,b,x,o,x,o,x,b,b,b,b,b,b,b,o,b,b,b,b,b,b,b,b,b,b,b,win ",
"o,b,b,b,b,b,o,x,b,b,b,b,b,b,b,b,b,b,x,o,x,o,x,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,win ",
"b,b,b,b,b,b,o,x,b,b,b,b,b,b,b,b,b,b,x,o,x,o,x,b,b,b,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,win ",
"b,b,b,b,b,b,o,o,b,b,b,b,b,b,b,b,b,b,x,o,x,o,x,b,b,b,b,b,b,b,x,b,b,b,b,b,b,b,b,b,b,b,win ",
"b,b,b,b,b,b,o,b,b,b,b,b,b,b,b,b,b,b,x,o,x,o,x,b,b,b,b,b,b,b,x,o,b,b,b,b,b,b,b,b,b,b,win ",
"o,b,b,b,b,b,o,b,b,b,b,b,b,b,b,b,b,b,x,o,x,o,x,b,b,b,b,b,b,b,x,b,b,b,b,b,b,b,b,b,b,b,win ",
"b,b,b,b,b,b,o,b,b,b,b,b,b,b,b,b,b,b,x,o,x,o,x,b,b,b,b,b,b,b,x,b,b,b,b,b,o,b,b,b,b,b,win ",
"x,b,b,b,b,b,o,o,b,b,b,b,b,b,b,b,b,b,x,o,x,o,x,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,draw",
"x,b,b,b,b,b,o,b,b,b,b,b,b,b,b,b,b,b,x,o,x,o,x,b,b,b,b,b,b,b,o,b,b,b,b,b,b,b,b,b,b,b,draw",
"x,o,b,b,b,b,o,b,b,b,b,b,b,b,b,b,b,b,x,o,x,o,x,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,draw",
"x,b,b,b,b,b,o,b,b,b,b,b,b,b,b,b,b,b,x,o,x,o,x,b,b,b,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,draw",
"b,b,b,b,b,b,o,o,b,b,b,b,b,b,b,b,b,b,x,o,x,o,x,b,b,b,b,b,b,b,b,b,b,b,b,b,x,b,b,b,b,b,loss",
"o,b,b,b,b,b,o,b,b,b,b,b,b,b,b,b,b,b,x,o,x,o,x,b,b,b,b,b,b,b,b,b,b,b,b,b,x,b,b,b,b,b,win ",
"b,b,b,b,b,b,o,b,b,b,b,b,b,b,b,b,b,b,x,o,x,o,x,b,b,b,b,b,b,b,b,b,b,b,b,b,x,o,b,b,b,b,win ",
"o,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,x,o,x,o,x,x,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,win ",
"o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,x,o,x,o,x,x,b,b,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,loss",
"o,b,b,b,b,b,b,b,b,b,b,b,x,o,b,b,b,b,x,o,x,o,x,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,win ",
"o,o,b,b,b,b,b,b,b,b,b,b,x,b,b,b,b,b,x,o,x,o,x,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,win ",
"o,b,b,b,b,b,b,b,b,b,b,b,x,b,b,b,b,b,x,o,x,o,x,b,b,b,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,win ",
"o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,x,o,x,o,x,b,x,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,win ",
"o,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,x,o,x,o,x,b,x,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,win ",
"o,b,b,b,b,b,x,o,b,b,b,b,b,b,b,b,b,b,x,o,x,o,x,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,loss",
"o,o,b,b,b,b,x,b,b,b,b,b,b,b,b,b,b,b,x,o,x,o,x,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,win ",
"o,b,b,b,b,b,x,b,b,b,b,b,b,b,b,b,b,b,x,o,x,o,x,b,b,b,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,win ",
"o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,x,o,x,o,x,b,b,b,b,b,b,b,x,o,b,b,b,b,b,b,b,b,b,b,win ",
"o,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,x,o,x,o,x,b,b,b,b,b,b,b,x,b,b,b,b,b,b,b,b,b,b,b,win ",
"o,x,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,x,o,x,o,x,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,win ",
"o,x,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,x,o,x,o,x,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,win ",
"o,x,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,x,o,x,o,x,b,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,win ",
"o,x,b,b,b,b,o,b,b,b,b,b,b,b,b,b,b,b,x,o,x,o,x,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,win ",
"o,x,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,x,o,x,o,x,b,b,b,b,b,b,b,o,b,b,b,b,b,b,b,b,b,b,b,draw",
"o,x,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,x,o,x,o,x,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,win ",
"o,x,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,x,o,x,o,x,b,b,b,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,draw",
"o,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,x,o,x,o,x,b,b,b,b,b,b,b,b,b,b,b,b,b,x,b,b,b,b,b,win ",
"o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,x,o,x,o,x,b,b,b,b,b,b,b,b,b,b,b,b,b,x,o,b,b,b,b,win ",
"b,b,b,b,b,b,b,b,b,b,b,b,x,o,b,b,b,b,x,o,x,o,o,x,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,win ",
"b,b,b,b,b,b,b,b,b,b,b,b,x,b,b,b,b,b,x,o,x,o,o,x,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,loss",
"b,b,b,b,b,b,o,b,b,b,b,b,x,b,b,b,b,b,x,o,x,o,o,x,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,loss",
"b,b,b,b,b,b,b,b,b,b,b,b,x,b,b,b,b,b,x,o,x,o,o,x,b,b,b,b,b,b,o,b,b,b,b,b,b,b,b,b,b,b,win ",
"o,b,b,b,b,b,b,b,b,b,b,b,x,b,b,b,b,b,x,o,x,o,o,x,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,win ",
"b,b,b,b,b,b,b,b,b,b,b,b,x,b,b,b,b,b,x,o,x,o,o,x,b,b,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,win ",
"b,b,b,b,b,b,b,b,b,b,b,b,x,x,b,b,b,b,x,o,x,o,o,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,win ",
"b,b,b,b,b,b,b,b,b,b,b,b,x,x,o,b,b,b,x,o,x,o,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,win ",
"b,b,b,b,b,b,b,b,b,b,b,b,x,x,b,b,b,b,x,o,x,o,o,b,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,win ",
"b,b,b,b,b,b,o,b,b,b,b,b,x,x,b,b,b,b,x,o,x,o,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,win ",
"b,b,b,b,b,b,b,b,b,b,b,b,x,x,b,b,b,b,x,o,x,o,o,b,b,b,b,b,b,b,o,b,b,b,b,b,b,b,b,b,b,b,win ",
"o,b,b,b,b,b,b,b,b,b,b,b,x,x,b,b,b,b,x,o,x,o,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,win ",
"b,b,b,b,b,b,b,b,b,b,b,b,x,x,b,b,b,b,x,o,x,o,o,b,b,b,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,win ",
"b,b,b,b,b,b,b,b,b,b,b,b,x,b,b,b,b,b,x,o,x,o,o,b,o,b,b,b,b,b,x,b,b,b,b,b,b,b,b,b,b,b,loss",
"x,b,b,b,b,b,o,b,b,b,b,b,x,b,b,b,b,b,x,o,x,o,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,loss",
"b,b,b,b,b,b,b,b,b,b,b,b,x,b,b,b,b,b,x,o,x,o,o,o,b,b,b,b,b,b,b,b,b,b,b,b,x,b,b,b,b,b,win ",
"b,b,b,b,b,b,b,b,b,b,b,b,x,o,b,b,b,b,x,o,x,o,o,b,b,b,b,b,b,b,b,b,b,b,b,b,x,b,b,b,b,b,win ",
"b,b,b,b,b,b,b,b,b,b,b,b,x,b,b,b,b,b,x,o,x,o,o,b,o,b,b,b,b,b,b,b,b,b,b,b,x,b,b,b,b,b,loss",
"b,b,b,b,b,b,o,b,b,b,b,b,x,b,b,b,b,b,x,o,x,o,o,b,b,b,b,b,b,b,b,b,b,b,b,b,x,b,b,b,b,b,loss",
"b,b,b,b,b,b,b,b,b,b,b,b,x,b,b,b,b,b,x,o,x,o,o,b,b,b,b,b,b,b,o,b,b,b,b,b,x,b,b,b,b,b,win ",
"o,b,b,b,b,b,b,b,b,b,b,b,x,b,b,b,b,b,x,o,x,o,o,b,b,b,b,b,b,b,b,b,b,b,b,b,x,b,b,b,b,b,win ",
"b,b,b,b,b,b,b,b,b,b,b,b,x,b,b,b,b,b,x,o,x,o,o,b,b,b,b,b,b,b,b,b,b,b,b,b,x,o,b,b,b,b,win ",
"b,b,b,b,b,b,b,b,b,b,b,b,x,o,o,b,b,b,x,o,x,o,x,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,win ",
"b,b,b,b,b,b,b,b,b,b,b,b,x,o,x,b,b,b,x,o,x,o,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,win ",
"b,b,b,b,b,b,b,b,b,b,b,b,x,o,x,o,b,b,x,o,x,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,win ",
"b,b,b,b,b,b,b,b,b,b,b,b,x,o,x,b,b,b,x,o,x,o,b,b,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,win ",
"b,b,b,b,b,b,o,b,b,b,b,b,x,o,x,b,b,b,x,o,x,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,win ",
"b,b,b,b,b,b,b,b,b,b,b,b,x,o,x,b,b,b,x,o,x,o,b,b,b,b,b,b,b,b,o,b,b,b,b,b,b,b,b,b,b,b,win ",
"o,b,b,b,b,b,b,b,b,b,b,b,x,o,x,b,b,b,x,o,x,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,win ",
"b,b,b,b,b,b,b,b,b,b,b,b,x,o,x,b,b,b,x,o,x,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,win ",
"b,b,b,b,b,b,b,b,b,b,b,b,x,o,b,b,b,b,x,o,x,o,b,b,o,b,b,b,b,b,x,b,b,b,b,b,b,b,b,b,b,b,loss",
"x,b,b,b,b,b,o,b,b,b,b,b,x,o,b,b,b,b,x,o,x,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,loss",
"b,b,b,b,b,b,b,b,b,b,b,b,x,o,o,b,b,b,x,o,x,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,x,b,b,b,b,b,win ",
"b,b,b,b,b,b,b,b,b,b,b,b,x,o,b,b,b,b,x,o,x,o,b,b,o,b,b,b,b,b,b,b,b,b,b,b,x,b,b,b,b,b,loss",
"b,b,b,b,b,b,o,b,b,b,b,b,x,o,b,b,b,b,x,o,x,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,x,b,b,b,b,b,loss",
"b,b,b,b,b,b,b,b,b,b,b,b,x,o,b,b,b,b,x,o,x,o,b,b,b,b,b,b,b,b,o,b,b,b,b,b,x,b,b,b,b,b,win ",
"o,b,b,b,b,b,b,b,b,b,b,b,x,o,b,b,b,b,x,o,x,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,x,b,b,b,b,b,win ",
"b,b,b,b,b,b,b,b,b,b,b,b,x,o,b,b,b,b,x,o,x,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,x,o,b,b,b,b,win ",
"b,b,b,b,b,b,b,b,b,b,b,b,x,x,o,b,b,b,x,o,x,o,b,b,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,loss",
"b,b,b,b,b,b,b,b,b,b,b,b,x,x,b,b,b,b,x,o,x,o,b,b,o,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,win ",
"b,b,b,b,b,b,o,b,b,b,b,b,x,x,b,b,b,b,x,o,x,o,b,b,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,win ",
"b,b,b,b,b,b,b,b,b,b,b,b,x,x,b,b,b,b,x,o,x,o,b,b,o,b,b,b,b,b,o,b,b,b,b,b,b,b,b,b,b,b,win ",
"o,b,b,b,b,b,b,b,b,b,b,b,x,x,b,b,b,b,x,o,x,o,b,b,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,win ",
"b,b,b,b,b,b,b,b,b,b,b,b,x,x,b,b,b,b,x,o,x,o,b,b,o,b,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,win ",
"b,b,b,b,b,b,b,b,b,b,b,b,x,b,b,b,b,b,x,o,x,o,o,b,o,x,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,win ",
"b,b,b,b,b,b,b,b,b,b,b,b,x,o,b,b,b,b,x,o,x,o,b,b,o,x,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,win ",
"b,b,b,b,b,b,b,b,b,b,b,b,x,b,b,b,b,b,x,o,x,o,b,b,o,x,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,loss",
"b,b,b,b,b,b,o,b,b,b,b,b,x,b,b,b,b,b,x,o,x,o,b,b,o,x,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,win ",
"b,b,b,b,b,b,b,b,b,b,b,b,x,b,b,b,b,b,x,o,x,o,b,b,o,x,b,b,b,b,o,b,b,b,b,b,b,b,b,b,b,b,win ",
"o,b,b,b,b,b,b,b,b,b,b,b,x,b,b,b,b,b,x,o,x,o,b,b,o,x,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,win ",
"b,b,b,b,b,b,b,b,b,b,b,b,x,b,b,b,b,b,x,o,x,o,b,b,o,x,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,win ",
"o,b,b,b,b,b,x,b,b,b,b,b,x,b,b,b,b,b,x,o,x,o,b,b,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,loss",
"b,b,b,b,b,b,b,b,b,b,b,b,x,b,b,b,b,b,x,o,x,o,b,b,o,o,b,b,b,b,x,b,b,b,b,b,b,b,b,b,b,b,loss",
"b,b,b,b,b,b,o,b,b,b,b,b,x,b,b,b,b,b,x,o,x,o,b,b,o,b,b,b,b,b,x,b,b,b,b,b,b,b,b,b,b,b,loss",
"b,b,b,b,b,b,b,b,b,b,b,b,x,b,b,b,b,b,x,o,x,o,b,b,o,b,b,b,b,b,x,o,b,b,b,b,b,b,b,b,b,b,loss",
"o,b,b,b,b,b,b,b,b,b,b,b,x,b,b,b,b,b,x,o,x,o,b,b,o,b,b,b,b,b,x,b,b,b,b,b,b,b,b,b,b,b,win ",
"b,b,b,b,b,b,b,b,b,b,b,b,x,b,b,b,b,b,x,o,x,o,b,b,o,b,b,b,b,b,x,b,b,b,b,b,o,b,b,b,b,b,loss",
"x,b,b,b,b,b,o,b,b,b,b,b,x,b,b,b,b,b,x,o,x,o,b,b,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,loss",
"b,b,b,b,b,b,b,b,b,b,b,b,x,b,b,b,b,b,x,o,x,o,b,b,o,o,b,b,b,b,b,b,b,b,b,b,x,b,b,b,b,b,loss",
"b,b,b,b,b,b,o,b,b,b,b,b,x,b,b,b,b,b,x,o,x,o,b,b,o,b,b,b,b,b,b,b,b,b,b,b,x,b,b,b,b,b,loss",
"b,b,b,b,b,b,b,b,b,b,b,b,x,b,b,b,b,b,x,o,x,o,b,b,o,b,b,b,b,b,o,b,b,b,b,b,x,b,b,b,b,b,draw",
"o,b,b,b,b,b,b,b,b,b,b,b,x,b,b,b,b,b,x,o,x,o,b,b,o,b,b,b,b,b,b,b,b,b,b,b,x,b,b,b,b,b,loss",
"b,b,b,b,b,b,b,b,b,b,b,b,x,b,b,b,b,b,x,o,x,o,b,b,o,b,b,b,b,b,b,b,b,b,b,b,x,o,b,b,b,b,win ",
"b,b,b,b,b,b,o,b,b,b,b,b,x,x,o,b,b,b,x,o,x,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,loss",
"b,b,b,b,b,b,o,o,b,b,b,b,x,x,b,b,b,b,x,o,x,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,win ",
"b,b,b,b,b,b,o,b,b,b,b,b,x,x,b,b,b,b,x,o,x,o,b,b,b,b,b,b,b,b,o,b,b,b,b,b,b,b,b,b,b,b,win ",
"o,b,b,b,b,b,o,b,b,b,b,b,x,x,b,b,b,b,x,o,x,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,win ",
"b,b,b,b,b,b,o,b,b,b,b,b,x,x,b,b,b,b,x,o,x,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,win ",
"b,b,b,b,b,b,o,b,b,b,b,b,x,b,b,b,b,b,x,o,x,o,b,b,x,b,b,b,b,b,o,b,b,b,b,b,b,b,b,b,b,b,loss",
"b,b,b,b,b,b,o,x,b,b,b,b,x,b,b,b,b,b,x,o,x,o,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,win ",
"b,b,b,b,b,b,o,x,b,b,b,b,x,o,b,b,b,b,x,o,x,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,win ",
"b,b,b,b,b,b,o,x,b,b,b,b,x,b,b,b,b,b,x,o,x,o,b,b,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,win ",
"b,b,b,b,b,b,o,x,o,b,b,b,x,b,b,b,b,b,x,o,x,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,loss",
"b,b,b,b,b,b,o,x,b,b,b,b,x,b,b,b,b,b,x,o,x,o,b,b,b,b,b,b,b,b,o,b,b,b,b,b,b,b,b,b,b,b,draw",
"o,b,b,b,b,b,o,x,b,b,b,b,x,b,b,b,b,b,x,o,x,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,win ",
"b,b,b,b,b,b,o,x,b,b,b,b,x,b,b,b,b,b,x,o,x,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,win ",
"x,b,b,b,b,b,o,o,b,b,b,b,x,b,b,b,b,b,x,o,x,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,loss",
"x,b,b,b,b,b,o,b,b,b,b,b,x,b,b,b,b,b,x,o,x,o,b,b,b,b,b,b,b,b,o,b,b,b,b,b,b,b,b,b,b,b,loss",
"x,o,b,b,b,b,o,b,b,b,b,b,x,b,b,b,b,b,x,o,x,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,loss",
"x,b,b,b,b,b,o,b,b,b,b,b,x,b,b,b,b,b,x,o,x,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,draw",
"b,b,b,b,b,b,o,o,b,b,b,b,x,b,b,b,b,b,x,o,x,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,x,b,b,b,b,b,loss",
"b,b,b,b,b,b,o,b,b,b,b,b,x,b,b,b,b,b,x,o,x,o,b,b,b,b,b,b,b,b,o,b,b,b,b,b,x,b,b,b,b,b,loss",
"o,b,b,b,b,b,o,b,b,b,b,b,x,b,b,b,b,b,x,o,x,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,x,b,b,b,b,b,loss",
"b,b,b,b,b,b,o,b,b,b,b,b,x,b,b,b,b,b,x,o,x,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,x,o,b,b,b,b,win ",
"b,b,b,b,b,b,b,b,b,b,b,b,x,x,o,b,b,b,x,o,x,o,b,b,b,b,b,b,b,b,o,b,b,b,b,b,b,b,b,b,b,b,win ",
"b,b,b,b,b,b,b,b,b,b,b,b,x,x,b,b,b,b,x,o,x,o,b,b,b,b,b,b,b,b,o,o,b,b,b,b,b,b,b,b,b,b,win ",
"o,b,b,b,b,b,b,b,b,b,b,b,x,x,b,b,b,b,x,o,x,o,b,b,b,b,b,b,b,b,o,b,b,b,b,b,b,b,b,b,b,b,win ",
"b,b,b,b,b,b,b,b,b,b,b,b,x,x,b,b,b,b,x,o,x,o,b,b,b,b,b,b,b,b,o,b,b,b,b,b,o,b,b,b,b,b,win ",
"b,b,b,b,b,b,b,b,b,b,b,b,x,b,b,b,b,b,x,o,x,o,o,b,b,b,b,b,b,b,o,x,b,b,b,b,b,b,b,b,b,b,win ",
"x,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,o,x,b,b,b,o,x,x,b,b,b,win ",
"x,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,x,b,b,b,b,o,x,x,o,o,b,win ",
"x,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,b,b,b,b,b,b,o,x,x,o,x,b,win ",
"x,o,o,x,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,b,b,b,b,b,b,o,x,x,b,b,b,win ",
"x,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,x,o,x,o,b,o,x,b,b,b,b,draw",
"x,x,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,b,b,b,b,b,b,o,o,x,o,x,b,win ",
"x,x,o,o,x,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,o,x,b,b,b,b,win ",
"x,b,b,b,b,b,o,o,x,b,b,b,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,x,x,b,b,b,win ",
"x,o,x,x,o,x,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,loss",
"x,b,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,x,x,o,x,o,draw",
"x,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,o,x,o,b,b,o,x,x,b,b,b,win ",
"x,b,b,b,b,b,b,b,b,b,b,b,o,o,x,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,o,x,x,b,b,b,loss",
"x,o,x,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,o,x,o,x,b,b,win ",
"x,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,b,b,b,b,b,b,o,x,x,x,o,b,win ",
"x,o,b,b,b,b,b,b,b,b,b,b,x,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,x,x,o,o,b,win ",
"x,b,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,x,o,x,x,b,loss",
"x,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,o,b,b,b,b,o,x,x,x,o,b,draw",
"x,x,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,o,o,x,x,b,b,win ",
"x,o,o,x,b,b,b,b,b,b,b,b,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,x,x,b,b,b,win ",
"x,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,o,x,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,x,x,o,b,b,loss",
"x,b,b,b,b,b,o,b,b,b,b,b,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,x,x,o,x,b,loss",
"x,x,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,x,o,o,x,b,loss",
"x,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,b,b,b,b,b,b,o,x,x,o,x,o,win ",
"x,b,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,b,b,b,b,b,b,o,x,o,x,x,b,loss",
"x,b,b,b,b,b,b,b,b,b,b,b,o,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,x,o,x,x,b,loss",
"x,b,b,b,b,b,o,o,x,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,x,x,b,b,b,loss",
"x,x,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,x,o,o,x,o,win ",
"x,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,b,b,b,b,b,b,o,x,o,x,x,o,win ",
"x,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,o,x,x,o,x,o,win ",
"x,o,x,x,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,o,x,b,b,b,win ",
"x,b,b,b,b,b,b,b,b,b,b,b,o,o,x,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,x,x,o,b,b,loss",
"x,b,b,b,b,b,o,b,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,x,o,x,x,b,loss",
"x,o,o,x,x,b,b,b,b,b,b,b,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,x,b,b,b,b,b,draw",
"x,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,x,x,o,x,o,win ",
"x,o,o,x,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,x,x,b,b,b,win ",
"x,x,x,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,o,o,x,b,b,win ",
"x,x,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,x,o,x,o,o,win ",
"x,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,o,x,o,b,b,b,b,b,b,b,b,o,x,x,b,b,b,win ",
"x,o,x,o,x,b,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,x,b,b,b,b,loss",
"x,o,o,x,x,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,x,b,b,b,b,b,o,b,b,b,b,b,win ",
"x,x,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,o,x,o,x,b,draw",
"x,b,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,x,x,x,o,b,loss",
"x,x,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,x,o,o,x,b,win ",
"x,o,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,x,x,o,x,b,win ",
"x,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,o,x,o,x,x,o,win ",
"x,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,x,o,x,x,o,draw",
"x,x,o,o,x,b,x,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,loss",
"x,b,b,b,b,b,o,o,x,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,b,b,b,b,b,b,o,x,x,b,b,b,draw",
"x,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,x,o,x,o,x,loss",
"x,o,b,b,b,b,x,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,x,x,o,o,b,win ",
"x,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,x,o,x,x,o,win ",
"x,o,o,o,x,b,x,b,b,b,b,b,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,x,b,b,b,b,b,b,b,b,b,b,b,win ",
"x,x,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,b,b,b,b,b,b,o,x,o,o,x,b,win ",
"x,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,o,x,b,b,b,o,x,x,o,b,b,win ",
"x,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,x,o,x,x,b,loss",
"x,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,o,b,b,b,b,b,b,b,b,b,b,o,x,x,o,x,b,loss",
"x,o,x,b,b,b,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,x,x,o,b,b,draw",
"x,x,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,o,x,o,o,x,b,win ",
"x,b,b,b,b,b,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,b,b,b,b,b,b,o,x,o,x,x,b,loss",
"x,x,o,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,o,x,x,b,b,draw",
"x,x,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,o,x,x,o,b,win ",
"x,x,o,x,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,x,o,b,b,b,win ",
"x,o,x,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,o,x,x,o,b,b,win ",
"x,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,x,o,b,b,b,o,x,x,b,b,b,loss",
"x,b,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,x,o,x,x,o,draw",
"x,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,o,x,b,b,b,b,b,b,b,b,b,o,x,x,o,b,b,win ",
"x,o,x,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,x,o,x,b,b,draw",
"x,x,b,b,b,b,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,o,x,x,o,b,win ",
"x,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,o,b,b,b,b,o,x,o,x,x,b,draw",
"x,o,o,o,x,b,b,b,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,x,x,b,b,b,b,loss",
"x,x,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,o,x,x,o,o,b,win ",
"x,o,o,o,x,b,b,b,b,b,b,b,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,x,x,b,b,b,b,win ",
"x,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,x,o,x,x,b,draw",
"x,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,o,b,b,b,b,b,b,b,b,b,b,o,x,o,x,x,b,loss",
"x,x,o,o,x,b,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,x,b,b,b,b,loss",
"x,b,b,b,b,b,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,o,x,x,x,o,b,draw",
"x,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,b,b,b,b,b,b,o,x,o,x,x,b,win ",
"x,o,o,x,x,b,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,x,b,b,b,b,b,draw",
"x,b,b,b,b,b,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,o,x,x,o,x,b,loss",
"x,o,o,x,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,x,x,b,b,b,loss",
"x,b,b,b,b,b,o,o,o,x,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,x,x,b,b,b,b,draw",
"x,o,x,o,b,b,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,x,x,b,b,b,loss",
"x,o,o,x,x,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,x,b,b,b,b,b,o,b,b,b,b,b,o,b,b,b,b,b,win ",
"x,b,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,b,b,b,b,b,b,o,o,x,b,b,b,b,b,b,b,b,b,o,x,x,b,b,b,loss",
"x,x,o,x,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,x,o,o,b,b,win ",
"x,x,b,b,b,b,o,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,o,x,x,b,b,loss",
"x,o,x,o,x,x,b,b,b,b,b,b,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,loss",
"x,x,o,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,x,x,o,b,b,draw",
"x,x,o,x,b,b,b,b,b,b,b,b,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,o,x,b,b,b,loss",
"x,o,o,o,x,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,x,x,b,b,b,b,win ",
"x,b,b,b,b,b,o,o,x,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,o,x,x,b,b,b,loss",
"x,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,o,b,b,b,b,b,b,b,b,b,b,o,x,x,x,o,b,draw",
"x,b,b,b,b,b,o,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,x,o,x,x,b,loss",
"x,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,x,b,b,b,b,b,b,b,b,b,b,b,o,x,x,o,o,b,win ",
"x,b,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,x,o,x,o,x,draw",
"x,x,o,o,x,b,b,b,b,b,b,b,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,x,b,b,b,b,draw",
"x,o,o,o,x,b,b,b,b,b,b,b,x,x,b,b,b,b,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,win ",
"x,o,o,x,x,b,o,b,b,b,b,b,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,x,b,b,b,b,b,win ",
"x,o,x,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,x,x,o,b,b,draw",
"x,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,o,x,b,b,b,b,b,b,b,b,b,o,x,x,b,b,b,win ",
"x,o,x,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,o,x,x,b,b,b,win ",
"x,x,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,x,o,x,o,b,win ",
"x,b,b,b,b,b,o,b,b,b,b,b,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,x,o,x,x,b,loss",
"x,b,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,o,x,o,x,x,b,draw",
"x,o,o,x,b,b,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,x,x,b,b,b,win ",
"x,b,b,b,b,b,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,x,o,x,x,o,win ",
"x,o,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,x,x,o,o,b,b,x,b,b,b,b,b,b,b,b,b,b,b,win ",
"x,x,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,x,x,o,o,b,win ",
"x,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,o,x,o,x,x,o,b,b,b,b,b,b,win ",
"x,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,o,x,x,x,o,b,b,b,b,b,b,b,win ",
"x,o,x,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,x,x,o,o,b,win ",
"x,x,o,o,x,b,b,b,b,b,b,b,b,b,b,b,b,b,x,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,win ",
"x,o,x,b,b,b,o,b,b,b,b,b,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,x,x,b,b,b,b,b,b,b,b,b,win ",
"x,x,o,b,b,b,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,o,x,x,b,b,draw",
"x,o,b,b,b,b,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,x,x,o,x,b,draw",
"x,x,o,x,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,o,x,o,b,b,win ",
"x,o,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,x,x,x,o,b,win ",
"x,b,b,b,b,b,o,b,b,b,b,b,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,x,x,x,o,b,draw",
"x,o,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,x,o,x,x,b,win ",
"x,b,b,b,b,b,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,x,x,o,x,o,draw",
"x,b,b,b,b,b,b,b,b,b,b,b,o,x,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,x,x,o,o,b,win ",
"x,o,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,x,x,o,x,b,draw",
"x,o,x,b,b,b,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,o,x,x,b,b,b,draw",
"x,x,x,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,o,x,o,b,b,win ",
"x,x,o,o,x,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,x,b,b,b,b,b,b,b,b,b,b,b,o,o,b,b,b,b,win ",
"x,o,x,o,x,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,x,o,b,b,b,draw",
"x,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,x,b,b,b,b,b,o,x,x,o,o,b,win ",
"x,o,o,x,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,x,x,o,b,b,win ",
"x,o,o,o,x,b,x,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,x,o,b,b,b,b,b,b,b,b,b,b,draw",
"x,b,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,o,x,x,x,o,b,draw",
"x,x,x,o,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,o,x,b,b,b,draw",
"x,o,o,o,x,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,x,b,b,b,b,b,o,b,b,b,b,b,x,b,b,b,b,b,draw",
"x,x,x,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,x,o,o,b,b,draw",
"x,b,b,b,b,b,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,x,o,x,o,x,loss",
"x,o,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,x,o,x,x,b,draw",
"x,x,x,o,b,b,b,b,b,b,b,b,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,x,o,b,b,b,draw",
"x,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,b,b,b,b,b,b,o,x,o,x,x,b,b,b,b,b,b,b,win ",
"x,b,b,b,b,b,o,o,x,o,x,b,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,x,b,b,b,b,b,draw",
"x,b,b,b,b,b,b,b,b,b,b,b,o,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,x,o,x,x,b,b,b,b,b,b,b,draw",
"x,b,b,b,b,b,o,x,b,b,b,b,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,x,o,o,x,b,b,win ",
"x,o,b,b,b,b,o,o,x,x,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,x,b,b,b,b,b,loss",
"x,x,o,b,b,b,o,x,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,x,o,o,b,b,b,win ",
"x,b,b,b,b,b,o,o,x,x,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,b,b,b,b,b,b,x,o,b,b,b,b,loss",
"x,x,o,o,b,b,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,x,x,b,b,b,draw",
"x,x,x,o,o,b,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,b,b,b,b,b,b,x,b,b,b,b,b,draw",
"x,x,o,o,x,b,o,o,b,b,b,b,b,b,b,b,b,b,x,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,draw",
"x,x,x,o,o,b,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,x,b,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,win ",
"x,x,b,b,b,b,o,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,o,x,x,b,b,b,loss",
"x,x,o,x,b,b,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,x,o,o,b,b,b,win ",
"x,o,b,b,b,b,o,o,x,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,x,x,b,b,b,b,loss",
"x,b,b,b,b,b,o,o,o,x,b,b,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,x,x,b,b,b,b,win ",
"x,x,x,o,x,b,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,o,b,b,b,b,draw",
"x,b,b,b,b,b,o,o,x,o,x,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,x,b,b,b,b,b,draw",
"x,x,x,o,b,b,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,o,x,b,b,b,win ",
"x,b,b,b,b,b,o,o,x,o,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,x,x,b,b,b,b,loss",
"x,b,b,b,b,b,o,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,x,x,o,x,o,b,loss",
"x,o,b,b,b,b,o,x,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,x,x,o,o,b,b,win ",
"x,x,o,x,o,x,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,loss",
"x,o,x,x,b,b,o,b,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,x,b,b,b,b,loss",
"x,b,b,b,b,b,o,o,x,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,x,x,o,b,b,b,loss",
"x,x,b,b,b,b,o,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,x,o,x,b,b,win ",
"x,o,b,b,b,b,o,b,b,b,b,b,x,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,x,x,o,o,b,b,loss",
"x,b,b,b,b,b,o,o,x,x,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,x,b,b,b,b,b,loss",
"x,x,o,x,x,o,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,b,b,b,b,b,b,draw",
"x,b,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,o,x,x,o,x,b,draw",
"x,x,o,o,x,b,o,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,x,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,loss",
"x,x,o,o,x,b,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,x,o,b,b,b,b,b,b,b,b,b,b,loss",
"x,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,o,b,b,b,b,o,x,x,o,x,b,draw",
"x,x,o,o,x,b,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,x,b,b,b,b,b,loss",
"x,x,b,b,b,b,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,x,o,x,o,b,win ",
"x,x,o,o,x,b,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,x,o,b,b,b,b,loss",
"x,x,b,b,b,b,o,o,o,x,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,x,b,b,b,b,b,draw",
"x,x,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,x,o,o,x,b,win ",
"x,x,o,x,x,b,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,o,b,b,b,b,b,draw",
"x,o,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,x,x,x,o,b,loss",
"x,x,x,o,o,b,o,b,b,b,b,b,x,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,win ",
"x,x,o,x,b,b,o,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,x,b,b,b,b,b,loss",
"x,x,o,x,b,b,o,o,x,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,loss",
"x,x,o,x,x,o,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,loss",
"x,x,x,o,b,b,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,x,o,b,b,b,b,b,b,b,b,b,loss",
"x,b,b,b,b,b,o,o,x,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,x,x,o,b,b,loss",
"x,x,b,b,b,b,o,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,x,x,o,b,b,b,loss",
"x,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,o,x,o,x,o,x,draw",
"x,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,o,x,x,o,x,b,win ",
"x,x,b,b,b,b,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,x,o,b,b,b,o,x,b,b,b,b,draw",
"x,x,o,x,o,b,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,x,b,b,b,b,loss",
"x,o,b,b,b,b,o,o,x,o,x,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,x,b,b,b,b,b,loss",
"x,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,b,b,b,b,b,b,o,x,o,x,o,x,draw",
"x,x,b,b,b,b,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,o,x,o,x,b,b,win ",
"x,o,x,o,b,b,b,b,b,b,b,b,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,x,x,b,b,b,draw",
"x,x,x,o,b,b,o,o,o,x,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,loss",
"x,x,o,b,b,b,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,x,o,x,b,b,b,loss",
"x,x,o,x,b,b,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,x,o,b,b,b,loss",
"x,x,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,o,x,o,x,b,draw",
"x,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,x,o,x,o,x,draw",
"x,x,o,x,x,b,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,o,b,b,b,b,loss",
"x,o,b,b,b,b,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,x,x,x,o,b,draw",
"x,o,o,x,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,x,o,x,b,b,b,draw",
"x,x,b,b,b,b,o,o,x,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,o,b,b,b,b,x,b,b,b,b,b,win ",
"x,o,o,x,x,b,o,b,b,b,b,b,x,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,win ",
"x,b,b,b,b,b,o,o,x,o,x,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,x,o,b,b,b,b,loss",
"x,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,x,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,x,x,o,o,b,win ",
"x,x,o,x,b,b,o,x,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,o,b,b,b,b,b,win ",
"x,x,o,b,b,b,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,o,x,x,b,b,b,draw",
"x,o,x,b,b,b,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,x,o,x,o,b,b,loss",
"x,b,b,b,b,b,o,o,x,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,x,o,x,o,b,b,loss",
"x,o,x,x,o,b,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,x,b,b,b,b,draw",
"x,x,o,b,b,b,o,o,x,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,x,b,b,b,b,loss",
"x,x,o,o,x,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,x,o,b,b,b,win ",
"x,o,b,b,b,b,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,x,x,o,x,o,b,draw",
"x,x,x,o,b,b,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,o,b,b,b,b,x,b,b,b,b,b,loss",
"x,b,b,b,b,b,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,o,x,o,x,x,b,loss",
"x,x,b,b,b,b,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,x,x,o,o,b,draw",
"x,x,x,o,o,b,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,x,b,b,b,b,b,o,b,b,b,b,b,win ",
"x,x,x,o,b,b,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,o,x,b,b,b,b,loss",
"x,x,o,b,b,b,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,x,o,x,o,b,b,loss",
"x,x,b,b,b,b,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,o,x,x,o,b,b,draw",
"x,x,x,o,b,b,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,x,o,b,b,b,b,draw",
"x,x,x,o,b,b,o,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,x,o,b,b,b,b,draw",
"x,x,b,b,b,b,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,o,x,o,x,b,draw",
"x,x,o,x,o,b,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,x,o,b,b,b,b,draw",
"x,x,o,b,b,b,o,o,o,x,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,x,b,b,b,b,b,draw",
"x,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,o,x,o,x,x,b,draw",
"x,x,x,o,o,b,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,x,b,b,b,b,draw",
"x,o,x,b,b,b,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,x,o,x,b,b,loss",
"x,o,o,x,b,b,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,x,x,o,b,b,b,draw",
"x,x,o,x,o,b,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,x,b,b,b,b,b,loss",
"x,x,o,o,x,b,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,x,b,b,b,b,b,o,b,b,b,b,b,loss",
"x,x,o,b,b,b,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,x,x,o,b,b,draw",
"x,x,b,b,b,b,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,x,o,x,o,b,b,draw",
"x,x,b,b,b,b,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,o,x,o,b,b,x,b,b,b,b,b,loss",
"x,o,o,x,x,b,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,x,o,b,b,b,b,draw",
"x,x,o,b,b,b,o,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,x,x,b,b,b,loss",
"x,x,x,o,b,b,o,o,x,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,loss",
"x,o,x,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,x,o,x,b,b,draw",
"x,x,x,o,b,b,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,x,o,b,b,b,draw",
"x,x,x,o,b,b,o,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,x,b,b,b,b,loss",
"x,x,o,o,x,b,o,b,b,b,b,b,x,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,draw",
"x,x,o,x,b,b,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,o,b,b,b,b,x,b,b,b,b,b,loss",
"x,b,b,b,b,b,o,x,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,x,o,x,b,b,loss",
"x,x,o,x,b,b,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,o,x,b,b,b,b,loss",
"x,o,o,x,b,b,o,x,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,x,b,b,b,b,b,draw",
"x,o,b,b,b,b,o,x,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,x,x,o,b,b,b,draw",
"x,x,o,b,b,b,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,x,o,x,b,b,loss",
"x,x,o,o,x,b,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,x,b,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,loss",
"x,o,b,b,b,b,o,o,x,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,x,x,b,b,b,loss",
"x,x,o,x,b,b,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,x,o,b,b,b,b,loss",
"x,b,b,b,b,b,o,x,o,x,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,x,o,b,b,b,draw",
"x,x,o,o,x,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,x,b,b,b,b,b,o,b,b,b,b,b,o,b,b,b,b,b,draw",
"x,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,o,x,x,x,o,b,draw",
"x,x,o,x,b,b,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,o,x,b,b,b,loss",
"x,b,b,b,b,b,o,o,x,o,x,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,b,b,b,b,b,x,b,b,b,b,b,loss",
"x,x,o,b,b,b,o,o,x,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,x,o,b,b,b,b,loss",
"x,x,b,b,b,b,o,x,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,o,x,b,b,b,loss",
"x,x,b,b,b,b,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,x,o,o,x,b,draw",
"x,x,b,b,b,b,o,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,x,x,o,b,b,loss",
"x,o,b,b,b,b,o,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,o,x,o,x,x,b,draw",
"x,o,o,o,x,b,o,b,b,b,b,b,x,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,b,x,b,b,b,b,b,draw"]

X =[]
Y =[]


#Changed the following to a setupt where human goes first. 050817 1653
#flipped the signs. Now loss=1, win=-1
for each_line in training_data:
    
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
    if class_label=="win ":
        class_label_int =-1
    elif class_label =='loss':
        class_label_int = 1  #signs inverted since human loss = AI win 
    elif class_label =='draw':
        class_label_int = 0
    
    
    
    X.append(board_rep_int)
    Y.append(class_label_int)
    




clf = tree.DecisionTreeClassifier()
clf = clf.fit(X,Y)

def OurBoard2TreeInput_TF(board_grid):
    TreeInputArr=[None]*42
    k=0
    for col in range(7):
        for row in [5,4,3,2,1,0]:
            
            if(board_grid[row][col]==2):
                slot_val = 1   #since ai is o
            elif(board_grid[row][col]==1):
                slot_val =-1
            elif(board_grid[row][col]==0):
                slot_val =0
            
            
            TreeInputArr[k]= slot_val
            k+=1
    
    return TreeInputArr
       
         
                
                        
                                
#################stuff above for decision tree.                                        
###############################################################################                                                                                                                                                             
                                                                                                                                
class GameWithDTreeAI(Connect4Interface.Connect4Game):                      

    def p2_next_move(self,currentBoard):
        
        player = 2
        depth = 4
        
        next_boards_utility={}
        
        for move in range(7):
            if Connect4Interface.is_move_valid(move, currentBoard):
                copy_of_board = [x[:] for x in currentBoard] #creates copy of currentBoard
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
    
    if(player==1):
        opponent = 2
    else:
        opponent =1
        
    next_boards = []  #a list of boards
    
    for move in range(7):
        if Connect4Interface.is_move_valid(move, boardX):
            new_board = [x[:] for x in boardX] #creates copy of currentBoard
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
    
    if(player==1):
        opponent = 2
    else:
        opponent =1
        
    next_boards =[] #a list of boards
    
    for move in range(7):
        if Connect4Interface.is_move_valid(move, boardX):
            new_board = [x[:] for x in boardX] #creates copy of currentBoard
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

    if Connect4Interface.player_won(boardX)==1:
        return -10000 - depth
    else:
        dtree_output = clf.predict([OurBoard2TreeInput_TF(boardX)])
        return dtree_output[0]
    
    
    
print "DTree Module Loaded"
game3 = GameWithDTreeAI()
game3.run_game()
    


    