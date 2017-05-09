#CS 156 Spring 2017
#Connect 4 Project 
#05-09-17

import Connect4Interface
#import random
import copy

from sklearn import tree #need to install Scikit Learn to import this module

##Below training data an excerpt from http://archive.ics.uci.edu/ml/datasets/Connect-4
##It is a small subset of the 67557 data points available on the website.
##See the website and our project report/presentation for details of data transformation.
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
#flipped the signs. Now loss=1, win=-1 (because in our game, human starts first)
#AI "win" necessarily mean human "loss"
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
        

def heuristic_function(boardX,player,depth):
    """New heuristic_function(), simply call uses the Decision Tree Classifier
    Model trained in the begining of this code to predict the theoretical outcome
    of the board win=+1, loss = -1, draw = 0."""

    if Connect4Interface.player_won(boardX)==1:
        return -10000 - depth #essentially assigning it to negative infinity.
    else:
        dtree_output = clf.predict([OurBoard2TreeInput_TF(boardX)])
        return dtree_output[0]
    
            
print "DTree Module Loaded"
game3 = GameWithDTreeAI()
game3.run_game()
    


    