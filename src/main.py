import numpy as np
from a_star import *


def plan():

    src, goal, length, clearance, radius = user_input()

    if(length == None):
        print("-------------------------------------")
        print("Invalid input, try again")
        print("-------------------------------------")
        return

    A_star(src, goal, length, clearance, radius)

    
if __name__ == "__main__":

    plan()
    