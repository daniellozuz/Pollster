# -*- coding: utf-8 -*-
"""
Created on Fri May 25 11:22:28 2018

@author: OMEN
"""

import numpy as np
import os
import pandas as pd
from operator import itemgetter


def has_duplicates(listObj):
    return len(listObj) != len(set(listObj))

def CreateColumnNames(file):
    #creating columns, in csv seperated by comas
    column_names = "Code, " + "".join(["Question " + str(i) + ", " for i in range(1,51)]) + "\n"
    file.write(column_names)
    
def GetCode(code):
    
    #getting list of only marked boxes
    marked = list(filter(lambda x: x[:][5] == 1, code))
    


    #getting the code
    name = ""
    
    for i in range(0,len(marked)):
        #checing if the code field was filled correctly
        if marked[i][0] != i:
            name += str(marked[i][1]) 
        else:
            return "Wrong Code!,"
    return name + ","
   
    
def GetAnswers(points):
    
    answers =  [[] for i in range(1,51)]
    #getting list of only marked boxes
    marked = list(filter(lambda x: x[:][5] == 1, points))
    #sorting by the questions
    marked = sorted(marked, key = itemgetter(0))
    
    #name is string containing answers
    #flag is used for preventing from saving error in multiple cells for one question
    idx = 0
    #getting the asnwers
    for i in range(0,len(marked)):
        idx = marked[i][0]
        answers[idx - 1].append(marked[i][1])
    return answers
        
  
#    
def ValidateAnswers(answers):
    
    validated = [[] for i in range(1,51)]
    for i in range(0,len(answers)):
        if len(answers[i]) > 1:
            validated[i] = 'Error'
        elif len(answers[i]) ==  0:
            validated[i] = "Empty, "
        else:
#                 converting index of filled box to letter indicating answer in form
            if answers[i][0] == 1:
                validated[i] = "A,"
            elif answers[i][0] == 2:
                validated[i] = "B,"
            elif answers[i][0] == 3:
                validated[i] = "C,"
            elif answers[i][0] == 4:
                validated[i] = "D,"
            elif answers[i][0] == 5:
                validated[i] = "E, "
                 
    #getting whole list into single string 
    return ''.join(str(r) for v in validated for r in v).replace(" ", "")
    
#main function saving data, the only one you need to use from this file
def SaveData(code, points):

    with open("out.csv", 'a') as csv:  
        #if file is empty initialize columns
        if os.path.getsize("out.csv") < 100:
            CreateColumnNames(csv)
        #read code from filled boxes
        form_data = GetCode(code)
        answers = GetAnswers(points)
        form_data += ValidateAnswers(answers)[0:-1] + "\n"
        csv.write(form_data)

        
    csv.close()
#To test SaveData function run final_detection.py, then uncommnet line below
#SaveData(kody,punkty)

        
      
