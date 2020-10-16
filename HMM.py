#!/usr/bin/env python
# coding: utf-8

# 

# In[9]:


import pandas as pd
import numpy as np
import math
import random


# In[10]:


def loadTextFile():
#     returned datalist contains 2 list, 1st is the states list that represent 1st column and 2nd is the umbrella sited representing the second column
    data=pd.read_csv("D:/550_pattern_recog/proj2/Project2Data - Copy.txt", names=['states','sequence'])
    sequencelist=(data.iloc[:,-1]).tolist();
    statelist=(data.iloc[:,-2]).tolist()
    datalist=[statelist,sequencelist]
    return datalist


# In[11]:


def numb_distinct_states(datalist):
#     calculating the distinct nuber of distinct states
    dat=loadTextFile()
    list1=np.array(dat[0])
    no_ofStates=np.unique(list1)
    no_ofStates.tolist()
    return len(no_ofStates)
    
# print(numb_distinct_states(loadTextFile()))     


# In[12]:


def transition(datalist):
    dat=datalist
#      counting the total number of foggy, sunny and rainy from the data, toget the transition because for ex total number of tranition from
# foggy to some state would be the total number of foggy states present in the the data. same wrt to sunny and rainy
    foggycount=dat[0].count('foggy')
    
    rainycount=dat[0].count('rainy')
     
    sunnycount=dat[0].count('sunny')
#     there would be no tansition from the last state that is present in the the data, so identifying that state and subtracting
# by -1 from that states count
    if(dat[0][-1]=='foggy'):
            foggycount=foggycount-1
    elif(dat[0][-1]=='rainy'):    
            rainycount=rainycount-1
    elif(dat[0][-1]=='sunny'):
            sunnycount=sunnycount-1

# i =1 because i start iterating from the 2nd row in the data .
    i=1
    f2f=0
    f2s=0
    f2r=0
    s2f=0
    s2s=0
    s2r=0
    r2f=0
    r2s=0
    r2r=0
# calculating the transition states . i is the the current state and i-1 is the previous state
    while i<len(dat[0]):
        if(dat[0][i-1]=='foggy' and dat[0][i]=='foggy'):
            f2f +=1
        elif(dat[0][i-1]=='foggy' and dat[0][i]=='sunny'):
            f2s +=1
        elif(dat[0][i-1]=='foggy' and dat[0][i]=='rainy'):
            f2r +=1
        elif(dat[0][i-1]=='sunny' and dat[0][i]=='foggy'):
            s2f +=1
        elif(dat[0][i-1]=='sunny' and dat[0][i]=='sunny'):
            s2s +=1
        elif(dat[0][i-1]=='sunny' and dat[0][i]=='rainy'):
            s2r +=1
        elif(dat[0][i-1]=='rainy' and dat[0][i]=='foggy'):
            r2f +=1
        elif(dat[0][i-1]=='rainy' and dat[0][i]=='sunny'):
            r2s +=1
        elif(dat[0][i-1]=='rainy' and dat[0][i]=='rainy'):
            r2r +=1
        i +=1
#     calculating the probablities and building the transition matrix
    f2fprob=f2f/foggycount
    f2sprob=f2s/foggycount
    f2rprob=f2r/foggycount
    s2fprob=s2f/sunnycount
    s2sprob=s2s/sunnycount
    s2rprob=s2r/sunnycount
    r2fprob=r2f/rainycount
    r2sprob=r2s/rainycount
    r2rprob=r2r/rainycount

    transition_matrix=[]
    foggylist=[f2fprob,f2sprob,f2rprob]
    sunnylist=[s2fprob,s2sprob,s2rprob]
    rainylist=[r2fprob,r2sprob,r2rprob]
    transition_matrix.append(foggylist)
    transition_matrix.append(sunnylist)
    transition_matrix.append(rainylist)
    
    return transition_matrix
   
# print(transition(loadTextFile()))


# In[13]:


def emission(datalist):
    dat=datalist
    i=0
#     calculating the count for foggy to yes , foggy to no, sunny to yes etc.. ex if current iteration has foggy as state and yes as umbrella sited. increment f2yes by1.
    f2yes,f2no,s2yes,s2no,r2yes,r2no=0,0,0,0,0,0
    while i<len(dat[0]):
        if(dat[0][i]=='foggy' and dat[1][i]=='yes'):
            f2yes +=1
        if(dat[0][i]=='foggy' and dat[1][i]=='no'):
            f2no +=1
        if(dat[0][i]=='sunny' and dat[1][i]=='yes'):
            s2yes +=1
        if(dat[0][i]=='sunny' and dat[1][i]=='no'):
            s2no +=1
        if(dat[0][i]=='rainy' and dat[1][i]=='yes'):
            r2yes +=1
        if(dat[0][i]=='rainy' and dat[1][i]=='no'):
            r2no +=1
        i +=1

# building the emission matrix 
    emission_matrix=[]
    foggy_emissionList=[f2yes/(f2yes+f2no),f2no/(f2yes+f2no)]
    sunny_emissionList=[s2yes/(s2yes+s2no),s2no/(s2yes+s2no)]   
    rainy_emissionlist=[r2yes/(r2yes+r2no),r2no/(r2yes+r2no)]

    emission_matrix.append(foggy_emissionList)
    emission_matrix.append(sunny_emissionList)
    emission_matrix.append(rainy_emissionlist)
    
    return emission_matrix

# print(emission(loadTextFile()))


# In[14]:


def viterbi(transitionMatrix,emissionMatrix,userSequence,noOfDistinctStates):
    
# the function takes as input transition matrix, emission matrix, no. distinct states and input sequence
    seq=userSequence
    transition_matrix=transitionMatrix
    emission_matrix=emissionMatrix
    noOfUniqueStates=noOfDistinctStates
# I am taking 3 list,maxVal_and_BacktrackVal will be inside values_perInstance,values_perInstance will be inside viterbi_list
# my Viterbi_list will look like - 
# [[[0.10466842305689616, 1], [0.7275295431588802, 1], [0.01028273378316844, 1]], [[0.07614937000974406, 1], [0.5292992361689689, 1], [0.007480992611692918, 1]]]
#   maxval for foggy,prevstate|maxval for sunny,previousstate......so on
# 
    viterbi_list=[]
    values_perInstance=[]
    maxVal_and_BacktrackVal=[]
    

# 3 is noof states, in our case i= 0 is foggy, i=1 is sunny,i=2 is rainy


  # if first value in seq is yes then transition_matrix[1][i]*emission_matrix[i][0],
#     where 1 represents the start state i.e sunny and i is all the three states so it takes values from transition matrix wrt to sunny to foffgy, sunny to sunny and sunny to rainy
# the 0 in emission_matrix[i][0] represents the input sequence yes, and 1 if input seq is no
    
    i=0
    while i<noOfUniqueStates:
        if(seq[0]=='yes'):
            maxVal_and_BacktrackVal.append(transition_matrix[1][i]*emission_matrix[i][0])
            maxVal_and_BacktrackVal.append(1)
            values_perInstance.append(maxVal_and_BacktrackVal)
            maxVal_and_BacktrackVal=[]
        else:
            maxVal_and_BacktrackVal.append(transition_matrix[1][i]*emission_matrix[i][1])
            maxVal_and_BacktrackVal.append(1)
            values_perInstance.append(maxVal_and_BacktrackVal)
            maxVal_and_BacktrackVal=[]
        i +=1 

         

#     appending the first instance
      
    viterbi_list.append(values_perInstance)
# emtying the values_perInstance list for reuse
    values_perInstance=[]
# the outer most loopis for the input sequence,the lopp inside it isto calculate values for the current instance that we want to calculate wrt states.
# the innermost loop is for iterating through states in the previous instance.
    i=1
    while i<len(seq):
        if(seq[i]=='yes'):
            k=0
            while k<noOfUniqueStates:
                max=0
                backtrackval=0
                j=0
                while j<noOfUniqueStates:
                    current_statevalue=viterbi_list[-1][j][0]*transition_matrix[j][k]*emission_matrix[k][0]
                    if(max<(current_statevalue)):
                        max=current_statevalue
                        backtrackval=j
                    j +=1
                maxVal_and_BacktrackVal.append(max)
                maxVal_and_BacktrackVal.append(backtrackval)
                values_perInstance.append(maxVal_and_BacktrackVal)
                maxVal_and_BacktrackVal=[]
                k+=1
        else:
            k=0
            while k<noOfUniqueStates:
                max=0
                backtrackval=0
                j=0
                while j<noOfUniqueStates:
                    current_statevalue=viterbi_list[-1][j][0]*transition_matrix[j][k]*emission_matrix[k][1]
                    if(max<current_statevalue):
                        max=current_statevalue
                        backtrackval=j
                    j +=1
                maxVal_and_BacktrackVal.append(max)
                maxVal_and_BacktrackVal.append(backtrackval)    
                values_perInstance.append(maxVal_and_BacktrackVal)
                maxVal_and_BacktrackVal=[]
                k+=1
        viterbi_list.append(values_perInstance)
        values_perInstance=[]
        i+=1
    
    
    return viterbi_list

# print(viterbi(transition(loadTextFile()),emission(loadTextFile()),['yes','no','yes'],numb_distinct_states(loadTextFile())))


# In[15]:


def decoding(viterbilist):
    resultlist=[]
    viterbi_list=viterbilist
    
    i=len(viterbi_list)-1
    prevstate=0
    current=0
    max=0
# considering the last instance which represents the instance values for the last sequenvce from the input , I am identifying the
# max from the last instance and backtracking according the its corresponding previous state and appending it in the result list.
    while i>0:
        if(i==len(viterbi_list)-1):
            itr=0
            while itr<3:
                if(max< viterbi_list[-1][itr][0]):
                    max=viterbi_list[-1][itr][0]
                    current=itr
                    prevstate=viterbi_list[-1][itr][1]
                itr +=1    
        
            resultlist.append(current)
            resultlist.append(prevstate)
        else:
            prevstate=viterbi_list[i][prevstate][1]
            resultlist.append(prevstate)
        i -=1
# reversing the resultlist        
    resultlist.reverse()
    
    
# resultlist is    
    index=0
    while index<len(resultlist):
        if(resultlist[index]==0):
                resultlist[index]="foggy"
        elif(resultlist[index]==1):
            resultlist[index]="sunny"
        else:
            resultlist[index]="rainy"
        index +=1
    return resultlist    
        


# l=viterbi(transition(loadTextFile()),emission(loadTextFile()),['yes','no','yes'],numb_distinct_states(loadTextFile()))
# print(decoding(l))


# In[16]:


def main():
    mySequence=['no', 'no', 'no', 'yes', 'no', 'no', 'yes', 'yes', 'no', 'yes']
    viterbiOutput=viterbi(transition(loadTextFile()),emission(loadTextFile()),mySequence,numb_distinct_states(loadTextFile()))
#     print(transition(loadTextFile()))
#     print(emission(loadTextFile()) )
    transmat=transition(loadTextFile())
    emissionmat=emission(loadTextFile())
    foggyrow=transmat[0]
    sunnyrow=transmat[1]
    rainyrow=transmat[2]
    
    
    
    print("for sequence - ",mySequence)
    print()
    print()
    print("Transition Matrix :-")
    print("------------------------------------------------------------------------------------------")
    print("              foggy                 sunny                rainy")
    print()
    print("foggy",foggyrow)
    print("sunny",sunnyrow)
    print("rainy",rainyrow)
    print()
    
    foggyrow=[]
    sunnyrow=[]
    rainyrow=[]
    
    foggyrow=emissionmat[0]
    sunnyrow=emissionmat[1]
    rainyrow=emissionmat[2]
    print()
    print("Emission Martrix :-")
    print("------------------------------------------------------------------------------------------")
    print()
    print("              YES                 No")
    print()
    print("foggy",foggyrow)
    print("sunny",sunnyrow)
    print("rainy",rainyrow)
    print()

    
    print()
    print("Result :-")
    print("------------------------------------------------------------------------------------------")
    print(decoding(viterbiOutput))
main()    
    


# In[ ]:





# In[ ]:




