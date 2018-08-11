#################################
#                               #
#   Examples of sorting algos   #
#                               #
#################################

import numpy as np

def bubbleSort(x):
    for i in range(0, len(x) - 1):
        for j in range(0, len(x) - 1 - i):    
            if x[j] > x[j+1]:
                x[j], x[j+1] = x[j+1], x[j]
    return x

def selectionSort(x):
    b = []
    markers = []
    toMark = 0
    for i in range(0, len(x)):
        minV = None
        for j in range(0, len(x)):
            if (minV is None or x[j] < minV) and j not in markers:
                minV = x[j]
                toMark = j
        markers.append(toMark)
        b.append(minV)
    return b

def insertionSort(x):
    for i in range(0, len(x)):
        sortLen = i
        for j in range(0, sortLen):
            if x[sortLen-j] < x[sortLen-j-1]:
                x[sortLen-j], x[sortLen-j-1] = x[sortLen-j-1], x[sortLen-j]
            else:
                continue
        print(x)
    return x

def shellSort(x):
    sep = len(x)
    for i in range(0, len(x)):
        sep = round(sep / 2)
        if not sep: continue
        for j in range(0, len(x) - sep):
            if x[j] > x[j+sep]:
                x[j], x[j+sep] = x[j+sep], x[j]
    return x

def heapsort(x):
    # no

ary = [1,0,5,3,2,9,6,7,8]
print("starting array")
print(ary)

#print(bubbleSort(ary))
#print(selectionSort(ary))
#print(insertionSort(ary))
print(shellSort(ary))