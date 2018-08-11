#!/usr/bin/env python3

#################################
#                               #
#   Examples of sorting algos   #
#                               #
#################################

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

# https://en.wikipedia.org/wiki/Heapsort
def heapsort(x):
    heapify(x,len(x))
    end = len(x) - 1
    while end > 0:
        x[end], x[0] = x[0], x[end]
        end -= 1
        sift_down(x, 0, end)
    return x

def heapify(x, count):
    start = int((count-2)/2)
    while start >= 0:
        sift_down(x, start, count - 1)
        start -= 1

def sift_down(x, start, end):
    root = start
    while (root * 2 + 1) <= end:
        child = root * 2 + 1
        swap = root
        if x[swap] < x[child]:
            swap = child
        if (child + 1) <= end and x[swap] < x[child+1]:
            swap = child + 1
        if swap != root:
            x[root], x[swap] = x[swap], x[root]
            root = swap
        else:
            return

ary = [1,0,5,3,2,9,6,7,8]
print("starting array..")
print(ary)

print("Buble Sort", bubbleSort(ary))
print("Selection Sort", selectionSort(ary))
print("Insertion Sort", insertionSort(ary))
print("Shell Sort", shellSort(ary))
print("Heap Sort", heapsort(ary))