"""
1.Two Sum
Given an array of integers, return indices of the two numbers such that they add up to a specific target.
You may assume that each input would have exactly one solution.
Given nums = [2, 7, 11, 15], target = 9,
Because nums[0] + nums[1] = 2 + 7 = 9,
return [0, 1].


This is a demo task.

Write a function:

def solution(A)

that, given an array A of N integers, returns the smallest positive integer (greater than 0) that does not occur in A.

For example, given A = [1, 3, 6, 4, 1, 2], the function should return 5.

Given A = [1, 2, 3], the function should return 4.

Given A = [−1, −3], the function should return 1.

Write an efficient algorithm for the following assumptions:

N is an integer within the range [1..100,000];
each element of array A is an integer within the range [−1,000,000..1,000,000].
"""

# you can write to stdout for debugging purposes, e.g.
# print("this is a debug message")
import pandas as pd
import numpy as np


def solution(A):

# firs step is to test is all are negative, if true, return 1 as result
    all = True

    for a in A:
        if a >= 1:
            all = False
            break
    if all == True:
        result = 1
        return result
                        
    # get range and compute missing value
    a_in_order= range(1, max(A))
    not_in_a = np.isin(A, a_in_order, invert=True)
    print(A[not_in_a])
    result = min(A[not_in_a])    
    return result
    

    print(solution([1, 3, 6, 4, 1, 2]))

test = [1, 3, 6, 4, 1, 2]


a_in_order= np.arange(1, max(test))
not_in_a = np.isin(test, a_in_order, invert=True)
not_in_a

min(test[not_in_a])    

a_in_order
np.arange(1, max(test))