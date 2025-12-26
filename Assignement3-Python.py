#!/usr/bin/env python
# coding: utf-8

# ## Module 4: Functions & Modules in Python 
# ### Task 1: Calculate Factorial Using a Function 
# 
# 
# Problem Statement: Write a Python program that:
# 1.   Defines a function named factorial that takes a number as an argument and calculates its factorial using a loop or recursion.
# 2.   Returns the calculated factorial.
# 3.   Calls the function with a sample number and prints the output.
#  
# 

# In[19]:


def fact(n):
    if n==1:
        return 1
    else:
        return n*fact(n-1)
    
n=int(input("Enter a number:")) 
print(f"The factorial of {n} is : {fact(n)}")


# In[20]:


def factorial(n):
    x=1
    for i in range(1,n+1):
        x=x*i
    return x
        
    
n=int(input("Enter a number:")) 
print(f"The factorial of {n} is : {factorial(n)}")
    


# 
# ## Task 2: Using the Math Module for Calculations
#  
# Problem Statement: Write a Python program that:
# 1.   Asks the user for a number as input.
# 2.   Uses the math module to calculate the:
# o   Square root of the number
# o   Natural logarithm (log base e) of the number
# o   Sine of the number (in radians)
# 3.   Displays the calculated results.
# 

# In[27]:


import math
k=int(input("Enter a number:")) 
print(f"Square root: {math.sqrt(k)}")
print(f"Logarithm: {math.log(k)}")
print(f"Sine: {math.sin(k)}")


# In[ ]:





# In[ ]:




