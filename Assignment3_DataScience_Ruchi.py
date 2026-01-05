#!/usr/bin/env python
# coding: utf-8

# ### 1. Write a Python program to simulate the following scenarios:  a. Tossing a coin 10,000 times and calculating the experimental probability of heads and tails.  
# 

# In[89]:


import random 
from fractions import Fraction 
outcomes=random.choices(["H","T"],k=10000)
print("Number of heads is",outcomes.count("H"))
print("Number of tails is",outcomes.count("T"))
print("Probability of heads is",Fraction(outcomes.count("H"),10000))
print("Probability of tails is",Fraction(outcomes.count("T"),10000))


# b. Rolling two dice and computing the probability of getting a sum of 7.  
#   Steps  
#       a. Use Python's random module for simulations.  
#       b. Implement loops for repeated trials.  
#       c. Track outcomes and compute probabilities.  
#       
# 

# In[90]:


from itertools import product
x=[1,2,3,4,5,6]
outcomes=list(product(x,repeat=2))
f_outcomes=list(out for out in outcomes if sum(out)==7)
print("Probability of getting a sum of 7 is",Fraction(len(f_outcomes),len(outcomes)))


# ### 2. Write a function to estimate the probability of getting at least one "6" in 10 rolls of a fair die.  
#   Steps  
#       a. Simulate rolling a die 10 times using a loop.  
#       b. Track trials where at least one "6" occurs.  
#       c. Calculate the proportion of successful trials.
# 

# In[91]:


outcomes_1=list(product(x,repeat=4))
f_outcomes=list(out for out in outcomes_1 if all(x != 6 for x in out))
print(" Probability of getting at least one 6:",1-Fraction(len(f_outcomes),len(outcomes_1)))


# ### 3. A bag contains 5 red, 7 green, and 8 blue balls. A ball is drawn randomly, its color noted, and it is put back into the bag. If this process is repeated 1000 times, write a Python program to estimate:  
#   a. The probability of drawing a red ball given that the previous ball was blue.  
#   b. Verify Bayes' theorem with the simulation results.  
# 
# 
# Steps  
#     a. Use random sampling to simulate the process.  
#     b. Compute conditional probabilities directly from the data.  
# 

# In[101]:


import random
colors = ['Red', 'Green', 'Blue']
probabilities = [5/20, 7/20, 8/20]
draws = random.choices(colors, probabilities, k=1000)
count_blue = 0
count_red_after_blue = 0

for i in range(1, len(draws)):
    if draws[i-1] == 'Blue':
        count_blue += 1
        if draws[i] == 'Red':
            count_red_after_blue += 1

prob_red_given_blue = count_red_after_blue / count_blue

print("Estimated P(Red | Previous was Blue):", prob_red_given_blue)


# ### Random Variables and Discrete Probability
# 
# 
# 4. Generate a sample of size 1000 from a discrete random variable with the following distribution:  
#   - P(X=1) = 0.25  
#   - P(X=2) = 0.35  
#   - P(X=3) = 0.4  
#   Compute the empirical mean, variance, and standard deviation of the sample.  
#   Steps  
#       a. Use numpy.random.choice() to generate the sample.  
#       b. Use numpy methods to calculate mean, variance, and standard deviation.
# 

# In[93]:


import numpy as np
sample=np.random.choice([1,2,3],size=5,p=[0.25,0.35,0.4]) 
np.mean(sample)


# In[94]:


np.std(sample)


# In[95]:


np.var(sample)


# ## 5. Simulate 2000 random samples from an exponential distribution with a mean of 5. Visualize the distribution using:  
#   a. A histogram.  
#   b. A probability density function (PDF) overlay.  
#   Steps  
#       a. Use numpy.random.exponential().  
#       b. Use matplotlib to create visualizations.  
# 

# In[96]:


mean=5
y=np.random.exponential(scale=mean,size=2000)
import matplotlib.pyplot as plt
plt.hist(y,bins=12)
plt.show()


# In[97]:


pdf=mean*np.exp(y/mean)
plt.scatter(y,pdf)


# In[98]:


import numpy as np
import matplotlib.pyplot as plt

mean = 5
samples = np.random.exponential(scale=mean, size=2000)

plt.hist(samples, bins=30, density=True)

x = np.linspace(0, max(samples), 200)
pdf = (1/mean) * np.exp(-x/mean)
plt.plot(x, pdf)

plt.xlabel("Value")
plt.ylabel("Density")
plt.title("Exponential Distribution (Mean = 5)")
plt.show()


# ### Central Limit Theorem
# 
# 
# 6. Simulate the Central Limit Theorem by following these steps  
#   a. Generate 10,000 random numbers from a uniform distribution.  
#   b. Draw 1000 samples of size n = 30.  
#   c. Calculate and visualize the distribution of sample means.  
#   Steps  
#       a. Use numpy.random.uniform().  
#       b. Plot both the uniform distribution and the sample mean distribution for comparison.
# 

# In[102]:


import numpy as np
import matplotlib.pyplot as plt

uniform_data = np.random.uniform(0, 1, 10000)


# In[103]:



sample_size = 30
num_samples = 1000

sample_means = [
    np.mean(np.random.choice(uniform_data, size=sample_size))
    for _ in range(num_samples)
]


# In[106]:


plt.figure(figsize=(12, 5))

# Plot uniform distribution
plt.subplot(1, 2, 1)
plt.hist(uniform_data, bins=30)
plt.title("Uniform Distribution (0,1)")
plt.xlabel("Value")
plt.ylabel("Frequency")

# Plot distribution of sample means
plt.subplot(1, 2, 2)
plt.hist(sample_means, bins=30)
plt.title("Distribution of Sample Means (n = 30)")
plt.xlabel("Sample Mean")
plt.ylabel("Frequency")

plt.show()


# In[ ]:




