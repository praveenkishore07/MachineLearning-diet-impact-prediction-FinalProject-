#!/usr/bin/env python
# coding: utf-8

# In[2]:


import numpy as np
import pandas as pd


# In[3]:


np.random.seed(42)
n_samples = 500

age = np.random.randint(4, 11, n_samples)
calories = np.random.normal(1800, 250, n_samples)  
diet_type = np.random.choice([0, 1], n_samples)  

growth_rate = (
    4.0 + 
    (age * 0.05) + 
    (calories * 0.0005) + 
    (diet_type * 0.8) + 
    np.random.normal(0, 0.2, n_samples)
)


df = pd.DataFrame({
    'Child_ID': range(1, n_samples + 1),
    'Age_Years': age,
    'Daily_Calorie_Intake': calories.astype(int),
    'Diet_Type': diet_type,  
    'Growth_Rate_cm': np.round(growth_rate, 2)
})

df.to_csv('child_growth_data.csv', index=False)


# In[4]:


print(df)


# In[ ]:




