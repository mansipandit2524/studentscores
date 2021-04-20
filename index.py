#!/usr/bin/env python
# coding: utf-8

# In[11]:


import numpy as np 
import matplotlib.pyplot as plt 

def estimate_coef(x, y): 
	# number of observations/points 
	n = np.size(x) 

	# mean of x and y vector 
	m_x, m_y = np.mean(x), np.mean(y) 

	# calculating cross-deviation and deviation about x 
	SS_xy = np.sum(y*x) - n*m_y*m_x 
	SS_xx = np.sum(x*x) - n*m_x*m_x 

	# calculating regression coefficients 
	b_1 = SS_xy / SS_xx 
	b_0 = m_y - b_1*m_x 

	return(b_0, b_1) 

def plot_regression_line(x, y, b): 
	# plotting the actual points as scatter plot 
	plt.scatter(x, y, color = "m", 
			marker = "o", s = 30) 

	# predicted response vector 
	y_pred = b[0] + b[1]*x 

	# plotting the regression line 
	plt.plot(x, y_pred, color = "g") 

	# putting labels 
	plt.xlabel('x') 
	plt.ylabel('y') 

	# function to show plot 
	plt.show() 

def main(): 
	# observations 
	x = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8, 9]) 
	y = np.array([1, 3, 2, 5, 7, 8, 8, 9, 10, 12]) 

	# estimating coefficients 
	b = estimate_coef(x, y) 
	print("Estimated coefficients:\nb_0 = {} \\nb_1 = {}".format(b[0], b[1])) 

	# plotting regression line 
	plot_regression_line(x, y, b) 

if __name__ == "__main__": 
	main() 


# In[5]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import linear_model


# In[7]:


data = pd.read_csv("D:\mansi\Downloads\FuelConsumptionCo2.csv")


# In[8]:


data.head()


# In[9]:


data = data[["ENGINESIZE","CO2EMISSIONS"]]


# In[10]:


plt.scatter(data["ENGINESIZE"],data["CO2EMISSIONS"],color="blue")
plt.xlabel("ENGINESIZE")
plt.ylabel("CO2EMISSIONS")
plt.show()


# In[11]:


train = data[:(int((len(data)*0.8)))]
test = data[(int((len(data)*0.8))):]


# In[13]:


from sklearn import linear_model
regr = linear_model.LinearRegression()
train_x = np.array(train[["ENGINESIZE"]])
train_y = np.array(train[["CO2EMISSIONS"]])
regr.fit(train_x,train_y)
print("coffecients : ",regr.coef_)
print("Intercept : ",regr.intercept_)


# In[16]:


plt.scatter(train["ENGINESIZE"],train["CO2EMISSIONS"],color="blue")
plt.plot(train_x,regr.coef_*train_x + regr.intercept_,'-r')
plt.xlabel("Engine size")
plt.ylabel("Emission")


# In[15]:


def get_regression_prediction(input_features,intercept,slope):
    predicted_values = input_features*slope + intercept
    
    return predicted_values


# In[17]:


my_engine_size = 3.5
estimated_emission = get_regression_prediction(my_engine_size,regr.intercept_[0],regr.coef_[0][0])
print("Estimated Emission : ",estimated_emission)


# In[19]:


from sklearn.metrics import r2_score
test_x = np.array(test[['ENGINESIZE']])
test_y = np.array(test[['CO2EMISSIONS']])
test_y_ = regr.predict(test_x)
print("Mean absolute error : %.2f" %np.mean(np.absolute(test_y_-test_y)))
print("Mean sum of squares (MSE): %.2f" %np.mean((test_y_-test_y)**2))
print("R2-score : %.2f" %r2_score(test_y_,test_y))


# In[ ]:




