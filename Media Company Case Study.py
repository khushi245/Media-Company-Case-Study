#!/usr/bin/env python
# coding: utf-8

# # Media Company Case Study
# 

# Problem Statement: A digital media company (similar to Voot, Hotstar, Netflix, etc.) had launched a show. Initially, the show got a good response, but then witnessed a decline in viewership. The company wants to figure out what went wrong.

# In[102]:


#Importing the packages
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import statsmodels.api as sm
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")
from sklearn.preprocessing import MinMaxScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.utils import resample
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold


# In[2]:


#Importing the dataset
df=pd.read_csv("mediacompany.csv")


# ### Data Preprocessing

# In[3]:


#Inspecting the Dataset
df.head()


# In[4]:


# Deleting the unnamed column
df=df.drop(["Unnamed: 7"],axis=1)


# In[5]:


#Checking the first five rows of the dataset
df.head()


# In[6]:


#Checking the datatypes of the variables
df.info()


# In[7]:


#Converting the Date from object to date format
df["Date"]=pd.to_datetime(df["Date"])


# In[8]:


#Checking the Date Column
df["Date"].head()


# ### Deriving Metrics

# In[9]:


#Making a column that shows the episode number
start=datetime(2017,2,28)
df["Days_Premier"]=df.Date-start


# In[10]:


#Checking the Days_Premier column
df.Days_Premier.head()


# In[11]:


#Converting its datatype to string 
df.Days_Premier=df.Days_Premier.astype(str)
df.Days_Premier=df.Days_Premier.apply(lambda x:x.split(" ")[0])


# In[12]:


#Checking the Dataset
df.head()


# In[13]:


#Assuming that the number of views in the today's episode are dependent on the number of views in the yesterday's episode and creating the variable lag view that shows the views in the last day
df["Lag_Views"]=np.roll(df.Views_show,1)


# In[14]:


df.head()


# In[15]:


#Replacing the first value in Lag_Days by 0
df.Lag_Views[0]=0


# In[16]:


#Checking the head of the Dataset again
df.head()


# In[17]:


#Making a column that specifies the day
df["Days"]=df.Date.dt.day_name()


# In[18]:


#Making a column that specifies whether or not it's a weekend
weekday=["Monday","Tuesday",'Wednesday', 'Thursday', 'Friday']
weekend=['Saturday', 'Sunday']
df["Weekend"]=df.Days.apply(lambda x:1 if x in weekend else 0)
#Numberic the week starting from the premier date
df.Days_Premier=df.Days_Premier.astype("int64")
df["Week"]=df["Days_Premier"].apply(lambda x:x//7+1)


# In[19]:


#Checking the dataset
df.head()


# In[20]:


#Checking for missing data
df.isnull().sum()


# There is no missing data in the dataset

# In[21]:


#Checking for outliers for some columns
numerical=['Views_show', 'Visitors', 'Views_platform', 'Ad_impression']
for i in numerical:
    plt.figure(figsize=[10,4])
    plt.subplot(1,2,1)
    sns.boxplot(df[i])
    plt.subplot(1,2,2)
    sns.distplot(df[i],kde=False)
    plt.tight_layout()
    plt.show()


# In[22]:


#Checking some statistics of the data
df[numerical].describe()


# The dataset does not appear to have major problem with outliers and the mean and median also don't have a major difference thus we can go ahead with the analyis of the data

# In[23]:


#We'll also use min max scaling on the numercial variables and storing it in a different dataset
numerical=['Views_show', 'Visitors', 'Views_platform', 'Ad_impression',"Lag_Views"]
scaler=MinMaxScaler()
df1=df.copy()
df1[numerical]=scaler.fit_transform(df1[numerical])
df1.describe()


# ## Exploratory Data Analysis

# Our target variable is Views_Show

# In[24]:


plt.figure(figsize=[10,5])
df.Views_show.plot.line()
plt.ylabel("Number of views\n")
plt.xlabel("\nEpisodes")
plt.title("Views Trend\n\n")
plt.show()


# As we see, there are spikes in the views that appear every week, probably on the weekends and overall the views decreased after about 50th episodes

# In[72]:


#Checking the overall distribution of the target variable
sns.distplot(df.Views_show)
plt.show()


# In[70]:


get_ipython().run_line_magic('pinfo', 'sns.distplot')


# In[26]:


#Checking the trends for other numerical variables
for i in numerical[1:]:
    plt.figure(figsize=[10,5])
    df[i].plot.line()
    plt.ylabel("{}\n".format(i))
    plt.xlabel("\nEpisodes")
    plt.show()


# As we see, the number of vistors and number of views on the platform has somewhat similar trend, very similar to that of number of views on the show, it decreases after about 50th episode. While we see that add impressions peaked at about 30th episode and constantly decreased. 

# In[27]:


plt.figure(figsize=[10,6])
df1.Views_show.plot.line(label="Show Views")
df1.Visitors.plot.line(label="Visitors")
plt.legend()
plt.show()


# In[28]:


plt.figure(figsize=[10,6])
df1.Views_show.plot.line(label="Show Views")
df1.Views_platform.plot.line(label="Platform Views")
plt.legend()
plt.show()


# In[29]:


plt.figure(figsize=[10,6])
df1.Views_show.plot.line(label="Show Views")
df1.Ad_impression.plot.line(label="Add Impressions")
plt.legend()
plt.show()


# As we see from the above graphs, Visitors is the variable that follows the views distribution the most. (Remember, the data is scaled here.)

# In[30]:


sns.pairplot(df[numerical])
plt.show()


# As we see, views on the show is somewhat related to Visitors, Platform views and Add impressions. Also, as we see, the variables other than target variable also appear to be highly correlated.

# The highest correlation the views has is with Add impressions and Weekends.

# Let's see which days of the week have the highest average views

# In[31]:


pd.pivot_table(df,index="Days",values="Views_show").sort_values(by="Views_show",ascending=False).plot.bar()
plt.show()


# The mean views are maximum for Sunday followed by Monday and Saturday. Monday might have more views because of the lag_views variable, larger number of people watch the show on Sunday and they might continue to watch it on Monday

# In[32]:


pd.pivot_table(df,index="Weekend",values="Views_show").plot.bar()
plt.show()


# Weekend have much higher mean views.

# In[33]:


df.groupby(by="Character_A").mean()["Views_show"].plot.bar()
plt.show()


# The mean views are significantly greater when the character A was present in the show.

# In[34]:


df.groupby(by="Cricket_match_india").mean()["Views_show"].plot.bar()
plt.show()


# The difference isn't significant and hence cricket isn't an important variable affecting the views on the shows

# In[35]:


df1.groupby(by="Week").mean()["Views_show"].plot.line()
plt.show()


# The mean views peaked in the 5th week of the premier and then started to decline

# In[36]:


#Plotting a heatmap for correlation coeffs
plt.figure(figsize=[10,8])
sns.heatmap(df.corr(),annot=True,cmap="YlGnBu")
plt.show()


# The highest correlation the views has is with Add impressions and Weekends.

# Now that we have a view about the data and relationships between the variables, we'll start fitting the models and try to find the best fit model.

# ## Model Fitting

# In[37]:


numerical=list(df1.select_dtypes("float").columns)+list(df1.select_dtypes("int64").columns)


# In[38]:


df1=df1[numerical]


# In[39]:


df1.head()


# In[40]:


df1.columns


# In[42]:


#For a rough idea, fitting a model with all the variables first
y=df1["Views_show"]
X=sm.add_constant(df1[['Visitors', 'Views_platform', 'Ad_impression', 'Lag_Views',
       'Cricket_match_india', 'Character_A', 'Days_Premier', 'Weekend', 'Week']])
model1=sm.OLS(y,X).fit()
model1.summary()


# In[43]:


#Checking the VIF values of the model
vif = pd.DataFrame()
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# #### The vif values are dangerously high for the model. Let's try building our model from scratch. We will  start with Views_platform

# In[44]:


X=sm.add_constant(df1["Views_platform"])
model2=sm.OLS(y,X).fit()
model2.summary()


# 36% of the variation in the views of the shows can be explained by the views on the platform. Adding ad_impression in the model and let's see the difference.

# In[45]:


X=sm.add_constant(df1[["Views_platform","Ad_impression"]])
model3=sm.OLS(y,X).fit()
model3.summary()


# That is a very good increase in the R^2 value and 65% of the variation is explained. Let's check the vif values of the model.

# In[46]:


vif = pd.DataFrame()
vif['Features'] = X.columns
vif['VIF'] = [variance_inflation_factor(X.values, i) for i in range(X.shape[1])]
vif['VIF'] = round(vif['VIF'], 2)
vif = vif.sort_values(by = "VIF", ascending = False)
vif


# The model appears to be a decent model already with good vif values. Let's try adding more variables to the model.

# In[47]:


X=sm.add_constant(df1[["Views_platform","Ad_impression","Weekend"]])
model4=sm.OLS(y,X).fit()
model4.summary()


# 80% variation in the views is explained by this model. But the problem here is that the platform views became insignificant suddenly. The correlation of platform views is high with both weekend and add impression and that might be the reason for it becoming insignificant. Let's try our model without the views platform and see how much decrease there is in the R^2.

# In[48]:


X=sm.add_constant(df1[["Ad_impression","Weekend"]])
model5=sm.OLS(y,X).fit()
model5.summary()


# The decrease in R^2 the model isn't very much. Let's go ahead with this model and try adding more variables to it. We will add Character_A to the model now.

# In[49]:


X=sm.add_constant(df1[["Ad_impression","Weekend","Character_A"]])
model6=sm.OLS(y,X).fit()
model6.summary()


# The model accuracy isn't affected much and also even though character A vairable has a positive correlation and there are higher mean views when the character A is present in the show, the coefficient is coming out to be negative (even though it's not significant.) This must be because Character A has a very high correlation with Ad_impression (as high as 0.64). Let's try to remove the ad_platform and add lag views variable and see what our model turns out to be.

# In[50]:


X=sm.add_constant(df1[["Weekend","Character_A","Lag_Views"]])
model7=sm.OLS(y,X).fit()
model7.summary()


# This model also appears to be a very decent model with 73% of r-square value. We noticed in EDA that the cricket match didn't have any significant effect on the model, lets add the variable to model5 and see it's effect.

# In[51]:


X=sm.add_constant(df1[["Weekend","Character_A","Lag_Views","Cricket_match_india"]])
model8=sm.OLS(y,X).fit()
model8.summary()


# There's no effect of the variable on our model and infact the adjusted R^2 decreased after addition of the variable. We'll drop the variable.

# We have ignored the variable visitors because it has 0.94 correlation with the views on platform and hence they majorly go hand in hand

# In[52]:


X=sm.add_constant(df1[["Weekend","Character_A","Lag_Views","Cricket_match_india"]])
model9=sm.OLS(y,X).fit()
model9.summary()


# We will go ahead and fit model 5 as it seems to be the most perfect with given the variables. 
# We also would fit model 7 later as Character A seem to be an important variable.

# ### Prediction using model 4

# In[78]:


X=sm.add_constant(df1[["Ad_impression","Weekend"]])
df1["y_pred_model4"]=model5.predict(X)


# In[79]:


plt.figure(figsize=[10,6])
df1.Views_show.plot.line(label="Show Views")
df1.y_pred_model4.plot.line(label="Predicted Views")
plt.title("Actual Views vs Predicted Views")
plt.ylabel("Normalized Views")
plt.legend()
plt.show()


# ### Residual Analysis

# In[80]:


res_model4=df1.Views_show-df1.y_pred_model4


# In[81]:


sns.distplot(res_model4)
plt.xlabel('Errors', fontsize = 18)     
plt.show()


# The residual seems to be normally distributed.

# ### Model 4 Evaluation

# In[82]:


r2_score(df1.Views_show,df1.y_pred_model4)


# In[104]:


from sklearn.linear_model import LinearRegression


# We will used cross validation for the model evaluation as the data is too small to be divided into test and train. 

# In[115]:


X=df1[["Ad_impression","Weekend"]]
lm=LinearRegression()
np.random.seed(2)
folds = KFold(n_splits = 5, shuffle = True, random_state = 9)
scores = cross_val_score(lm, X, y, scoring='r2', cv=folds)
scores.mean()


# The R^2 comes out to be very close to the actual R^2 of the model on the data.

# ## Prediction using model 7

# In[118]:


X=sm.add_constant(df1[["Weekend","Character_A","Lag_Views"]])
df1["y_pred_model7"]=model7.predict(X)


# In[120]:


plt.figure(figsize=[10,6])
df1.Views_show.plot.line(label="Show Views")
df1.y_pred_model7.plot.line(label="Predicted Views")
plt.title("Actual Views vs Predicted Views")
plt.ylabel("Normalized Views")
plt.legend()
plt.show()


# ### Residual Analysis

# In[122]:


res_model7=df1.Views_show-df1.y_pred_model7
sns.distplot(res_model7)
plt.xlabel('Errors', fontsize = 18)     
plt.show()


# The residual seems to be normally distributed.

# ### Model 7 Evaluation

# In[123]:


r2_score(df1.Views_show,df1.y_pred_model7)


# In[124]:


X=df1[["Weekend","Character_A","Lag_Views"]]
lm=LinearRegression()
np.random.seed(2)
folds = KFold(n_splits = 5, shuffle = True, random_state = 9)
scores = cross_val_score(lm, X, y, scoring='r2', cv=folds)
scores.mean()


# The R^2 comes out to be very close to the actual R^2 of the model on the data.

# Thus we can say that in order to increase the viewership :-
#     1. The company should increase the advertisment impressions, especially for the weekends.
#     2. They should bring Character A more frequently.

# In[ ]:




