#!/usr/bin/env python
# coding: utf-8

# # import library

# In[1]:


import pandas as pd
import seaborn as sb
import numpy as np


# In[2]:


from warnings import filterwarnings
filterwarnings("ignore")


# # Read data and preview dataset

# In[3]:


df_train = pd.read_csv("C:/Users/barsh/Downloads/training_set (1).csv")
df_test = pd.read_csv("C:/Users/barsh/Downloads/testing_set (1).csv")


# In[4]:


df_train.head(2)


# In[5]:


df_train.info()


# In[6]:


df_train.describe()


# In[7]:


df_train.Loan_Status.value_counts()


# In[8]:


df_train.shape


# # missing value treatement

# In[9]:


df_train.isna().sum()


# In[ ]:





# In[10]:


df_train.Loan_Amount_Term.value_counts()


# In[11]:


for i in df_train.columns:
    if(df_train[i].dtypes=="object"):
        x=df_train[i].mode()[0]
        df_train[i]=df_train[i].fillna(x)
    else:
        x=df_train[i].mean()
        df_train[i]=df_train[i].fillna(x)


# In[12]:


df_train.isna().sum()


# In[13]:


for i in df_test.columns:
    if(df_test[i].dtypes=="object"):
        x=df_test[i].mode()[0]
        df_test[i]=df_test[i].fillna(x)
    else:
        x=df_test[i].mean()
        df_test[i]=df_test[i].fillna(x)


# In[14]:


df_train.shape


# # outliers

# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[15]:


df_train.shape


# In[ ]:





# # 1.Check eligibility of the Customer given the inputs described above

# In[16]:


df1_train = df_train
df1_test = df_test


# In[17]:


df1_train["TotalIncome"]=df1_train["ApplicantIncome"] + df1_train["CoapplicantIncome"]
df1_test["TotalIncome"]= df1_test["ApplicantIncome"] + df1_test["CoapplicantIncome"]

# as we taken total income of aplicant we can drop coaplicant income and aplicant income
# In[18]:


df1_train = df1_train.drop(labels=["ApplicantIncome","CoapplicantIncome"],axis=1)
df1_test = df1_test.drop(labels=["ApplicantIncome","CoapplicantIncome"],axis=1)


# In[19]:


df1_train.head(2)


# In[20]:


df1_test.head(2)


# # Define X and Y Columns

# In[21]:


Y = df1_train[["Loan_Status"]]
X = df1_train.drop(labels=["Loan_Status","Loan_ID"],axis=1)


# In[22]:


X #convert this into categorical


# In[23]:


Xnew = pd.get_dummies(X)


# In[24]:


Xnew.columns


# # split dataset into train and test

# In[25]:


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest = train_test_split(Xnew,Y,test_size=0.2,random_state=21)


# # Model-1 Logistic Regression

# In[26]:


from sklearn.linear_model import LogisticRegression
lr = LogisticRegression()
model = lr.fit(xtrain,ytrain)
tr_pred = model.predict(xtrain)
ts_pred = model.predict(xtest)

from sklearn.metrics import accuracy_score
tr_acc = accuracy_score(ytrain,tr_pred)
ts_acc = accuracy_score(ytest,ts_pred)

ts_acc,tr_acc


# # 2. KNN Classifier

# In[27]:


Y = df_train[["Loan_Status"]]
X = df_train.drop(labels = ["Loan_Status"],axis=1)

con=[]
cat=[]
for i in X.columns:
    if df_train[i].dtypes=='object':
        cat.append(i)
    else:
        con.append(i)

from sklearn.preprocessing import StandardScaler
ss=StandardScaler()
X1=pd.DataFrame(ss.fit_transform(X[con]),columns=con)
X2=pd.get_dummies(X[cat])
Xnewk=X1.join(X2)

from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(Xnewk,Y,test_size=0.2,random_state=21)

from sklearn.neighbors import KNeighborsClassifier
knc=KNeighborsClassifier()
model=knc.fit(xtrain,ytrain)
tr_pred=model.predict(xtrain)
ts_pred=model.predict(xtest)
from sklearn.metrics import accuracy_score
tr_acc = accuracy_score(ytrain,tr_pred)
ts_acc = accuracy_score(ytest,ts_pred)
print(tr_acc,ts_acc)

from sklearn.neighbors import KNeighborsClassifier
knc=KNeighborsClassifier(n_neighbors=6)
model=knc.fit(xtrain,ytrain)
tr_pred=model.predict(xtrain)
ts_pred=model.predict(xtest)
from sklearn.metrics import accuracy_score
tr_acc = accuracy_score(ytrain,tr_pred)
ts_acc = accuracy_score(ytest,ts_pred)
print(tr_acc,ts_acc)

from sklearn.neighbors import KNeighborsClassifier
knc=KNeighborsClassifier(n_neighbors=13)
model=knc.fit(xtrain,ytrain)
tr_pred=model.predict(xtrain)
ts_pred=model.predict(xtest)
from sklearn.metrics import accuracy_score
tr_acc = accuracy_score(ytrain,tr_pred)
ts_acc = accuracy_score(ytest,ts_pred)
tr_acc,ts_acc


# # Model-2 Decision Tree Classifier

# In[28]:


from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion="entropy",random_state=21)
model = dtc.fit(xtrain,ytrain)  #randomness
tr_pred = model.predict(xtrain)
ts_pred = model.predict(xtest)

from sklearn.metrics import accuracy_score
tr_acc= accuracy_score(ytrain,tr_pred)
ts_acc = accuracy_score(ytest,ts_pred)

tr_acc,ts_acc


# In[29]:


from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(criterion="gini",random_state=21)
model = dtc.fit(xtrain,ytrain)   #purity of node
tr_pred = model.predict(xtrain)
ts_pred = model.predict(xtest)

from sklearn.metrics import accuracy_score
tr_acc = accuracy_score(ytrain,tr_pred)
ts_acc = accuracy_score(ytest,ts_pred)

tr_acc,ts_acc


# # Adaboost Classifier

# In[30]:


from sklearn.ensemble import AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
dtc = DecisionTreeClassifier(random_state=21,max_depth=2)
abc = AdaBoostClassifier(dtc,n_estimators=100)
model = abc.fit(xtrain,ytrain)
tr_pred = model.predict(xtrain)
ts_pred = model.predict(xtest)

from sklearn.metrics import accuracy_score
tr_acc = accuracy_score(ytrain,tr_pred)
ts_acc = accuracy_score(ytest,ts_pred)

tr_acc,ts_acc


# # RandomForest Classifier

# In[31]:


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(random_state=21)
model = rfc.fit(xtrain,ytrain)
tr_pred = model.predict(xtrain)
ts_pred = model.predict(xtest)

from sklearn.metrics import accuracy_score
tr_acc = accuracy_score(ytrain,tr_pred)
ts_acc = accuracy_score(ytest,ts_pred)

tr_acc,ts_acc


# # Overfitting - Prunning

# In[32]:


from sklearn.ensemble import RandomForestClassifier
rfr = RandomForestClassifier(n_estimators=150,max_depth=2)
model = rfr.fit(xtrain,ytrain)
tr_pred = model.predict(xtrain)
ts_pred = model.predict(xtest)
from sklearn.metrics import accuracy_score
tr_acc = accuracy_score(ytrain,tr_pred)
ts_acc  = accuracy_score(ytest,ts_pred)

tr_acc,ts_acc


# # GridSearchCV for Tunning

# In[33]:


tg = {"max_depth":range(2,20,1),"n_estimators":range(2,200,10)}
rfc = RandomForestClassifier(random_state=21)
from sklearn.model_selection import GridSearchCV
cv = GridSearchCV(rfc,tg,scoring="accuracy",cv=4)
cvmodel = cv.fit(Xnew,Y)
cvmodel.best_params_


# In[34]:


from sklearn.ensemble import RandomForestClassifier
rfr = RandomForestClassifier(n_estimators=32,max_depth=4)
model = rfr.fit(xtrain,ytrain)
tr_pred = model.predict(xtrain)
ts_pred = model.predict(xtest)
from sklearn.metrics import accuracy_score
tr_acc = accuracy_score(ytrain,tr_pred)
ts_acc  = accuracy_score(ytest,ts_pred)

tr_acc,ts_acc


# In[35]:


tg = {"min_samples_leaf":range(2,20,1),"n_estimators":range(2,200,10)}
rfc = RandomForestClassifier(random_state=21)
from sklearn.model_selection import GridSearchCV
cv = GridSearchCV(rfc,tg,scoring="accuracy",cv=4)
cvmodel = cv.fit(Xnew,Y)
cvmodel.best_params_


# In[36]:


from sklearn.ensemble import RandomForestClassifier
rfr = RandomForestClassifier(n_estimators=62,min_samples_leaf=5)
model = rfr.fit(xtrain,ytrain)
tr_pred = model.predict(xtrain)
ts_pred = model.predict(xtest)
from sklearn.metrics import accuracy_score
tr_acc = accuracy_score(ytrain,tr_pred)
ts_acc  = accuracy_score(ytest,ts_pred)

tr_acc,ts_acc


# In[37]:


tg = {"min_samples_split":range(2,20,1),"n_estimators":range(2,200,10)}
rfc = RandomForestClassifier(random_state=21)
from sklearn.model_selection import GridSearchCV
cv = GridSearchCV(rfc,tg,scoring="accuracy",cv=4)
cvmodel = cv.fit(Xnew,Y)
cvmodel.best_params_


# In[38]:


from sklearn.ensemble import RandomForestClassifier
rfr = RandomForestClassifier(n_estimators=22,min_samples_split=18)
model = rfr.fit(xtrain,ytrain)
tr_pred = model.predict(xtrain)
ts_pred = model.predict(xtest)
from sklearn.metrics import accuracy_score
tr_acc = accuracy_score(ytrain,tr_pred)
ts_acc  = accuracy_score(ytest,ts_pred)

tr_acc,ts_acc


# In[39]:


Z =df1_test.drop(labels=["Loan_ID"],axis=1) 
Znew = pd.get_dummies(Z)


# In[40]:


from sklearn.ensemble import RandomForestClassifier
rfc = RandomForestClassifier(random_state=21,max_depth=4,min_samples_leaf=4,n_estimators=32)
rfc_model_final = rfc.fit(Xnew,Y)


# In[41]:


loan_status = pd.DataFrame(rfc_model_final.predict(Znew),columns=["Loan_Status"])
pred_1 = df1_test.join(loan_status)[["Loan_ID","Loan_Status"]]


# In[42]:


pred_1.head()


# In[ ]:





# In[ ]:





# # 2- If customer is not eligible for the input required amount and duration:
# 
# a.)what can be amount for the given duration.(Regression)
# 

# In[43]:


df2_train = df_train[df_train.Loan_Status=="Y"]


# In[44]:


df2_test = df_test.join(loan_status)
df2_test = df2_test[df2_test.Loan_Status=="N"]


# In[45]:


df2_train.shape


# In[46]:


df2_test.shape


# In[47]:


df2_train.index = range(0,422,1)


# In[48]:


df2_test.index = range(0,59,1)


# # Remove insignificant columns

# In[49]:


df3_train = df2_train.drop(labels=["Loan_ID","ApplicantIncome","CoapplicantIncome","Loan_Status"],axis=1)
df3_test = df2_test.drop(labels=["Loan_ID","ApplicantIncome","CoapplicantIncome","LoanAmount","Loan_Status"],axis=1)


# # Define x and Y columns

# In[50]:


Y = df3_train[['LoanAmount']]
X = df3_train.drop(labels=["LoanAmount"],axis=1)


# In[51]:


cat = []
con = []
for i in X.columns:
    if(X[i].dtypes=="object"):
        cat.append(i)
    else:
        con.append(i)  


# In[52]:


X1 = pd.get_dummies(X[cat])


# # Standardisation

# In[53]:


from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X2 = pd.DataFrame(ss.fit_transform(X[con]),columns=con)
Xnew = X1.join(X2)


# In[54]:


Xnew.shape


# # Split data into train and test

# In[ ]:





# In[55]:


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=21)


# In[ ]:





# In[56]:


Z = df3_test

cat1 = []
con1 = []

for i in Z.columns:
    if(Z[i].dtypes=="object"):
        cat1.append(i)
    else:
        con1.append(i)
 
  


# In[57]:


Z1 = pd.get_dummies(Z[cat1])    
from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
Z2 = pd.DataFrame(ss.fit_transform(Z[con1]),columns=con1)
Znew = Z1.join(Z2) 


# In[58]:


Znew.shape


# In[ ]:





# # Split data into train and test

# In[59]:


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=21)


# In[ ]:





# # model-1 KNN-Model

# In[60]:


from sklearn.neighbors import KNeighborsRegressor
knr = KNeighborsRegressor(n_neighbors=5)
model = knr.fit(xtrain,ytrain)
tr_pred = model.predict(xtrain)
ts_pred = model.predict(xtest)
from sklearn.metrics import mean_absolute_error
tr_err = mean_absolute_error(ytrain,tr_pred)
ts_err  = mean_absolute_error(ytest,ts_pred)

tr_err,ts_err


# # DecisionTree Regressor

# In[61]:


from sklearn.tree import DecisionTreeRegressor
dtr = DecisionTreeRegressor(random_state=21)
model = dtr.fit(xtrain,ytrain)
tr_pred = model.predict(xtrain)
ts_pred = model.predict(xtest)
from sklearn.metrics import mean_absolute_error
tr_err = mean_absolute_error(ytrain,tr_pred)
ts_err  = mean_absolute_error(ytest,ts_pred)

tr_err,ts_err


# In[62]:


tr = []
ts = []
for i in range(2,12,1):
    knr = KNeighborsRegressor(n_neighbors=i)
    model = knr.fit(xtrain,ytrain)
    tr_pred = model.predict(xtrain)
    ts_pred = model.predict(xtest)
    from sklearn.metrics import mean_absolute_error
    tr_err = mean_absolute_error(ytrain,tr_pred)
    ts_err  = mean_absolute_error(ytest,ts_pred)
    tr.append(tr_err)
    ts.append(ts_err)
import matplotlib.pyplot as plt
plt.plot(tr)
plt.plot(ts)


# In[63]:


from sklearn.neighbors import KNeighborsRegressor
knr = KNeighborsRegressor(n_neighbors=6)
model = knr.fit(xtrain,ytrain)
tr_pred = model.predict(xtrain)
ts_pred = model.predict(xtest)
from sklearn.metrics import mean_absolute_error
tr_err = mean_absolute_error(ytrain,tr_pred)
ts_err  = mean_absolute_error(ytest,ts_pred)

tr_err,ts_err


# # model-3 RandomForestRegressor

# In[64]:


from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(random_state=21)
model = rfr.fit(xtrain,ytrain)
tr_pred = model.predict(xtrain)
ts_pred = model.predict(xtest)
from sklearn.metrics import mean_absolute_error
tr_err = mean_absolute_error(ytrain,tr_pred)
ts_err  = mean_absolute_error(ytest,ts_pred)

tr_err,ts_err


# # Overfitting-pruning

# In[65]:


from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(n_estimators=12,max_depth=2)
model = rfr.fit(xtrain,ytrain)
tr_pred = model.predict(xtrain)
ts_pred = model.predict(xtest)
from sklearn.metrics import mean_absolute_error
tr_err = mean_absolute_error(ytrain,tr_pred)
ts_err  = mean_absolute_error(ytest,ts_pred)

tr_err,ts_err


# # GridSearchCV - tunning

# In[66]:


tg = {"max_depth":range(2,20,1),"n_estimators":range(2,200,10)}
rfr = RandomForestRegressor(random_state=21)
from sklearn.model_selection import GridSearchCV
cv = GridSearchCV(rfr,tg,scoring = "neg_mean_absolute_error",cv=4)
cvmodel = cv.fit(Xnew,Y)
cvmodel.best_params_


# In[67]:


tg = {"min_samples_leaf":range(2,20,1),"n_estimators":range(2,200,10)}
rfr = RandomForestRegressor(random_state=21)

from sklearn.model_selection import GridSearchCV
cv = GridSearchCV(rfr,tg,scoring = "neg_mean_absolute_error",cv=4)
cvmodel = cv.fit(Xnew,Y)
cvmodel.best_params_


# In[68]:


tg = {"min_samples_split":range(2,20,1),"n_estimators":range(2,200,10)}
rfr = RandomForestRegressor(random_state=21)

from sklearn.model_selection import GridSearchCV
cv = GridSearchCV(rfr,tg,scoring = "neg_mean_absolute_error",cv=4)
cvmodel = cv.fit(Xnew,Y)
cvmodel.best_params_


# In[69]:


from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(n_estimators=12,max_depth=5)
model = rfr.fit(xtrain,ytrain)
tr_pred = model.predict(xtrain)
ts_pred = model.predict(xtest)
from sklearn.metrics import mean_absolute_error
tr_err = mean_absolute_error(ytrain,tr_pred)
ts_err  = mean_absolute_error(ytest,ts_pred)

tr_err,ts_err


# In[70]:


from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(n_estimators=12,min_samples_leaf=5)
model = rfr.fit(xtrain,ytrain)
tr_pred = model.predict(xtrain)
ts_pred = model.predict(xtest)
from sklearn.metrics import mean_absolute_error
tr_err = mean_absolute_error(ytrain,tr_pred)
ts_err  = mean_absolute_error(ytest,ts_pred)

tr_err,ts_err


# In[71]:


from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(n_estimators=12,min_samples_split=19)
model = rfr.fit(xtrain,ytrain)
tr_pred = model.predict(xtrain)
ts_pred = model.predict(xtest)
from sklearn.metrics import mean_absolute_error
tr_err = mean_absolute_error(ytrain,tr_pred)
ts_err  = mean_absolute_error(ytest,ts_pred)

tr_err,ts_err


# In[72]:


rfr = RandomForestRegressor(random_state=21,max_depth=2,n_estimators=12)
rfr_final_model=rfr.fit(Xnew,Y)


# # Final Predection On Testing Data

# In[73]:


#give test input data to Random Forest Regressor


# In[74]:


loan_amount_new = pd.DataFrame(rfr_final_model.predict(Znew),columns=["LoanAmount_New"])
pred_2 = df2_test.join(loan_amount_new)[["Loan_ID","Loan_Status","LoanAmount","LoanAmount_New"]]


# In[75]:


pred_2.head()


# In[ ]:





# # 3.if duration is less than equal to 20 years, is customer eligible for required amount for some longer duration? What is that duration?
# 
# 

# In[76]:


df4_train = df_train[df_train.Loan_Status=="Y"]


# In[77]:


df4_test = df_test.join(loan_status)
df4_test = df4_test[(df4_test.Loan_Status=="N") & (df4_test.Loan_Amount_Term<=240)]


# In[78]:


df4_train.shape


# In[79]:


df4_test.shape


# In[80]:


df4_train.head()


# In[81]:


df4_train.index = range(0,422,1)


# In[82]:


df4_test.index = range(0,5,1)


# # Remove columns

# In[83]:


df5_train = df4_train.drop(labels=["Loan_ID","ApplicantIncome","CoapplicantIncome","Loan_Status"],axis=1)
df5_test = df4_test.drop(labels=["Loan_ID","ApplicantIncome","CoapplicantIncome","Loan_Amount_Term","Loan_Status"],axis=1)


# In[84]:


df5_train.shape


# In[85]:


df5_test.shape


# # Define X and Y columns

# In[86]:


Y = df5_train[["Loan_Amount_Term"]]
X = df5_train.drop(labels=["Loan_Amount_Term"],axis=1)
Z = df5_test


# In[87]:


cat = []
con = []
for i in X.columns:
    if(X[i].dtypes=="object"):
        cat.append(i)
    else:
        con.append(i) 


# In[88]:


cat1 = []
con1 = []
for i in Z.columns:
    if(Z[i].dtypes=="object"):
        cat1.append(i)
    else:
        con1.append(i)


# In[89]:


X1 = pd.get_dummies(X[cat])
Z1 = pd.get_dummies(Z[cat1])


# In[90]:


len(X1.columns),len(Z1.columns)


# In[91]:


col_names = []
for i in X1.columns:
    if i not in Z1.columns:
        col_names.append(i)


# In[92]:


len(col_names)


# In[93]:


Z1[col_names]=0


# # Preprocessing

# In[94]:


from sklearn.preprocessing import StandardScaler
ss = StandardScaler()
X2 = pd.DataFrame(ss.fit_transform(X[con]),columns=con)
Z2 = pd.DataFrame(ss.fit_transform(Z[con1]),columns=con1)


# In[95]:


Xnew = X1.join(X2)
Znew = Z1.join(Z2)


# In[96]:


Xnew.shape


# In[97]:


Znew.shape


# # Split into training and testing set

# In[98]:


from sklearn.model_selection import train_test_split
xtrain,xtest,ytrain,ytest=train_test_split(Xnew,Y,test_size=0.2,random_state=21)


# # Model-1 KNNKNeighborsRegressor

# In[99]:


from sklearn.neighbors import KNeighborsRegressor
knr = KNeighborsRegressor(n_neighbors=5)
model = knr.fit(xtrain,ytrain)
tr_pred = model.predict(xtrain)
ts_pred = model.predict(xtest)
from sklearn.metrics import mean_absolute_error
tr_err = mean_absolute_error(ytrain,tr_pred)
ts_err  = mean_absolute_error(ytest,ts_pred)

tr_err,ts_err


# # model-2 DecisionTreeRegressor

# In[100]:


from sklearn.tree import DecisionTreeRegressor
dtr = DecisionTreeRegressor(random_state=21)
model = dtr.fit(xtrain,ytrain)
tr_pred = model.predict(xtrain)
ts_pred = model.predict(xtest)
from sklearn.metrics import mean_absolute_error
tr_err = mean_absolute_error(ytrain,tr_pred)
ts_err  = mean_absolute_error(ytest,ts_pred)

tr_err,ts_err


# # model-3 RandomForestRegressor

# In[101]:


from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(random_state=21)
model = rfr.fit(xtrain,ytrain)
tr_pred = model.predict(xtrain)
ts_pred = model.predict(xtest)
from sklearn.metrics import mean_absolute_error
tr_err = mean_absolute_error(ytrain,tr_pred)
ts_err  = mean_absolute_error(ytest,ts_pred)

tr_err,ts_err


# # Overfitting - prunning

# In[102]:


from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(n_estimators=12,max_depth=2)
model = rfr.fit(xtrain,ytrain)
tr_pred = model.predict(xtrain)
ts_pred = model.predict(xtest)
from sklearn.metrics import mean_absolute_error
tr_err = mean_absolute_error(ytrain,tr_pred)
ts_err  = mean_absolute_error(ytest,ts_pred)

tr_err,ts_err


# # GridSearchCV -tunning

# In[103]:


tg = {"max_depth":range(2,20,1),"n_estimators":range(2,200,10)}
rfr = RandomForestRegressor(random_state=21)
from sklearn.model_selection import GridSearchCV
cv = GridSearchCV(rfr,tg,scoring = "neg_mean_absolute_error",cv=4)
cvmodel = cv.fit(Xnew,Y)
cvmodel.best_params_


# In[104]:


tg = {"min_samples_leaf":range(2,20,1),"n_estimators":range(2,200,10)}
rfr = RandomForestRegressor(random_state=21)

from sklearn.model_selection import GridSearchCV
cv = GridSearchCV(rfr,tg,scoring = "neg_mean_absolute_error",cv=4)
cvmodel = cv.fit(Xnew,Y)
cvmodel.best_params_


# In[105]:


tg = {"min_samples_split":range(2,20,1),"n_estimators":range(2,200,10)}
rfr = RandomForestRegressor(random_state=21)

from sklearn.model_selection import GridSearchCV
cv = GridSearchCV(rfr,tg,scoring = "neg_mean_absolute_error",cv=4)
cvmodel = cv.fit(Xnew,Y)
cvmodel.best_params_


# In[106]:


from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(n_estimators=2,max_depth=4)
model = rfr.fit(xtrain,ytrain)
tr_pred = model.predict(xtrain)
ts_pred = model.predict(xtest)
from sklearn.metrics import mean_absolute_error
tr_err = mean_absolute_error(ytrain,tr_pred)
ts_err  = mean_absolute_error(ytest,ts_pred)

tr_err,ts_err


# In[107]:


from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(n_estimators=2,min_samples_leaf=3)
model = rfr.fit(xtrain,ytrain)
tr_pred = model.predict(xtrain)
ts_pred = model.predict(xtest)
from sklearn.metrics import mean_absolute_error
tr_err = mean_absolute_error(ytrain,tr_pred)
ts_err  = mean_absolute_error(ytest,ts_pred)

tr_err,ts_err


# In[108]:


from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(n_estimators=2,min_samples_split=6)
model = rfr.fit(xtrain,ytrain)
tr_pred = model.predict(xtrain)
ts_pred = model.predict(xtest)
from sklearn.metrics import mean_absolute_error
tr_err = mean_absolute_error(ytrain,tr_pred)
ts_err  = mean_absolute_error(ytest,ts_pred)

tr_err,ts_err


# # Random Forest Regressor

# In[109]:


#give test input data to Random Forest Regressor


# In[110]:


rfr = RandomForestRegressor(random_state=21,max_depth=2,n_estimators=12)
rfr_final_model_2=rfr.fit(Xnew,Y)


# In[111]:


loan_amount_term_new = pd.DataFrame(rfr_final_model_2.predict(Znew),columns=["Loan_Amount_Term_New"])
pred_3 = df4_test.join(loan_amount_term_new)[["Loan_ID","Loan_Status","Loan_Amount_Term","Loan_Amount_Term_New"]]


# In[112]:


pred_3


# In[ ]:




