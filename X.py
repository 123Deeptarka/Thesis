#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Libraries
#Hyper parameter tuned using Keras Tuner 
import streamlit as st 
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D,Activation,MaxPooling1D,Dense,Flatten
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import keras

st.write(""" #  Pushover Curve Predicion App
         """)

st.sidebar.header("Specify Input Parameters")
# In[2]:


#import data
df = pd.read_excel('C:/Users/Deeptarka.Roy/Desktop/A.xlsx')
df.head()


# In[3]:


x = df[["D","LD","fc","fyl","fyt","pl","pt","Ny"]]
y = df[["DS1","DS2","DS3","DS4","F1","F2","F3","F4"]]


# In[4]:
def user_input_features():
    D =st.sidebar.slider("D (mm)",x.D.min(),x.D.max(),x.D.mean())
    LD =st.sidebar.slider("L/D",x.LD.min(),x.LD.max(),x.LD.mean())   
    fc =st.sidebar.slider("fc (MPa)",x.fc.min(),x.fc.max(),x.fc.mean())
    fyl =st.sidebar.slider("fyl (MPa)",x.fyl.min(),x.fyl.max(),x.fyl.mean())
    fyt =st.sidebar.slider("fyt (MPa)",x.fyt.min(),x.fyt.max(),x.fyt.mean())
    pl =st.sidebar.slider("pl",x.pl.min(),x.pl.max(),x.pl.mean())
    pt =st.sidebar.slider("pt",x.pt.min(),x.pt.max(),x.pt.mean())
    Ny =st.sidebar.slider("Ny",x.Ny.min(),x.Ny.max(),x.Ny.mean())
    data={"D (mm)":D,"L/D":LD,"fc (MPa)":fc,"fyl (MPa)":fyl,"fyt (MPa)":fyt,"pl":pl,"pt":pt,"Ny":Ny}
    features=pd.DataFrame(data,index=[0])
    return features

df=user_input_features()
#Df=df.style.format("{:.2f}")
#df_round=Df.round(2)
st.header("Input Parameters")
styles = [
    dict(selector="th", props=[("font-size", "20px"), ("font-weight", "bold"), ("color", "#484848")]),
    dict(selector="td", props=[("font-size", "16px"),("font-weight", "bold") ,("color", "#484848")])
]
# Apply styling to dataframe
st_df = df.style.set_table_styles(styles)

st.table(st_df)

#Standardize the data
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
x_new = scaler.fit_transform(x)
x_new = pd.DataFrame(x_new,columns = ['D','LD','fc','fyl','fyt','pl','pt','Ny'])
df_new=pd.DataFrame(df,columns = ['D','LD','fc','fyl','fyt','pl','pt','Ny'])
y_new=scaler.fit_transform(y)
y_new = pd.DataFrame(y_new,columns = ['DS1','DS2','DS3','DS4','F1',"F2","F3","F4"])
#y_new.shape


# In[5]:


#print(x_new.to_numpy)
A=x_new.to_numpy().reshape(len(x),8,1)
B=y_new.to_numpy().reshape(len(y),8,1)
#A.shape
df_new_test=df_new.to_numpy().reshape(len(df),8,1)

# In[7]:


from sklearn.model_selection import train_test_split
x_new_train,x_new_test,y_new_train,y_new_test=train_test_split(A,B,test_size = 0.3,random_state=42)


# In[8]:



#print(y_new_train)
#print(x_new_train)
#print(y_new_train.shape)


# In[95]:


model=Sequential([
    Conv1D(100,3,activation="relu",input_shape=(8,1)),
    MaxPooling1D(1),
    #Conv1D(20,3,activation="relu"),
    #MaxPooling1D(1),
    Flatten(),
    Dense(128,activation='relu',kernel_regularizer="l2"),
    Dense(90,activation='relu'),
    Dense(70,activation='relu'),
    keras.layers.Dropout(rate=0.30),
    Dense(8,activation="linear")
    ])
model.compile(optimizer="Adam",loss="mean_squared_error",metrics=["mae"])
model.summary()


# In[96]:


history=model.fit(x_new_train,y_new_train,epochs=100,batch_size=60)
   


# In[97]:


y_pred=model.predict(x_new_test)
#print(y_pred.shape)
y_pred_train=model.predict(x_new_train)
Y_pred=scaler.inverse_transform(y_pred)
Y_pred_train=scaler.inverse_transform(y_pred_train)

prediction=model.predict(df_new_test)
Prediction=scaler.inverse_transform(prediction)

st.header("Predicted Output Parameters")
#st.write(Prediction)
# In[98]:


from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score
X=y_new_test.reshape(300,8)
Y_new_test=scaler.inverse_transform(X)
X_train=y_new_train.reshape(700,8)
Y_new_train=scaler.inverse_transform(X_train)


# In[99]:


print("The R2 score on test set is :",r2_score(X,y_pred))
print("The R2 score on train set is :",r2_score(X_train,y_pred_train))


# In[100]:


print("The MAE on test set is ",mean_absolute_error(Y_pred,Y_new_test))
print("The MAE on train set is ",mean_absolute_error(Y_pred_train,Y_new_train))


# In[16]:


Y=scaler.inverse_transform(y_pred)
#Y


# In[17]:


P=pd.DataFrame(Prediction,columns=["DS1","DS2","DS3","DS4","F1 (kN)","F2 (kN)","F3 (kN)","F4 (kN)"])
P_DS=P[["DS1","DS2","DS3","DS4"]]
P_F=P[["F1 (kN)","F2 (kN)","F3 (kN)","F4 (kN)"]]
styles = [
    dict(selector="th", props=[("font-size", "20px"), ("font-weight", "bold"), ("color", "#484848")]),
    dict(selector="td", props=[("font-size", "16px"),("font-weight", "bold") ,("color", "#484848")])
]
# Apply styling to dataframe
styled_df = P.style.set_table_styles(styles)

# Display styled dataframe in Streamlit
st.table(styled_df)
#st.write(P_DS)
#st.write(P_F)
#st.dataframe(P)

# In[18]:


R=scaler.inverse_transform(X)
Q=pd.DataFrame(R,columns=["DS1","DS2","DS3","DS4","F1","F2","F3","F4"])
Q_DS=Q[["DS1","DS2","DS3","DS4"]]
Q_F=Q[["F1","F2","F3","F4"]]



# In[19]:


Outpred_DR_0=P_DS.to_numpy().reshape(4,)

Outpred_F_0=P_F.to_numpy().reshape(4,)
Outtest_DR_0=Q_DS.iloc[7].to_numpy().reshape(4,)
Outtest_F_0=Q_F.iloc[7].to_numpy().reshape(4,)


# In[20]:


a=np.insert(Outpred_DR_0,0,0)
b=np.insert(Outpred_F_0,0,0)
a1=np.insert(Outtest_DR_0,0,0)
b1=np.insert(Outtest_F_0,0,0)




# In[21]:
st.header("Predicted Pushover Curve ")
fig,ax=plt.subplots(figsize=(6,3))
ax.plot(a,b,label="Predicted Pushover Curve",marker="o")
#ax.plot(a1,b1,label="Simulated Pushover Curve",marker="o")
ax.set_xlabel("Drift Ratio (%)")
ax.set_ylabel("Force (kN)")
#ax.set_title("Predicted VS Simulated Pushover Curves")
#ax.legend()
#ax.show()
st.pyplot(fig)

# In[ ]:


