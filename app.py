import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error,r2_score
import streamlit as st
st.title("🏠 House Price Prediction App")
st.write("Predict median house prices using multiple linear regression")
@st.cache_data
def load_data():
    df=pd.read_csv("Boston.csv")
    return df 
df=load_data()
df.columns=df.columns.str.strip()
st.subheader("Dataset preview")
st.write(df.head())
features=['RM','LSTAT','PTRATIO','INDUS','NOX','AGE']
target='MEDV'
X=df[features]
y=df[target]
X=X.fillna(X.mean())
y=y.fillna(y.mean())
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
model=LinearRegression()
model.fit(X_train,y_train)
y_pred=model.predict(X_test)
mse=mean_squared_error(y_test,y_pred)
r2=r2_score(y_test,y_pred)
st.subheader("Model Performance")
st.write(f"**Mean Squared Error:**{mse:.2f}")
st.write(f"**R2 Score:**{r2:.2f}")
st.subheader("Feature Importance")
coeff_df=pd.DataFrame({'Feature':features,'Coefficient':model.coef_})
st.write(coeff_df)
st.subheader("Actual vs Predicted Prices")
fig,ax=plt.subplots()
ax.scatter(y_test,y_pred)
ax.set_xlabel("Actual Prices")
ax.set_ylabel("Predicted  Prices")
ax.set_title("Actual vs Predicted")
st.pyplot(fig)
st.subheader("Predict house price")
rm = st.slider("Average Rooms (RM)", 0.0, 10.0, 5.0)
lstat = st.slider("Lower Status Population (%)", 0.0, 40.0, 10.0)
ptratio = st.slider("Pupil-Teacher Ratio", 10.0, 30.0, 18.0)
indus = st.slider("Industrial Area (%)", 0.0, 30.0, 10.0)
nox = st.slider("NOX (Pollution)", 0.0, 1.0, 0.5)
age = st.slider("Old Houses (%)", 0.0, 100.0, 50.0)
input_data=np.array([[rm,lstat,ptratio,indus,nox,age]])
prediction=model.predict(input_data)
st.success(f"🏡 Predicted House Price: ${prediction[0]*1000:.2f}")