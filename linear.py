import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
df=pd.read_csv("student_scores.csv")
x=df.iloc[:. :-1].values #features
y=df.iloc[:. -1],values
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
model=LinearRegression()
model.fit(x_train,y_train)
st.title("exam score prediction model")
st.write("enter the no.of hours you have studeied for the exam")
hours=st.number_input("hours studied",min_value=0.0,step=0.1)
if st.button("predict score"):
  predicted_score=model.predict([[hours]])[0]
  st.succed(f"predicted score:{predicted_score: .2f}")
st.write("sample training Data")
st.dataframe(df)
