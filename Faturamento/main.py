import pandas as pd
import streamlit as st

st.title('Faturamento')
st.write('Análise de Faturamento')
pd = pd.read_csv('Faturamento.csv', sep=';')
st.dataframe(pd)