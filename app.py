import pandas as pd
import streamlit as st 
import joblib
import numpy as np
import warnings
import scipy 

warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")

# Carregar o modelo treinado
model = joblib.load('model_etr.pkl')

st.set_page_config(
    page_title="AppPredcrim"

)

st.title('Previsão de Roubos em Campo Grande - MS')


st.header("Mês")
df1 = pd.DataFrame(
    data = [[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]],
    columns =['Jan','Fev','Mar','Abr','Mai','Jun','Jul','Ago','Set','Out','Nov','Dez'],

)
st.dataframe(df1,hide_index=True)

st.header("Área urbana")
df2 = pd.DataFrame(
    data = [[0, 1, 2, 3, 4, 5, 6]],
    columns=['Centro','Prosa','Imbirussu','Bandeira','Lagoa','Segredo','Anhanduizinho'],

)
st.dataframe(df2,hide_index=True)


# Coletando os dados de entrada do usuário
mes = st.number_input('Informe o mês da previsão:', min_value=0, max_value=11, step=0)
area = st.number_input('Em qual área urbana de Campo Grande?', min_value=0, max_value=6, step=0)
pessoas_abordadas = st.number_input('Quantidade de pessoas arbodadas:', min_value=0, max_value=9999, step=0)
veiculos_abordados_duas_rodas = st.number_input('Quantidade de veículos duas rodas abordados:', min_value=0, max_value=9999, step=0)
veiculos_duas_rodas_recuperados = st.number_input('Quantidade de veículos duas rodas recuperados (Produtos de Furto/Roubo):', min_value=0, max_value=9999, step=0)
veiculos_quatro_rodas_recuperados = st.number_input('Quantidade de veículos quatro rodas recuperados (Produtos de Furto/Roubo):', min_value=0, max_value=9999, step=0)
masculino_adolescente_conduzido = st.number_input('Quantidade de adolescente masculino conduzido:', min_value=0, max_value=9999, step=0)
masculino_adulto_foragido = st.number_input('Quantidade de adulto masculino foragido (Nº de presos por mandado de prisão/evasão):', min_value=0, max_value=9999, step=0)
masculino_adolescente_foragido = st.number_input('Quantidade de adolescente masculino foragido (Nº de apreendidos por mandado de apreensão/evasão):', min_value=0, max_value=9999, step=0)    
apf = st.number_input('Quantidade de ocorrências que resultaram em auto de prisão em flagrante delito:', min_value=0, max_value=9999, step=0)
rondas_preventivas_urbana = st.number_input('Quantidade de rondas preventivas realizadas na área urbana:', min_value=0, max_value=9999, step=0)
       
# Previsão
if st.button('Prever'):
    features = pd.DataFrame({
        'Mês': [mes],
        'Área': [area],
        'Pessoas Abordadas': [pessoas_abordadas],
        'Abordagem a veículos duas rodas': [veiculos_abordados_duas_rodas],
        'Recuperação de veículos duas rodas': [veiculos_duas_rodas_recuperados],
        'Recuperação de veículos quatro rodas': [veiculos_quatro_rodas_recuperados],
        'Condução de adolescente masculino': [masculino_adolescente_conduzido],
        'Masculino adulto foragido': [masculino_adulto_foragido],
        'Masculino adolescente foragido': [masculino_adolescente_foragido],
        'Auto de prisão em flagrante': [apf],
        'Rondas preventivas urbana': [rondas_preventivas_urbana]
    })  

    prediction = model.predict(features)

    st.write(f'Com base no número de ações preventivas informadas, a quantidade esperada de roubos é: {round(prediction[0])}')


st.header("Ficha técnica do modelo")
df = pd.DataFrame(
    data = [['ExtraTreesRegressor',8.91, 134.46, 11.59, 0.61, 25.51]],
    columns=['Algoritmo','MAE','MSE','RMSE','R2','MAPE'],

)
st.dataframe(df,hide_index=True)
