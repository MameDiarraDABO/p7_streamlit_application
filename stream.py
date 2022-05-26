import urllib
from urllib.request import urlopen
import requests
import json
import pandas as pd
import streamlit as st
import pickle
import sklearn
import numpy as np
import matplotlib.pyplot as plt
from lime import lime_tabular
import streamlit.components.v1 as components
import plotly.graph_objects as go
import plotly.express as px


#st.title('hello world')
#st.write('hello world')


#numclient = "100005"
dataframe = pd.read_csv("TEST_FINAL_SCALEE.csv")
dataframe = dataframe.iloc[0:1000, :]
numclient = dataframe['SK_ID_CURR'].values
data = dataframe.copy()
data = data.drop(['SK_ID_CURR'], axis=1)
donnee_entree = pd.read_csv("donnee_entree.csv")
donnee_entree  = donnee_entree.drop(['SK_ID_CURR'], axis=1)


#load model
model = pickle.load(open('Credit_model_reg.pkl','rb'))

# Télécherger l'image
st.sidebar.image('image-gallery-01-11-2016-b.png')


html_temp = """ 
           <div style ="background-color:darkgreen;padding:13px"> 
           <h1 style ="color:black;text-align:center;"> Dashboard Scoring Credit
           </h1> 
           </div> 
           <p></p>
           <p></p>
           """
    # display the front end aspect
st.markdown(html_temp, unsafe_allow_html=True)


#affichage formulaire
#st.title('Dashboard Scoring Credit')
st.markdown("Prédictions de scoring client, notre seuil de choix est de 50 %.")

#id_client = st.selectbox(" Veuillez choisir un identifiant à saisir: ", numclient)
id_input = st.number_input(label='Veuillez saisir l\'identifiant du client demandeur de crédit:',format="%i", value=0)
if id_input not in numclient:
    st.write("L'identifiant client n'est pas bon")

else:
    base_url = urlopen("https://app-api-2022.herokuapp.com/prediction/client/" + str(id_input))
    data_json = json.loads(base_url.read())
    #hello = str(data_json)
    #st.write(hello)
    st.write(data_json)

    i = dataframe['SK_ID_CURR'] == id_input
    Y = dataframe[i]
    Y = Y.drop(['SK_ID_CURR'], axis=1)
    # transformation des données
    num = np.array(Y)
    
    pr = model.predict_proba(num)[:, 1]
    if pr > 0.5:

        prevision = 'Rejet de la demande de crédit'
        figd = go.Figure(go.Indicator(
        domain = {'x': [0, 1], 'y': [0, 1]},
        value = 50,
        mode = "gauge+number+delta",
        title = {'text': "Avis Défavorable"},
        #delta = {'reference': 380},
        gauge = {'axis': {'range': [None, 100]},
                'steps' : [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 100], 'color': "darkred"}],
                'threshold' : {'line': {'color': "gray", 'width': 4}, 'thickness': 0.5, 'value': 50}}))

        st.write(figd)

    else :

        prevision= 'Acceptation demande de crédit'
        figv = go.Figure(go.Indicator(
        domain = {'x': [0, 1], 'y': [0, 1]},
        value = 50,
        mode = "gauge+number+delta",
        title = {'text': "Avis Favorable"},
        #delta = {'reference': 380},
        gauge = {'axis': {'range': [None, 100]},
                'steps' : [
                    {'range': [0, 50], 'color': "lightgray"},
                    {'range': [50, 100], 'color': "darkred"}],
                'threshold' : {'line': {'color': "gray", 'width': 4}, 'thickness': 0.5, 'value': 50}}))

        st.write(figv)

    #i = dataframe['SK_ID_CURR'] == id_input
    #Y = dataframe[i]
    #Y = Y.drop(['SK_ID_CURR'], axis=1)

    # transformation des données
    #num = np.array(Y)

    # afficher les données
    st.subheader('Les données du client')
    st.write(id_input)
    st.write(Y)

    explainer1 = lime_tabular.LimeTabularExplainer(
        training_data=num,
        feature_names=data.columns,
        class_names=['Crédit Accepté', 'Crédit Refusé'],
        mode='classification',
        discretize_continuous=False
    )

    # interprétabilité
    st.subheader('Interprétabilité de notre prévision')
    st.markdown("##### Premième graphique")

    exp = explainer1.explain_instance(data_row=num[0], predict_fn=model.predict_proba)
    exp.show_in_notebook(show_table=True)

    components.html(exp.as_html(), height=800)
    plt.tight_layout()
    exp.as_list()

    st.markdown("##### Deuxième graphique")
    exp.show_in_notebook(show_table=True)
    fig = exp.as_pyplot_figure()
    # plt.tight_layout()
    st.write(fig)

    # graphique
    # tickers=Y.columns
    # print(tickers)
    st.sidebar.subheader('Exploration des caractéristiques du client')
    # dropdown=st.multiselect('Choisir une variable du client',tickers)
    select_box = st.sidebar.multiselect(label='Features', options=data.columns)
    plt.figure()
    fig1 = px.bar(data, x=select_box)
    st.plotly_chart(fig1)
    # fig1.show()
    st.sidebar.subheader("Exploration des caractéristiques de l'ensemble des clients")
    select_box1 = st.sidebar.multiselect(label='Features', options=dataframe.columns)
    fig2 = px.histogram(dataframe, x=select_box1, nbins=50)
    st.plotly_chart(fig2)
    # fig2.show()

    # graphique deux variables quantitatives
    st.sidebar.subheader("Analyse Bivariée des caractéristiques de nos clients")
    select_box2 = st.sidebar.selectbox(label='Axe des abscisses', options=dataframe.columns)
    select_box3 = st.sidebar.selectbox(label='Axe des ordonnées', options=dataframe.columns)
    fig3 = px.scatter(dataframe, x=select_box2, y=select_box3)
    st.plotly_chart(fig3)