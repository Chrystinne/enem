from turtle import position
import streamlit as st
import pandas as pd
import numpy as np

import plotly.express as px
import chart_studio.plotly as py
import plotly.graph_objs as go

import leafmap.foliumap as leafmap

# can only set this once, first thing to set
st.set_page_config(layout="wide")

titles_and_graphs = {
    "1 - Geographical belongingness": {"type": 'geo', "questions": ""},
    "2 - Age": {"type": None, "questions": ""},
    "3 - Gender": {"type": "pyramid", "questions": ""},
    "4 - Marital status": {"type": None, "questions": ""},
    "5 - Ethnicity": {"type": None, "questions": ""},
    "6 - Parents's education level": {"type": "parallel", "questions": ["Q001", "Q002"]},
    "7 - Parents's profession": {"type": "parallel", "questions": ["Q003", "Q004"]},
    "8 - Income": {"type": 'cloro', "questions": ""},
    "9 - Socioeconomic Status": {"type": None, "questions": ""},
}

factors = titles_and_graphs.keys()

# Get data
@st.cache()
def load_data():
    return pd.read_csv("../datasets/MICRODADOS_ENEM_2018.csv", sep=';', encoding='cp1252', nrows=10000)

df = load_data().copy(deep=True)

# Top text area
with st.container():
    st.title("Education Manager's Guide 📊")

# User choose type
chart_type = st.selectbox("Choose the factor you would like to analyse", factors, 2)

def our_plot(params, df, st):
    """ return plotly plots """

    if params["type"] == "geo":
        st.title('Geographical belongingness')


        filepath = pd.read_csv("../datasets/heatmap.csv")

        m = leafmap.Map(center=[-16, -50], tiles="stamentoner", zoom_start=5)
        m.add_heatmap(
            filepath,
            "latitude",
            "longitude",
            "Q006",
            # "TP_SEXO",
            # "NU_IDADE",
            # "NU_ANO",
            # name="Heat map",
            radius=15,

        )
        m.to_streamlit(width=700, height=1000)
    if params["type"] == "cloro":
        st.title('Geographical belongingness')


        filepath = pd.read_csv("../datasets/heatmap.csv")

        m = leafmap.Map(center=[-16, -50], tiles="stamentoner", zoom_start=5)
        m.add_heatmap(
            filepath,
            "latitude",
            "longitude",
            "Q006",
            # "TP_SEXO",
            # "NU_IDADE",
            # "NU_ANO",
            # name="Heat map",
            radius=15,

        )
        m.to_streamlit(width=700, height=1000)

    elif params["type"] == "pyramid":

        filtro = df.groupby(['SG_UF_RESIDENCIA', 'TP_SEXO'])['NU_NOTA_MT'].mean().sort_index(ascending=False)
        women = filtro[filtro.index.get_level_values('TP_SEXO').isin(['F'])]
        men = filtro[filtro.index.get_level_values('TP_SEXO').isin(['M'])]
        estados = men.index.get_level_values(0)
        men_bins = np.array(men)
        women_bins = np.array(women)
        women_bins *= -1
        y = estados

        layout = go.Layout(yaxis=go.layout.YAxis(title='Mathematics Grades per State'),
                        xaxis=go.layout.XAxis(
                            range=[-1000, 1000],
                            tickvals=[-1000, -700, -300, 0, 300, 700, 1000],
                            ticktext=[1000, 700, 300, 0, 300, 700, 1000],
                            title='Mean Grade'),
                        barmode='overlay',
                        bargap=0.1, width=50, height=800)

        data_ = [go.Bar(y=y,
                    x=men_bins,
                    orientation='h',
                    name='Men',
                    hoverinfo='y',
                    text=men_bins.astype('int'),
                    marker=dict(color='purple')
                    ),
                go.Bar(y=y,
                    x=women_bins,
                    orientation='h',
                    name='Women',
                    text=-1 * women_bins.astype('int'),
                    hoverinfo='y',
                    marker=dict(color='seagreen')
                    )]

        st.plotly_chart({"data":data_, "layout":layout}, use_container_width=True)

    elif params["type"] == "parallel":
        print(params)
        question = params["question"]
        print(question)
        df_par = df[[question, 'NU_NOTA_COMP1', 'NU_NOTA_COMP2', 'NU_NOTA_COMP3', 'NU_NOTA_COMP4', 'NU_NOTA_COMP5']]
        df_par[question] = df_par[question].map({"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6, "H": 7})
        cmax = df_par[question].max()
        dimensions = []
        for column in df_par.columns:
            item = dict(range = [df_par[column].min(), df_par[column].max()],
                        label = column, 
                        values = df_par[column])
            dimensions.append(item)
            
        fig = go.Figure(data=
            go.Parcoords(
                # line_color='yellow',
                line = dict(color = df_par[question],
                        colorscale = [[0,'green'],[0.25,'blue'],[0.5,'red'],[1,'gold']],
                        showscale = True,
                        cmin = 0,
                        cmax = cmax),
                dimensions = dimensions
            )
        )
        st.plotly_chart(fig, use_container_width=True)

num_questions = len(titles_and_graphs[chart_type]["questions"])

if num_questions > 0:
    cols = st.columns(num_questions)
else:
    with st.container():
        plot = our_plot(titles_and_graphs[chart_type], df, st)

for index, question in enumerate(titles_and_graphs[chart_type]["questions"]):
    titles_and_graphs[chart_type]['question'] = question
    print(titles_and_graphs[chart_type])
    with cols[index]:
        if index == 0:
            st.subheader(f"Father")
        else:
            st.subheader(f"Mother")
        plot = our_plot(titles_and_graphs[chart_type], df)

# display data
with st.container():
    show_data = st.checkbox("See the raw data?")

    if show_data:
        df