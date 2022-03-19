from turtle import position
import streamlit as st
import pandas as pd
import dask.dataframe as dd
import numpy as np

import plotly.express as px
import chart_studio.plotly as py
import plotly.graph_objs as go

import leafmap.foliumap as leafmap

# from utils import load_parquets



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

    path = '../datasets/integrated_datas.parquet.gzip/part.0.parquet'

    # return pd.read_parquet(path)

    return dd.read_parquet('../datasets/integrated_datas.parquet.gzip', ignore_metadata_file=True)


ddf_original = load_data().copy(deep=False)

# Top text area
with st.container():
    st.title("Education Manager's Guide 📊")

left_column, right_column = st.columns(2)
with left_column:
    chart_type = st.selectbox("Choose the factor you would like to analyse", factors, 2)
with left_column:
    years = ddf_original.NU_ANO.astype(int).unique().compute()
    year = st.selectbox("Year", years, len(years)-1)

ddf = ddf_original[ddf_original['NU_ANO'] == year]

def our_plot(params, ddf, st):
    """ return plotly plots """

    if params["type"] == "geo":
        st.title('Geographical belongingness')


        ddf = pd.read_csv("../datasets/heatmap.csv")

        m = leafmap.Map(center=[-16, -50], tiles="stamentoner", zoom_start=5)
        m.add_heatmap(
            ddf,
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


        ddf = pd.read_csv("../datasets/heatmap.csv")

        m = leafmap.Map(center=[-16, -50], tiles="stamentoner", zoom_start=5)
        m.add_heatmap(
            ddf,
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
        # display data
        # with st.container():
        #     show_data = st.checkbox("See the raw data?")

        #     if show_data:
        #         ddf

    elif params["type"] == "pyramid":

        filtro = ddf.groupby(['SG_UF_RESIDENCIA', 'TP_SEXO', 'NU_ANO'])['NU_NOTA_MT'].mean().sort_index(ascending=False)#.compute()
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
        # display data
        # with st.container():
        #     show_data = st.checkbox("See the raw data?")

        #     if show_data:
        #         filtro

    elif params["type"] == "parallel":
        print(params)
        question = params["question"]
        print(question)
        df_par = ddf[[question, 'NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_LC', 'NU_NOTA_MT', 'NU_NOTA_REDACAO']].compute()
        # campos das notas da redação
        # df_par = ddf[[question, 'NU_NOTA_COMP1', 'NU_NOTA_COMP2', 'NU_NOTA_COMP3', 'NU_NOTA_COMP4', 'NU_NOTA_COMP5']].copy(deep=True).iloc[:1000]

        dict_names_exams = {'NU_NOTA_CN' : 'Ciências da Natureza',
                            'NU_NOTA_CH' : 'Ciências Humanas',
                            'NU_NOTA_LC' : 'Linguagens e Códigos',
                            'NU_NOTA_MT' : 'Matemática',
                            'NU_NOTA_REDACAO' : 'Redação',

                            'NU_NOTA_COMP1': 'Formal Writting', 
                            'NU_NOTA_COMP2': 'Topic Understanding', 
                            'NU_NOTA_COMP3': 'Conciseness/Organization', 
                            'NU_NOTA_COMP4': 'Linguistic Mechanisms', 
                            'NU_NOTA_COMP5': 'Respect for human rights',

                            'Q003': "Fahter's Profession",
                            
                            }

        groups_by_question = {'Q003': {'A' : 'Grupo 1: Lavrador, agricultor sem empregados, bóia fria, criador de animais (gado, porcos, galinhas, ovelhas, cavalos etc.), apicultor, pescador, lenhador, seringueiro, extrativista.',
                                        'B' : 'Grupo 2: Diarista, empregado doméstico, cuidador de idosos, babá, cozinheiro (em casas particulares), motorista particular, jardineiro, faxineiro de empresas e prédios, vigilante, porteiro, carteiro, office-boy, vendedor, caixa, atendente de loja, auxiliar administrativo, recepcionista, servente de pedreiro, repositor de mercadoria.',
                                        'C' : 'Grupo 3: Padeiro, cozinheiro industrial ou em restaurantes, sapateiro, costureiro, joalheiro, torneiro mecânico, operador de máquinas, soldador, operário de fábrica, trabalhador da mineração, pedreiro, pintor, eletricista, encanador, motorista, caminhoneiro, taxista.',
                                        'D' : 'Grupo 4: Professor (de ensino fundamental ou médio, idioma, música, artes etc.), técnico (de enfermagem, contabilidade, eletrônica etc.), policial, militar de baixa patente (soldado, cabo, sargento), corretor de imóveis, supervisor, gerente, mestre de obras, pastor, microempresário (proprietário de empresa com menos de 10 empregados), pequeno comerciante, pequeno proprietário de terras, trabalhador autônomo ou por conta própria.',
                                        'E' : 'Grupo 5: Médico, engenheiro, dentista, psicólogo, economista, advogado, juiz, promotor, defensor, delegado, tenente, capitão, coronel, professor universitário, diretor em empresas públicas ou privadas, político, proprietário de empresas com mais de 10 empregados.',
                                        'F' : 'Não sei.'}}

        if question == 'Q003':
            question = "Fahter's Profession" 
            dict_names_exams['Q003'] = question 
        if question == 'Q004':
            question = "Mother's Profession" 
            dict_names_exams['Q004'] = question 
        df_par = df_par.rename(columns=dict_names_exams)         
        df_par[question] = df_par[question].map({"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6, "H": 7})
        cmax = df_par[question].max()
        # n_answers = df_par.to_dask_array(lengths=True)[question].nunique().compute()
        # group_vals = list(range(1,n_answers))
        # group_names = [f'Group {num}' for num in range(1,n_answers)]
        dimensions = []
        # print(f"Columns: {df_par.columns}")
        # print(f"Question: {question}")
        # print(f"Groups: {group_names}")
        for column in df_par.columns:
            df_column = df_par[column]#.compute() 
            item = dict(range = [df_column.min(), df_column.max()],
                        label = column, 
                        values = list(df_column.values))
            # if column == question:
                # item['tickvals'] = group_vals
                # item['ticktext'] = group_names
            # print(item) 
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
        # display data
        # with st.container():
        #     show_data = st.checkbox("See the raw data?", key='data')

        #     if show_data:
        #         df_par

num_questions = len(titles_and_graphs[chart_type]["questions"])

if num_questions > 0:
    cols = st.columns(num_questions)
else:
    with st.container():
        plot = our_plot(titles_and_graphs[chart_type], ddf, st)

for index, question in enumerate(titles_and_graphs[chart_type]["questions"]):
    titles_and_graphs[chart_type]['question'] = question
    print(titles_and_graphs[chart_type])
    with cols[index]:
        if index == 0:
            st.subheader(f"Father")
        else:
            st.subheader(f"Mother")
        plot = our_plot(titles_and_graphs[chart_type], ddf, st)

