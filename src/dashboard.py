from turtle import position
import streamlit as st
import pandas as pd
import dask.dataframe as dd
import numpy as np

import plotly.express as px
import chart_studio.plotly as py
import plotly.graph_objs as go

import leafmap.foliumap as leafmap

from tempo import inicio, fim

# can only set this once, first thing to set
st.set_page_config(layout="wide")

titles_and_graphs = {
    "Geographical belongingness": {"type": 'geo', "questions": ""},
    "Age": {"type": None, "questions": "", "dimension": "age"},
    "Gender": {"type": "pyramid"},
    "Parents' education level": {"type": "pyramid", "questions": ["Q001", "Q002"]},
    "Marital status": {"type": None, "questions": ""},
    "Ethnicity": {"type": None, "questions": ""},
    "Father's education level": {"type": "parallel", "questions": ["Q001"]},
    "Mother's education level": {"type": "parallel", "questions": ["Q002"]},
    "Father's profession": {"type": "parallel", "questions": ["Q003"]},
    "Mother's profession": {"type": "parallel", "questions": ["Q004"]},
    "Income": {"type": 'cloro', "questions": ""},
    "Socioeconomic Status": {"type": None, "questions": ""},
}

factors = titles_and_graphs.keys()

# Top text area
with st.container():
    st.title("Education Manager's Guide 📊")

# column_1, column_2, column_3 = st.columns(3)
(column_1, column_2), test_data = st.columns(2), False
with column_1:
    chart_type = st.selectbox("Choose the factor you would like to analyse", factors, 2)
with column_2:
    years = list(range(2015,2021))
    year = st.selectbox("Year", years, len(years)-1)
# with column_3:
#     test_data = st.checkbox('Test Data', True)
# with column_3:
#     test_data = st.checkbox('Test Data', True)
test = '_test' if test_data else ''

print(f"Year: {year}")
print(f"Test data: {test_data}")

# Get 2015 data
@st.cache()
def load_2015_data():

    path = f'../datasets/integrated_datas_2015{test}.parquet.gzip'
    return dd.read_parquet(path, ignore_metadata_file=True)#, columns=cols_used)

# Get 2016 data
@st.cache()
def load_2016_data():

    path = f'../datasets/integrated_datas_2016{test}.parquet.gzip'
    return dd.read_parquet(path, ignore_metadata_file=True)#, columns=cols_used)

# Get 2017 data
@st.cache()
def load_2017_data():

    path = f'../datasets/integrated_datas_2017{test}.parquet.gzip'
    return dd.read_parquet(path, ignore_metadata_file=True)#, columns=cols_used)

# Get 2018 data
@st.cache()
def load_2018_data():

    path = f'../datasets/integrated_datas_2018{test}.parquet.gzip'
    return dd.read_parquet(path, ignore_metadata_file=True)#, columns=cols_used)

# Get 2019 data
@st.cache()
def load_2019_data():

    path = f'../datasets/integrated_datas_2019{test}.parquet.gzip'
    return dd.read_parquet(path, ignore_metadata_file=True)#, columns=cols_used)

# Get 2020 data
@st.cache()
def load_2020_data():

    path = f'../datasets/integrated_datas_2020{test}.parquet.gzip'
    return dd.read_parquet(path, ignore_metadata_file=True)#, columns=cols_used)


if year == 2020:
    ddf = load_2020_data()
elif year == 2019:
    ddf = load_2019_data()
elif year == 2018:
    ddf = load_2018_data()
elif year == 2017:
    ddf = load_2017_data()
elif year == 2016:
    ddf = load_2016_data()
elif year == 2015:
    ddf = load_2015_data()
else:
    ddf = load_2020_data()

def our_plot(params, ddf_par, st):
    """ return plotly plots """

    init = inicio()
    if params["type"] == "geo":
        st.title('Geographical belongingness')


        ddf_par = pd.read_csv("../datasets/heatmap.csv")

        m = leafmap.Map(center=[-16, -50], tiles="stamentoner", zoom_start=5)
        m.add_heatmap(
            ddf_par,
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


        ddf_par = pd.read_csv("../datasets/heatmap.csv")

        m = leafmap.Map(center=[-16, -50], tiles="stamentoner", zoom_start=5)
        m.add_heatmap(
            ddf_par,
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


        if 'questions' in params:
            q1 = params['questions'][1]
            q2 = params['questions'][0]
            filtro1 = ddf_par.groupby([q1, 'NU_ANO'])['NU_NOTA_LC'].mean().compute().sort_index(ascending=False)
            filtro2 = ddf_par.groupby([q2, 'NU_ANO'])['NU_NOTA_LC'].mean().compute().sort_index(ascending=False)
            women = filtro1.index.get_level_values(q1)
            men = filtro2.index.get_level_values(q2)
            women_bins = pd.Series(filtro1.values)
            men_bins = pd.Series(filtro2.values)
            women_bins *= -1
            y = men

            print(men_bins)
            print(women_bins)

            layout = go.Layout(yaxis=go.layout.YAxis(title='Languages and Codes Grades per Parents\' Education Level',
                                                     ),
                            xaxis=go.layout.XAxis(
                                range=[-700, 700],
                                tickvals=[-700, -350, 0, 350, 700],
                                ticktext=[700, 350, 0, 350, 700],
                                title='Mean Grade',
                                ),
                            barmode='overlay',
                            bargap=0.1, width=50, height=800)

            data_ = [go.Bar(y=y,
                        x=men_bins,
                        orientation='h',
                        name='Men',
                        hoverinfo='y',
                        text=men_bins.apply(lambda y: f"{y:.2f}"),
                        marker=dict(color='purple')
                        ),
                    go.Bar(y=y,
                        x=women_bins,
                        orientation='h',
                        name='Women',
                        text= women_bins.apply(lambda y: f"{(-1 * y):.2f}"),
                        hoverinfo='y',
                        marker=dict(color='seagreen')
                        )]

        else:

            filtro = ddf_par.groupby(['SG_UF_RESIDENCIA', 'TP_SEXO', 'NU_ANO'])['NU_NOTA_MT'].mean().compute().sort_index(ascending=False)
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
                        name='Fathers',
                        hoverinfo='y',
                        text=men_bins.astype('int'),
                        marker=dict(color='purple')
                        ),
                    go.Bar(y=y,
                        x=women_bins,
                        orientation='h',
                        name='Mothers',
                        text=-1 * women_bins.astype('int'),
                        hoverinfo='y',
                        marker=dict(color='seagreen')
                        )]
        fig = go.Figure(data=data_, layout=layout)
        st.plotly_chart(fig, use_container_width=True)

    elif params["type"] == "parallel":
        print(params)
        columns = params["questions"]
        questions = params["questions"].copy()
        columns.extend(['NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_LC', 'NU_NOTA_MT', 'NU_NOTA_REDACAO', 'NU_NOTA_SUM', 'NU_ANO'])
        print(f'Colunas: {columns}')
        # ddf_par =  dd.read_parquet(f'../datasets/integrated_datas_{year}{test}.parquet.gzip', ignore_metadata_file=True, columns=columns)
        # ddf_par =  pd.read_parquet(f'../datasets/integrated_top_grade_data{test}.parquet.gzip', columns=columns)
        ddf_par =  pd.read_parquet(f'../datasets/integrated_10000_top_grade_data.parquet.gzip', columns=columns)
        ddf_par = ddf_par[ddf_par.NU_ANO == year]
        print(f"Registros: {len(ddf_par)}")

        # ddf_par = ddf_par.compute()
        # campos das notas da redação   
        # df_par = ddf[[question, 'NU_NOTA_COMP1', 'NU_NOTA_COMP2', 'NU_NOTA_COMP3', 'NU_NOTA_COMP4', 'NU_NOTA_COMP5']].copy(deep=True).iloc[:1000]

        dict_names_exams = {'NU_NOTA_CN' : 'Ciências da Natureza',
                            'NU_NOTA_CH' : 'Ciências Humanas',
                            'NU_NOTA_LC' : 'Linguagens e Códigos',
                            'NU_NOTA_MT' : 'Matemática',
                            'NU_NOTA_REDACAO' : 'Redação',
                            'NU_NOTA_SUM' : 'Soma das Notas',

                            'NU_NOTA_COMP1': 'Formal Writting', 
                            'NU_NOTA_COMP2': 'Topic Understanding', 
                            'NU_NOTA_COMP3': 'Conciseness/Organization', 
                            'NU_NOTA_COMP4': 'Linguistic Mechanisms', 
                            'NU_NOTA_COMP5': 'Respect for human rights',

                            'Q001': "Fahter's Education Level",
                            'Q002': "Mother's Education Level",

                            'Q003': "Fahter's Profession",
                            'Q004': "Mother's Profession",
                            
                            }

        groups_by_question = {'Q003': {'A' : 'Grupo 1: Lavrador, agricultor sem empregados, bóia fria, criador de animais (gado, porcos, galinhas, ovelhas, cavalos etc.), apicultor, pescador, lenhador, seringueiro, extrativista.',
                                        'B' : 'Grupo 2: Diarista, empregado doméstico, cuidador de idosos, babá, cozinheiro (em casas particulares), motorista particular, jardineiro, faxineiro de empresas e prédios, vigilante, porteiro, carteiro, office-boy, vendedor, caixa, atendente de loja, auxiliar administrativo, recepcionista, servente de pedreiro, repositor de mercadoria.',
                                        'C' : 'Grupo 3: Padeiro, cozinheiro industrial ou em restaurantes, sapateiro, costureiro, joalheiro, torneiro mecânico, operador de máquinas, soldador, operário de fábrica, trabalhador da mineração, pedreiro, pintor, eletricista, encanador, motorista, caminhoneiro, taxista.',
                                        'D' : 'Grupo 4: Professor (de ensino fundamental ou médio, idioma, música, artes etc.), técnico (de enfermagem, contabilidade, eletrônica etc.), policial, militar de baixa patente (soldado, cabo, sargento), corretor de imóveis, supervisor, gerente, mestre de obras, pastor, microempresário (proprietário de empresa com menos de 10 empregados), pequeno comerciante, pequeno proprietário de terras, trabalhador autônomo ou por conta própria.',
                                        'E' : 'Grupo 5: Médico, engenheiro, dentista, psicólogo, economista, advogado, juiz, promotor, defensor, delegado, tenente, capitão, coronel, professor universitário, diretor em empresas públicas ou privadas, político, proprietário de empresas com mais de 10 empregados.',
                                        'F' : 'Não sei.'}}

        for question in questions:     
            print(question)
            ddf_par[question] = ddf_par[question].map({"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6, "H": 7})
        ddf_par = ddf_par.rename(columns=dict_names_exams)    
        # print(ddf_par[['Ciências da Natureza', 'Ciências Humanas', 'Linguagens e Códigos', 'Matemática']])
        question = dict_names_exams[questions[0]]
        cmax = ddf_par[question].max()#.compute()
        # n_answers = df_par.to_dask_array(lengths=True)[question].nunique().compute()
        # group_vals = list(range(1,n_answers))
        # group_names = [f'Group {num}' for num in range(1,n_answers)]
        dimensions = []
        # print(f"Columns: {df_par.columns}")
        # print(f"Question: {question}")
        # print(f"Groups: {group_names}")
        ddf_par.drop(columns=['NU_ANO'], inplace=True)
        for column in ddf_par.columns:
            df_column = ddf_par[column]#.compute()
            min, max = df_column.min(), df_column.max()
            if column == 'NU_ANO':
                min = 2015
                max = 2020
            print(f"Column: {column}")
            print(f"min: {min}")
            print(f"max: {max}")
            item = dict(range = [min, max],
                        label = column, 
                        values = list(df_column.values))
            # if column == question:
                # item['tickvals'] = group_vals
                # item['ticktext'] = group_names
            # print(item) 
            dimensions.append(item)
            
        
        fig = go.Figure(data=
            go.Parcoords(
                line = dict(color = ddf_par[question],
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

    duration = fim(init)
    print(f"Duração: {duration}\n")

with st.container():
    plot = our_plot(titles_and_graphs[chart_type], ddf, st)
