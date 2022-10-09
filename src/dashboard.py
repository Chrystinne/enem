from turtle import position
from matplotlib.pyplot import margins
import streamlit as st
import pandas as pd
import dask.dataframe as dd
import numpy as np
from math import floor

import plotly.express as px
import chart_studio.plotly as py
import plotly.graph_objs as go

# import leafmap.foliumap as leafmap

from tempo import inicio, fim

# can only set this once, first thing to set
st.set_page_config(layout="wide")

titles_and_graphs = {
    # "Geographical belongingness": {"type": 'geo', "questions": ""},
    "Income": {"type": 'bar_income'},
    # "Age": {"type": 'bar_age'},
    "Gender": {"title": "Gender", "type": "bar_gender"},
    # "Parents' education level": {"title": "Parents' education level", "type": "pyramid", "questions": ["Q001", "Q002"]},
    # "Parents' profession": {"title": "Parents' profession", "type": "pyramid", "questions": ["Q003", "Q004"]},
    "Marital status": {"title": "Marital_status", "type": "bar_marital_status"},
    "Ethnicity": {"title": "Ethnicity", "type": "bar_ethnicity"},
    "Father's education level": {"type": "parallel", "questions": ["Q001"]},
    "Mother's education level": {"type": "parallel", "questions": ["Q002"]},
    "Father's profession": {"type": "parallel", "questions": ["Q003"]},
    "Mother's profession": {"type": "parallel", "questions": ["Q004"]},
    # "Socioeconomic Status": {"type": None, "questions": ""},
}

grades_names_to_columns = {"Mathematics": "NU_NOTA_MT", 
                     "Languages and Codes": "NU_NOTA_LC", 
                     "Human Sciences": "NU_NOTA_CH", 
                     "Nature Sciences": "NU_NOTA_CN",
                     "Essay": "NU_NOTA_REDACAO"}

columns_to_grades_names = {item[1]: item[0] for item in grades_names_to_columns.items()}                     

factors = titles_and_graphs.keys()
grades = grades_names_to_columns.keys()

# Top text area
with st.container():
    # st.title("EduVizBR 📊")
    st.title("EduVizBR")

# column_1, column_2, column_3 = st.columns(3)
(column_1, column_2, column_3), test_data = st.columns(3), False
with column_1:
    chart_type = st.selectbox("Choose the factor you would like to analyse", factors, 2)
with column_2:
    years = list(range(2015,2021))
    year = st.selectbox("Year", years, len(years)-1)
with column_3:
    grade = st.selectbox("Grade", grades, len(grades)-1)
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
            filtro1 = ddf_par.groupby([q1, 'NU_ANO'])[grades_names_to_columns[grade]].mean().compute().sort_index(ascending=False)
            filtro2 = ddf_par.groupby([q2, 'NU_ANO'])[grades_names_to_columns[grade]].mean().compute().sort_index(ascending=False)
            women = filtro1.index.get_level_values(q1)[1:]
            men = filtro2.index.get_level_values(q2)[1:]
            women_bins = pd.Series(filtro1.values)[1:]
            men_bins = pd.Series(filtro2.values)[1:]
            
            y = men

            max_men_value = floor(men_bins.max())
            min_men_value = floor(men_bins.min())
            max_women_value = floor(women_bins.max())
            min_women_value = floor(women_bins.min())
            women_bins *= -1

            print(men_bins)
            print(women_bins)

            layout = go.Layout(yaxis=go.layout.YAxis(title=params['title']),
                            xaxis=go.layout.XAxis(
                                range=[-710, 710],
                                tickvals=[-1*max_women_value, -1*min_women_value, 0, min_men_value, max_men_value],
                                ticktext=[max_women_value, min_women_value, 0, min_men_value, max_men_value],
                                tickwidth=4,
                                title=f'Mean {grade} Grades in {year}',
                                ),
                            barmode='overlay',
                            bargap=0.1, width=50, height='400px')

            data_ = [go.Bar(y=y,
                        x=men_bins,
                        orientation='h',
                        name='Fathers',
                        hoverinfo='y',
                        text=men_bins.apply(lambda y: f"{y:.2f}"),
                        marker=dict(color='blue')
                        ),
                    go.Bar(y=y,
                        x=women_bins,
                        orientation='h',
                        name='Mothers',
                        text= women_bins.apply(lambda y: f"{(-1 * y):.2f}"),
                        hoverinfo='y',
                        marker=dict(color='seagreen')
                        )]

    elif params["type"] == "pizza_graph":
        fig = go.Figure(go.Pie(
        values = [40000000, 20000000, 30000000, 10000000],
        labels = ["Wages", "Operating expenses", "Cost of sales", "Insurance"],
        texttemplate = "%{label}: %{value:$,s} <br>(%{percent})"
        # ,textposition = "inside"
        ))
        fig.update_layout(legend=dict(yanchor="top", y=1.49, xanchor="left", x=0.01))
        fig.show()
    
    elif params["type"] == "bar_gender":

        filtro = ddf_par.groupby(['SG_UF_RESIDENCIA', 'TP_SEXO'])[grades_names_to_columns[grade]].mean().reset_index().compute()
        
        women = filtro[filtro.TP_SEXO == 'F'][grades_names_to_columns[grade]]
        men = filtro[filtro.TP_SEXO == 'M'][grades_names_to_columns[grade]]
        dif = (men.values - women.values)
        estados = filtro.iloc[men.index]['SG_UF_RESIDENCIA'].values

        filtro = pd.DataFrame({'men': men.values, 'women': women.values, 'estados': estados, 'dif': dif}).sort_values('dif').reset_index(drop=True)

        layout = go.Layout(yaxis=go.layout.YAxis(title=f'Mean Grades of {grade} in {year}',),
                           xaxis=go.layout.XAxis(title="Brazilian States"),
                        #    barmode='overlay',
                           bargap=0.1, width=1000, height=550,
                           legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))

        data_ = [go.Bar(y=filtro.men,
                    x=filtro.estados,
                    name='Men',
                    hoverinfo='x+name+y',
                    text=filtro.men.apply(lambda y: f"{y:.0f}"),
                    marker=dict(color='#7eb0d5'),
                    textfont=dict(family="Arial",
                                  size=15),
                    textposition='outside'
                    ),
                go.Bar(y=filtro.women,
                    x=filtro.estados,
                    name='Women',
                    text=filtro.women.apply(lambda y: f"{y:.0f}"),
                    hoverinfo='x+name+y',
                    marker=dict(color='#fd7f6f'),
                    textfont=dict(family="Arial",
                                  size=15),
                    textposition='outside'
                    ),
                go.Scatter(x=filtro.estados, 
                           y=filtro.dif, 
                           hoverinfo='x+name+y',
                           name='Difference',
                           textfont=dict(color='black',
                                            family="Arial",
                                                    size=15),
                           text=filtro.dif.apply(lambda y: f"{y:.0f}"), 
                           marker=dict(color='red'),
                           mode='lines+markers+text',
                           textposition='top center')
                ]
        fig = go.Figure(data=data_, layout=layout)
        fig.update_layout(barmode='group', font=dict(size=10, family="Arial", color="black"))
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightPink')
        fig['layout']['xaxis']['titlefont'] = dict(size=14)
        fig['layout']['xaxis']['tickfont'] = dict(size=12)
        fig['layout']['yaxis']['titlefont'] = dict(size=14)
        fig['layout']['yaxis']['tickfont'] = dict(size=12)
        fig['layout']['legend']['font'] = dict(size=12)
        st.plotly_chart(fig, use_container_width=True)

    elif params["type"] == "bar_marital_status":

        filtro = ddf_par.groupby(['SG_UF_RESIDENCIA', 'TP_ESTADO_CIVIL'])[grades_names_to_columns[grade]].mean().reset_index().compute()
        single = filtro[filtro.TP_ESTADO_CIVIL == 1][grades_names_to_columns[grade]]        
        married = filtro[filtro.TP_ESTADO_CIVIL == 2][grades_names_to_columns[grade]]
        divorced = filtro[filtro.TP_ESTADO_CIVIL == 3][grades_names_to_columns[grade]]
        estados = filtro.iloc[single.index]['SG_UF_RESIDENCIA'].values

        filtro = pd.DataFrame({'single': single.values, 'married': married.values, 'divorced': divorced.values, 'estados': estados}).reset_index(drop=True)

        layout = go.Layout(yaxis=go.layout.YAxis(title=f'Mean Grades of {grade} in {year}',
                                                 ),
                           xaxis=go.layout.XAxis(title="Brazilian States"),
                        #    barmode='overlay',
                           bargap=0.1, width=1000, height=550,
                           legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))

        data_ = [go.Bar(y=filtro.single,
                    x=filtro.estados,
                    name='Single',
                    hoverinfo='x+name+y',
                    text=filtro.single.apply(lambda y: f"{y:.0f}"),
                    marker=dict(color='#fd7f6f'),
                    textfont=dict(family="Arial",
                                  size=80),
                    textposition='outside'
                    ),
                go.Bar(y=filtro.married,
                    x=filtro.estados,
                    name='Married',
                    text=filtro.married.apply(lambda y: f"{y:.0f}"),
                    hoverinfo='x+name+y',
                    marker=dict(color='#7eb0d5'),
                    textfont=dict(family="Arial",
                                  size=60),
                    textposition='outside'
                    ),
                go.Bar(y=filtro.divorced,
                    x=filtro.estados,
                    name='Divorced',
                    text=filtro.divorced.apply(lambda y: f"{y:.0f}"),
                    hoverinfo='x+name+y',
                    marker=dict(color='#b2e061'),
                    textfont=dict(family="Arial",
                                  size=60),
                    textposition='outside'
                    ),
                ]
        fig = go.Figure(data=data_, layout=layout)
        fig.update_layout(barmode='group', font=dict(size=80, family="Arial", color="black"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightPink')
        fig['layout']['xaxis']['titlefont'] = dict(size=14)
        fig['layout']['xaxis']['tickfont'] = dict(size=12)
        fig['layout']['yaxis']['titlefont'] = dict(size=14)
        fig['layout']['yaxis']['tickfont'] = dict(size=12)
        fig['layout']['legend']['font'] = dict(size=12)
        st.plotly_chart(fig, use_container_width=True)

    elif params["type"] == "bar_ethnicity":

        filtro = ddf_par.groupby(['SG_UF_RESIDENCIA', 'TP_COR_RACA'])[grades_names_to_columns[grade]].mean().reset_index().compute()
        white = filtro[filtro.TP_COR_RACA == 1][grades_names_to_columns[grade]]        
        black = filtro[filtro.TP_COR_RACA == 2][grades_names_to_columns[grade]]
        brown = filtro[filtro.TP_COR_RACA == 3][grades_names_to_columns[grade]]
        yellow = filtro[filtro.TP_COR_RACA == 4][grades_names_to_columns[grade]]        
        indigenous = filtro[filtro.TP_COR_RACA == 5][grades_names_to_columns[grade]]
        estados = filtro.iloc[white.index]['SG_UF_RESIDENCIA'].values

        filtro = pd.DataFrame({'white': white.values, 'black': black.values, 'brown': brown.values, 'yellow': yellow.values, 'indigenous': indigenous.values, 'estados': estados}).reset_index(drop=True)

        layout = go.Layout(yaxis=go.layout.YAxis(title=f'Mean Grades of {grade} in {year}',
                                                 ),
                           xaxis=go.layout.XAxis(title="Brazilian States"),
                        #    barmode='overlay',
                           bargap=0.1, width=1000, height=550)
        #paleta
        # "#fd7f6f", "#7eb0d5", "#b2e061", "#bd7ebe", "#ffb55a", "#ffee65", "#beb9db", "#fdcce5", "#8bd3c7"
        data_ = [go.Bar(y=filtro.white,
                    x=filtro.estados,
                    name='White',
                    hoverinfo='x+name+y',
                    text=filtro.white.apply(lambda y: f"{y:.0f}"),
                    marker=dict(color='#fd7f6f'),
                    textfont=dict(family="Arial",
                                  size=80),
                    textposition='outside'
                    ),
                go.Bar(y=filtro.black,
                    x=filtro.estados,
                    name='Black',
                    text=filtro.black.apply(lambda y: f"{y:.0f}"),
                    hoverinfo='x+name+y',
                    marker=dict(color='#7eb0d5'),
                    textfont=dict(family="Arial",
                                  size=60),
                    textposition='outside'
                    ),
                go.Bar(y=filtro.brown,
                    x=filtro.estados,
                    name='Brown',
                    text=filtro.brown.apply(lambda y: f"{y:.0f}"),
                    hoverinfo='x+name+y',
                    marker=dict(color='#b2e061'),
                    textfont=dict(family="Arial",
                                  size=60),
                    textposition='outside'
                    ),
                go.Bar(y=filtro.yellow,
                    x=filtro.estados,
                    name='Yellow',
                    text=filtro.yellow.apply(lambda y: f"{y:.0f}"),
                    hoverinfo='x+name+y',
                    marker=dict(color='#bd7ebe'),
                    textfont=dict(family="Arial",
                                  size=60),
                    textposition='outside'
                    ),
                go.Bar(y=filtro.indigenous,
                    x=filtro.estados,
                    name='Indigenous',
                    text=filtro.indigenous.apply(lambda y: f"{y:.0f}"),
                    hoverinfo='x+name+y',
                    marker=dict(color='#ffb55a'),
                    textfont=dict(family="Arial",
                                  size=60),
                    textposition='outside'
                    ),
                ]
        fig = go.Figure(data=data_, layout=layout)
        fig.update_layout(barmode='group', font=dict(size=10, family="Arial", color="black"), 
            # legend=dict(yanchor="top", y=1.49, xanchor="left", x=0.01)
            legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightPink')
        fig['layout']['xaxis']['titlefont'] = dict(size=14)
        fig['layout']['xaxis']['tickfont'] = dict(size=12)
        fig['layout']['yaxis']['titlefont'] = dict(size=14)
        fig['layout']['yaxis']['tickfont'] = dict(size=12)
        fig['layout']['legend']['font'] = dict(size=12)
        st.plotly_chart(fig, use_container_width=True)

    elif params["type"] == "bar_income":

        filtro = ddf_par.groupby(['SG_UF_RESIDENCIA', 'TP_SES_INCOME'])[grades_names_to_columns[grade]].mean().reset_index().compute()
        a_class = filtro[filtro.TP_SES_INCOME == 'A'][grades_names_to_columns[grade]]
        b_class = filtro[filtro.TP_SES_INCOME == 'B'][grades_names_to_columns[grade]]        
        c_class = filtro[filtro.TP_SES_INCOME == 'C'][grades_names_to_columns[grade]]
        d_class = filtro[filtro.TP_SES_INCOME == 'D'][grades_names_to_columns[grade]]
        e_class = filtro[filtro.TP_SES_INCOME == 'E'][grades_names_to_columns[grade]]        
        # f_class = filtro[filtro.TP_SES_INCOME == 'F'][grades_names_to_columns[grade]]
        estados = filtro.iloc[b_class.index]['SG_UF_RESIDENCIA'].values

        filtro = pd.DataFrame({'a_class': a_class.values, 'b_class': b_class.values, 'c_class': c_class.values, 'd_class': d_class.values, 'e_class': e_class.values, 'estados': estados}).reset_index(drop=True)

        layout = go.Layout(yaxis=go.layout.YAxis(title=f'Mean Grades of {grade} in {year}',
                                                 ),
                           xaxis=go.layout.XAxis(title="Brazilian States"),
                        #    barmode='overlay',
                           bargap=0.1, width=1000, height=550)

        data_ = [go.Bar(y=filtro.a_class,
                    x=filtro.estados,
                    name='A Class (up to 20 MS)',
                    text=filtro.a_class.apply(lambda y: f"{y:.0f}"),
                    hoverinfo='x+name+y',
                    marker=dict(color='#fd7f6f'),
                    textfont=dict(family="Arial",
                                  size=60),
                    textposition='outside'
                    ),
                go.Bar(y=filtro.b_class,
                    x=filtro.estados,
                    name='B Class (10-20 MS)',
                    hoverinfo='x+name+y',
                    text=filtro.b_class.apply(lambda y: f"{y:.0f}"),
                    marker=dict(color='#7eb0d5'),
                    textfont=dict(family="Arial",
                                  size=80),
                    textposition='outside'
                    ),
                go.Bar(y=filtro.c_class,
                    x=filtro.estados,
                    name='C Class (4-10 MS)',
                    text=filtro.c_class.apply(lambda y: f"{y:.0f}"),
                    hoverinfo='x+name+y',
                    marker=dict(color='#b2e061'),
                    textfont=dict(family="Arial",
                                  size=60),
                    textposition='outside'
                    ),
                go.Bar(y=filtro.d_class,
                    x=filtro.estados,
                    name='D Class (2-4 MS)',
                    text=filtro.d_class.apply(lambda y: f"{y:.0f}"),
                    hoverinfo='x+name+y',
                    marker=dict(color='#bd7ebe'),
                    textfont=dict(family="Arial",
                                  size=60),
                    textposition='outside'
                    ),
                go.Bar(y=filtro.e_class,
                    x=filtro.estados,
                    name='E Class (up to 2 MS)',
                    text=filtro.e_class.apply(lambda y: f"{y:.0f}"),
                    hoverinfo='x+name+y',
                    marker=dict(color='#ffb55a'),
                    textfont=dict(family="Arial",
                                  size=60),
                    textposition='outside'
                    ),
                ]

        fig = go.Figure(data=data_, layout=layout)
        fig.update_layout(barmode='group', font=dict(size=10, family="Arial", color="black"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='LightPink')
        fig['layout']['xaxis']['titlefont'] = dict(size=14)
        fig['layout']['xaxis']['tickfont'] = dict(size=12)
        fig['layout']['yaxis']['titlefont'] = dict(size=14)
        fig['layout']['yaxis']['tickfont'] = dict(size=12)
        fig['layout']['legend']['font'] = dict(size=12)
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

                            'Q001': "Father's Education Level",
                            'Q002': "Mother's Education Level",

                            'Q003': "Father's Profession",
                            'Q004': "Mother's Profession",
                            
                            }

        legend_education_level  = "1- No study\n2- Incomplete primary school\n3- Primary school\n4- Secondary school\n5- High school\n6- Graduated\n7- Post graduated\n8- Unknown"

        groups_by_education_level = {'Q001': {
                                        'A' : 'No study',
                                        'B' : 'Incomplete proimary school',
                                        'C' : 'Primary school',
                                        'D' : 'Secondary school',
                                        'E' : 'High school',
                                        'F' : 'Graduated',
                                        'G' : 'Post graduated',
                                        'H' : 'Unknow'
                                    }}

        legend_parents_profession  = "1- Farmer, fisherman/fisherwoman etc\n2- Elderly caregiver, doorman/portress, salesperson etc\n3- Baker, jeweler, painter, electrician, plumber, driver etc\n4- Professor, technician, police etc\n5- Physician, engineer, judge, lawyer etc\n6- Unknown"

        groups_by_question = {'Q003': {'A' : 'Grupo 1: Lavrador, agricultor sem empregados, bóia fria, criador de animais (gado, porcos, galinhas, ovelhas, cavalos etc.), apicultor, pescador, lenhador, seringueiro, extrativista.',
                                        'B' : 'Grupo 2: Diarista, empregado doméstico, cuidador de idosos, babá, cozinheiro (em casas particulares), motorista particular, jardineiro, faxineiro de empresas e prédios, vigilante, porteiro, carteiro, office-boy, vendedor, caixa, atendente de loja, auxiliar administrativo, recepcionista, servente de pedreiro, repositor de mercadoria.',
                                        'C' : 'Grupo 3: Padeiro, cozinheiro industrial ou em restaurantes, sapateiro, costureiro, joalheiro, torneiro mecânico, operador de máquinas, soldador, operário de fábrica, trabalhador da mineração, pedreiro, pintor, eletricista, encanador, motorista, caminhoneiro, taxista.',
                                        'D' : 'Grupo 4: Professor (de ensino fundamental ou médio, idioma, música, artes etc.), técnico (de enfermagem, contabilidade, eletrônica etc.), policial, militar de baixa patente (soldado, cabo, sargento), corretor de imóveis, supervisor, gerente, mestre de obras, pastor, microempresário (proprietário de empresa com menos de 10 empregados), pequeno comerciante, pequeno proprietário de terras, trabalhador autônomo ou por conta própria.',
                                        'E' : 'Grupo 5: Médico, engenheiro, dentista, psicólogo, economista, advogado, juiz, promotor, defensor, delegado, tenente, capitão, coronel, professor universitário, diretor em empresas públicas ou privadas, político, proprietário de empresas com mais de 10 empregados.',
                                        'F' : 'Não sei.'}}

        for question in questions:     
            print(question)
            # ddf_par[question] = ddf_par[question].map({"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6, "H": 7})
            if (questions[0] == 'Q001' or questions[0] == 'Q002'):
                ddf_par[question] = ddf_par[question].map({"H": 1, "A": 2, "B": 3, "C": 4, "D": 5, "E": 6, "F": 7, "G": 8})
            elif (questions[0] == 'Q003' or questions[0] == 'Q004'):
                ddf_par[question] = ddf_par[question].map({"F": 1, "A": 2, "B": 3, "C": 4, "D": 5, "E": 6})
            
        ddf_par = ddf_par.rename(columns=dict_names_exams)    
        # print(ddf_par[['Ciências da Natureza', 'Ciências Humanas', 'Linguagens e Códigos', 'Matemática']])
        question = dict_names_exams[questions[0]]
        print(question)
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
                        colorscale = [[0,'#7eb0d5'],[0.2,'#b2e061'],[0.4,'#bd7ebe'],[0.6,'#ffb55a'], [0.8,'#beb9db'], [1,'#fd7f6f']],
                        showscale = True,
                        cmin = 0,   
                        cmax = cmax),
                dimensions = dimensions
            )
        )
        st.plotly_chart(fig, use_container_width=True)
        print(dict_names_exams[questions[0]])
        if (questions[0] == 'Q001' or questions[0] == 'Q002'):
            st.text(legend_education_level)
        elif (questions[0] == 'Q003' or questions[0] == 'Q004'):
            st.text(legend_parents_profession)
        # with st.container():
            

    duration = fim(init)
    print(f"Duração: {duration}\n")

with st.container():
    plot = our_plot(titles_and_graphs[chart_type], ddf, st)
