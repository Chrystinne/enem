from turtle import position
from matplotlib.pyplot import margins
import streamlit as st
import pandas as pd
import dask.dataframe as dd
import numpy as np
from math import floor
from PIL import Image
import plotly.express as px
import chart_studio.plotly as py
import plotly.graph_objs as go
from turtle import color
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import ranksums
from plotly.subplots import make_subplots
from plotly.offline import plot
# import leafmap.foliumap as leafmap

from tempo import inicio, fim

# can only set this once, first thing to set
st.set_page_config(layout="wide")

titles_and_graphs = {
    # "Geographical belongingness": {"type": 'geo', "questions": ""},
    "Marital status": {"title": "Marital_status", "type": "bar_marital_status"},
    "Gender": {"title": "Gender", "type": "bar_gender"},
    "Ethnicity": {"title": "Ethnicity", "type": "bar_ethnicity"},
    "Income": {"type": 'bar_income'},
    # "Age": {"type": 'bar_age'},
    # "Parents' education level": {"title": "Parents' education level", "type": "pyramid", "questions": ["Q001", "Q002"]},
    # "Parents' profession": {"title": "Parents' profession", "type": "pyramid", "questions": ["Q003", "Q004"]},
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

grades_names_to_columns = {"Mathematics": "NU_NOTA_MT", 
                     "Languages and Codes": "NU_NOTA_LC", 
                     "Human Sciences": "NU_NOTA_CH", 
                     "Nature Sciences": "NU_NOTA_CN",
                     "Essay": "NU_NOTA_REDACAO"}

factors = titles_and_graphs.keys()
grades = grades_names_to_columns.keys()
image = Image.open('../images/Logo15.png')

col1, col2, col3, col4 = st.columns([2,0.87,2,2])

with col1:
    st.write(' ')

with col2:
    st.write('  ')

with col3:
    st.write(' ')

with col4:
    st.write(' ')

# Top text area
with st.container():
    with col3:
        st.image(image, width=180)

(column_1, column_2, column_3, column_4), test_data = st.columns(4), False
with column_1:
    chart_type = st.selectbox("Choose the factor you would like to analyse", factors, 0)
with column_2:
    years = list(range(2015,2021))
    year = st.selectbox("Year", years, len(years)-1)
with column_3:
    grade = st.selectbox("Grade", grades, len(grades)-1)
with column_4:
    brazilian_states = ['RS','PB','BA','AL','PA','TO','SP','CE','AM','SE','MG','MA','PI',
                            'PE','MT','RJ','GO','RN','ES','AP','DF','SC','PR','RR','RO','MS','AC','ALL STATES']
    brazilian_state = st.selectbox("State", brazilian_states, 0)
    show_statistical_test = st.checkbox('Show statiscal tests', False)
# with column_3:
#     test_data = st.checkbox('Test Data', True)
test = '_test' if test_data else ''

print(f"Year: {year}")
# print(f"Test data: {test_data}")
print(f"Brazilian state: {brazilian_state}")

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

def graficoSexo(data):
    colors = ['#fd7f6f', '#7eb0d5']
    fig = go.Figure(go.Pie(
        values = data['TP_SEXO'].value_counts().compute().sort_index(),
        labels = ['F', 'M'],
        texttemplate = "%{label}: %{value:,s} <br>(%{percent})",
        ))
    fig.update_layout(
        title=f'Students {chart_type} in {year}', # title of plot
        font=dict(
            size=13,
        )
    )
    fig.update_traces(marker=dict(colors=colors))   
    fig.update_layout(legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig, use_container_width=True)

# def ploting_distribution_female_male_per_state_old(data, state, course):
#     mulher = data[(data['TP_SEXO'] == 'F') & (data['SG_UF_RESIDENCIA'] == state)]
#     homem = data[(data['TP_SEXO'] == 'M') & (data['SG_UF_RESIDENCIA'] == state)]

#     fig = go.Figure()
#     fig.add_trace(go.Histogram(x=mulher[course]))
#     fig.add_trace(go.Histogram(x=homem[course]))

#     fig.update_layout(barmode='overlay')
#     # Reduce opacity to see both histograms
#     fig.update_traces(opacity=0.75)
#     fig.show()

def ploting_boxplot_gender_per_state(data, state, course):
    df = data[data['SG_UF_RESIDENCIA'] == state]
    mulher = df[(df['TP_SEXO'] == 'F')]
    homem = df[(df['TP_SEXO'] == 'M')]

    colors = ['#fd7f6f', '#7eb0d5']
    fig = px.box(df, x=df[course])
    fig = go.Figure()
    fig.add_trace(go.Box(y=mulher[course], name="Women", marker_color = '#fd7f6f'))
    fig.add_trace(go.Box(y=homem[course], name="Men", marker_color = '#7eb0d5'))
    fig.update_layout(
        title_text=f'{state} - Grades of {grade} in {year}', # title of plot
        xaxis_title_text='Gender', # xaxis label
        yaxis_title_text=f'Grades of {grade}', # yaxis label
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)

def ploting_distribution_female_male_per_state(data, state, course):
    colors = ['#fd7f6f', '#7eb0d5']
    df = data[data['SG_UF_RESIDENCIA'] == state]
    fig = px.histogram(df, x=df[course], color=df['TP_SEXO'], color_discrete_sequence=colors)
    # Reduce opacity to see both histograms
    fig.update_traces(opacity=0.7)
    fig.update_layout(
        barmode='overlay',
        title_text=f'{state} - Grades of {grade} in {year}', # title of plot
        xaxis_title_text=f'Grades of {grade} in {year}', # xaxis label
        yaxis_title_text='Frequency', # yaxis label
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    st.plotly_chart(fig, use_container_width=True)
    # fig.show()

def results_ranksum_per_state(data, state, course):
    mulher = data[(data['TP_SEXO'] == 'F')]
    homem = data[(data['TP_SEXO'] == 'M')]
    dataset_women = mulher[mulher['SG_UF_RESIDENCIA'] == state]
    dataset_man = homem[homem['SG_UF_RESIDENCIA'] == state]
    test, p = ranksums(dataset_women[course], dataset_man[course])
    print(f'p:', p, f'test:', test)
    test_result= f'[p_value: {p} , test: {test}]' 
    st.text(test_result)
    
def plot_distribution_ranksum_per_state(data, state, course):
    print(state)
    ploting_distribution_female_male_per_state(data, state, course)
    results_ranksum_per_state(data, state, course)

def plot_distribution_ranksum_all_states(data, states, course):
    for state in states:
        print(state)
        ploting_distribution_female_male_per_state(data, state, course)
        results_ranksum_per_state(data, state, course)
        
def plot_statistical_tests_all_states(data, course):
    states = list(data['SG_UF_RESIDENCIA'].unique())
    plot_distribution_ranksum_all_states(data, states, course)

def plot_statistical_tests_per_state(data, brazilian_state, course):
    if brazilian_state == 'ALL STATES':
        plot_distribution_ranksum_all_states(data, list(ddf['SG_UF_RESIDENCIA'].unique()), course)
    else:
        plot_distribution_ranksum_per_state(data, brazilian_state, course)

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
                           bargap=0.25, width=1000, height=550,
                           legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))

        data_ = [go.Bar(y=filtro.men,
                    x=filtro.estados,
                    name='Men',
                    hoverinfo='x+name+y',
                    text=filtro.men.apply(lambda y: f"{y:.0f}"),
                    marker=dict(color='#7eb0d5'),
                    textfont=dict(family="Arial",
                                  size=15),
                    textposition='outside',
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
                ]
        fig = go.Figure(data=data_, layout=layout)
        fig.update_layout(barmode='group', font=dict(size=10, family="Arial", color="black"))
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='White')
        fig['layout']['xaxis']['titlefont'] = dict(size=14)
        fig['layout']['xaxis']['tickfont'] = dict(size=12)
        fig['layout']['yaxis']['titlefont'] = dict(size=14)
        fig['layout']['yaxis']['tickfont'] = dict(size=12)
        fig['layout']['legend']['font'] = dict(size=12)
        st.plotly_chart(fig, use_container_width=True)

        if show_statistical_test:
            graficoSexo(ddf)
            plot_statistical_tests_per_state(ddf, brazilian_state, grades_names_to_columns[grade])
            ploting_boxplot_gender_per_state(ddf, brazilian_state, grades_names_to_columns[grade])
            # print(grades_names_to_columns[grade])

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
                           bargap=0.25, width=1000, height=550,
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
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='White')
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
                           bargap=0.25, width=1000, height=550)
        #paleta
        # "#fd7f6f", "#7eb0d5", "#b2e061", "#bd7ebe", "#ffb55a", "#ffee65", "#beb9db", "#fdcce5", "#8bd3c7"
        data_ = [go.Bar(y=filtro.white,
                    x=filtro.estados,
                    name='White',
                    hoverinfo='x+name+y',
                    # text=filtro.white.apply(lambda y: f"{y:.0f}"),
                    marker=dict(color='#fd7f6f'),
                    textfont=dict(family="Arial",
                                  size=80),
                    textposition='outside'
                    ),
                go.Bar(y=filtro.black,
                    x=filtro.estados,
                    name='Black',
                    # text=filtro.black.apply(lambda y: f"{y:.0f}"),
                    hoverinfo='x+name+y',
                    marker=dict(color='#7eb0d5'),
                    textfont=dict(family="Arial",
                                  size=60),
                    textposition='outside'
                    ),
                go.Bar(y=filtro.brown,
                    x=filtro.estados,
                    name='Brown',
                    # text=filtro.brown.apply(lambda y: f"{y:.0f}"),
                    hoverinfo='x+name+y',
                    marker=dict(color='#b2e061'),
                    textfont=dict(family="Arial",
                                  size=60),
                    textposition='outside'
                    ),
                go.Bar(y=filtro.yellow,
                    x=filtro.estados,
                    name='Yellow',
                    # text=filtro.yellow.apply(lambda y: f"{y:.0f}"),
                    hoverinfo='x+name+y',
                    marker=dict(color='#bd7ebe'),
                    textfont=dict(family="Arial",
                                  size=60),
                    textposition='outside'
                    ),
                go.Bar(y=filtro.indigenous,
                    x=filtro.estados,
                    name='Indigenous',
                    # text=filtro.indigenous.apply(lambda y: f"{y:.0f}"),
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
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='White')
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
                           bargap=0.25, width=1000, height=550)

        data_ = [go.Bar(y=filtro.a_class,
                    x=filtro.estados,
                    name='A Class (up to 20 MS)',
                    # text=filtro.a_class.apply(lambda y: f"{y:.0f}"),
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
                    # text=filtro.b_class.apply(lambda y: f"{y:.0f}"),
                    marker=dict(color='#7eb0d5'),
                    textfont=dict(family="Arial",
                                  size=80),
                    textposition='outside'
                    ),
                go.Bar(y=filtro.c_class,
                    x=filtro.estados,
                    name='C Class (4-10 MS)',
                    # text=filtro.c_class.apply(lambda y: f"{y:.0f}"),
                    hoverinfo='x+name+y',
                    marker=dict(color='#b2e061'),
                    textfont=dict(family="Arial",
                                  size=60),
                    textposition='outside'
                    ),
                go.Bar(y=filtro.d_class,
                    x=filtro.estados,
                    name='D Class (2-4 MS)',
                    # text=filtro.d_class.apply(lambda y: f"{y:.0f}"),
                    hoverinfo='x+name+y',
                    marker=dict(color='#bd7ebe'),
                    textfont=dict(family="Arial",
                                  size=60),
                    textposition='outside'
                    ),
                go.Bar(y=filtro.e_class,
                    x=filtro.estados,
                    name='E Class (up to 2 MS)',
                    # text=filtro.e_class.apply(lambda y: f"{y:.0f}"),
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
        fig.update_xaxes(showgrid=True, gridwidth=1, gridcolor='White')
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

        legend_education_level  = "**1**- Unknown;\n**2**- No study;\n**3**- Incomplete primary school;\n**4**- Primary school;\n**5**- Secondary school;\n**6**- High school;\n**7**- Graduated;\n**8**- Post graduated."

        legend_parents_profession  = "**1**- Unknown;\n**2**- Farmer, fisherman/fisherwoman etc;\n**3**- Elderly caregiver, doorman/portress, salesperson etc;\n**4**- Baker, painter, electrician, driver etc;\n**5**- Professor, technician, police etc;\n**6**- Physician, engineer, judge, lawyer etc."

        for question in questions:     
            print(question)
            if (questions[0] == 'Q001' or questions[0] == 'Q002'):
                ddf_par[question] = ddf_par[question].map({"H": 1, "A": 2, "B": 3, "C": 4, "D": 5, "E": 6, "F": 7, "G": 8})
            elif (questions[0] == 'Q003' or questions[0] == 'Q004'):
                ddf_par[question] = ddf_par[question].map({"F": 1, "A": 2, "B": 3, "C": 4, "D": 5, "E": 6})
            
        ddf_par = ddf_par.rename(columns=dict_names_exams)    
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
        if (questions[0] == 'Q001' or questions[0] == 'Q002'):
            st.caption(legend_education_level)
        elif (questions[0] == 'Q003' or questions[0] == 'Q004'):
            st.caption(legend_parents_profession)

    duration = fim(init)
    print(f"Duração: {duration}\n")

with st.container():
    plot = our_plot(titles_and_graphs[chart_type], ddf, st)
