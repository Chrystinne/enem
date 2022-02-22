import streamlit as st
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

import plotly.express as px
import chart_studio.plotly as py
import plotly.graph_objs as go

import altair as alt
from bokeh.plotting import figure
from make_plots import (
    matplotlib_plot,
    sns_plot,
    pd_plot,
    plotly_plot,
    altair_plot,
    bokeh_plot,
)


# can only set this once, first thing to set
st.set_page_config(layout="wide")

# plot_types = (
#     "Scatter",
#     "Histogram",
#     "Bar",
#     "Line",
#     "3D Scatter",
# )  # maybe add 'Boxplot' after fixes

titles_and_graphs = {
    "1 - Geographical belongingness": {"type": None, "question": ""},
    "2 - Age": {"type": None, "question": ""},
    "3 - Gender": {"type": "pyramid", "question": ""},
    "4 - Marital status": {"type": None, "question": ""},
    "5 - Ethnicity": {"type": None, "question": ""},
    "6.1 - Father's education level": {"type": "parallel", "question": "Q001"},
    "6.2 - Mother's education level": {"type": "parallel", "question": "Q002"},
    "7.1 - Father's profession": {"type": None, "question": "Q003"},
    "7.2 - Mother's profession": {"type": None, "question": "Q004"},
    "8 - Income": {"type": None, "question": ""},
    "9 - Socioeconomic Status": {"type": None, "question": ""},
}

factors = titles_and_graphs.keys()

# Get data
@st.cache()
def load_data():
    return pd.read_csv("../datasets/MICRODADOS_ENEM_2018.csv", sep=';', encoding='cp1252', nrows=10000)

df = load_data().copy(deep=True)

# Top text area
with st.container():
    st.title("Analysis of factors that affect student's quality of academic performance 📊")
    # st.header("Popular plots in popular plotting libraries")
    # st.subheader("""See the code and plots for six libraries at once!""")


# User choose type
chart_type = st.selectbox("Choose the factor you would like to analyse", factors, 2)

col1, col2 = st.columns(2)


# create plots
# def show_plot(kind: str):
#     st.write(kind)
#     if kind == "Matplotlib":
#         plot = matplotlib_plot(chart_type, df)
#         st.pyplot(plot)
#     elif kind == "Seaborn":
#         plot = sns_plot(chart_type, df)
#         st.pyplot(plot)
#     elif kind == "Plotly Express":
#         plot = plotly_plot(chart_type, df)
#         st.plotly_chart(plot, use_container_width=True)
#     elif kind == "Altair":
#         plot = altair_plot(chart_type, df)
#         st.altair_chart(plot, use_container_width=True)
#     elif kind == "Pandas Matplotlib":
#         plot = pd_plot(chart_type, df)
#         st.pyplot(plot)
#     elif kind == "Bokeh":
#         plot = bokeh_plot(chart_type, df)
#         st.bokeh_chart(plot, use_container_width=True)


def plotly_plot(params, df):
    """ return plotly plots """

    if params["type"] == "pyramid":

        filtro = df.groupby(['SG_UF_RESIDENCIA', 'TP_SEXO'])['NU_NOTA_MT'].mean()
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
                        bargap=0.1, width=600, height=1000)

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

        return {"data":data_, "layout":layout}

    elif params["type"] == "parallel":
        question = params["question"]
        print(question)
        df_par = df[[question, 'NU_NOTA_COMP1', 'NU_NOTA_COMP2', 'NU_NOTA_COMP3', 'NU_NOTA_COMP4', 'NU_NOTA_COMP5']]
        df_par[question] = df_par[question].map({"A": 0, "B": 1, "C": 2, "D": 3, "E": 4, "F": 5, "G": 6, "H": 7})
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
                        cmax = 2),
                dimensions = dimensions
            )
        )
        return fig

with st.container():
    st.subheader(f"Enem Mean Grades by Gender")
    plot = plotly_plot(titles_and_graphs[chart_type], df)
    st.plotly_chart(plot, use_container_width=True)

with col1:
    st.subheader(f"Father's education level")
    plot = plotly_plot(titles_and_graphs["6.1 - Father's education level"], df)
    st.plotly_chart(plot, use_container_width=True)

with col2:
    st.subheader(f"Mother's education level")
    plot = plotly_plot(titles_and_graphs["6.2 - Mother's education level"], df)
    st.plotly_chart(plot, use_container_width=True)


# output plots
# if two_cols:
#     with col1:
#         show_plot(kind="Matplotlib")
#     with col2:
#         show_plot(kind="Seaborn")
#     with col1:
#         show_plot(kind="Plotly Express")
#     with col2:
#         show_plot(kind="Altair")
#     with col1:
#         show_plot(kind="Pandas Matplotlib")
#     with col2:
#         show_plot(kind="Bokeh")
# else:
#     with st.container():
#         for lib in libs:
#             show_plot(kind=lib)

# display data
with st.container():
    show_data = st.checkbox("See the raw data?")

    if show_data:
        df

    # notes
    # st.subheader("Notes")
    # st.write(
    #     """
    #     - This app uses [Streamlit](https://streamlit.io/) and the [Palmer Penguins](https://allisonhorst.github.io/palmerpenguins/) dataset.      
    #     - To see the full code check out the [GitHub repo](https://github.com/discdiver/data-viz-streamlit).
    #     - Plots are interactive where that's the default or easy to add.
    #     - Plots that use Matplotlib under the hood have fig and ax objects defined before the code shown.
    #     - Lineplots should have sequence data, so I created a date index with a sequence of dates for them. 
    #     - Where an axis label shows by default, I left it at is. Generally where it was missing, I added it.
    #     - There are multiple ways to make some of these plots.
    #     - You can choose to see two columns, but with a narrow screen this will switch to one column automatically.
    #     - Python has many data visualization libraries. This gallery is not exhaustive. If you would like to add code for another library, please submit a [pull request](https://github.com/discdiver/data-viz-streamlit).
    #     - For a larger tour of more plots, check out the [Python Graph Gallery](https://www.python-graph-gallery.com/density-plot/) and [Python Plotting for Exploratory Data Analysis](https://pythonplot.com/).
    #     - The interactive Plotly Express 3D Scatterplot is cool to play with. Check it out! 😎
        
    #     Made by [Jeff Hale](https://www.linkedin.com/in/-jeffhale/). 
        
    #     Subscribe to my [Data Awesome newsletter](https://dataawesome.com) for the latest tools, tips, and resources.
    #     """
    # )