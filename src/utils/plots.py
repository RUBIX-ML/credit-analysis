import pandas as pd
import numpy as np
import re
import plotly.graph_objects as go
from plotly.graph_objs import *
from plotly.offline import plot
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
import IPython, graphviz
from sklearn.tree import export_graphviz


def missing_value_plot(df):
    patterns = ['Nan', 'Null', 'NULL', 'NAN', '']
    col_names = []
    indexes = []
    for column in df.columns:
        current_indexes = list(df[df[column].isin(patterns)].index)
        indexes = indexes + current_indexes
        col_names = col_names + [column for x in current_indexes]
    y0 = indexes
    x0 = col_names

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        name='Matching points',
        x=x0,
        y=y0,
        mode='text',
        text='______________',
        textfont=dict(
            color="red",
            size=11)))

    fig.add_trace(go.Bar(
        name='All pints',
        x=df.columns,
        y=[len(df) for x in df.columns],
        marker=dict(color='green'),
        # fillcolor='green',
        opacity=0.6
        # marker_size=2,
        # mode='markers'
    ))

    fig.update_layout(
        title='Pattern Filtering Cross All Columns',
        paper_bgcolor='rgba(255,255,255,255)',
        plot_bgcolor='rgba(255,255,255,255)',
        font=dict(size=10, color="#333333"),
        width=1200,
        height=len(df) * 8,
        xaxis=dict(autorange=True, gridcolor='#eeeeee', title='Columns'),
        yaxis=dict(autorange='reversed', gridcolor='#eeeeee', title='Index number'),
        bargap=0.2)
    
    fig = go.Figure(data=data, layout=layout)
    plot.show()

    
def histogram_plot(df, column):
    trace = go.Histogram(
        x=df[column],
        marker=dict(color='#337ab7'),
        opacity=1)

    data = [trace]

    layout = go.Layout(
        paper_bgcolor='rgba(255,255,255.255)',
        plot_bgcolor='rgba(255,255,255,255)',
        font=dict(size=8, color="#333333"),
        width=600,
        height=100,
        xaxis=dict(autorange=True, gridcolor='#eeeeee'),
        yaxis=dict(autorange=True, gridcolor='#eeeeee'),
        bargap=0.1,
        margin=go.layout.Margin(l=10, r=10, b=40, t=20, pad=0))

    fig = go.Figure(data=data, layout=layout).update_xaxes(categoryorder='total descending')
    fig.show()


def bar_plot(df):
    traces = []

    for i in range(len(x_cols)):
        for j in range(len(y_cols)):
            trace = go.Bar(
                x=df[x_cols[i]],
                y=df[y_cols[j]],
                name=str(y_cols[j]) + ' | ' + str(x_cols[i]),
            )
            traces.append(trace)
    data = traces
    layout = go.Layout(
        title=chart_name,
        height=500,
        xaxis=dict(title=x_title, autorange=True, gridcolor='#eeeeee'),
        yaxis=dict(title=y_title, autorange=True, gridcolor='#eeeeee'),
        barmode='group',
        paper_bgcolor='rgba(255,255,255,255)',
        plot_bgcolor='rgba(255,255,255,255)',
        font=dict(size=9, color="#333333"))

    fig = go.Figure(data=data, layout=layout)
    fig.show()


def correlation_matrix(df):
    trace = {
        "type": "heatmap",
        "x": df.columns,
        "y": df.columns, 
        "z": [list(df.iloc[row, :]) for row in range(df.shape[0])],
        "colorscale": "Viridis"
        }
    data = [trace]
    layout = {
        "title": "Correlation Matrix", 
        "width": 800, 
        "xaxis": {"automargin": True}, 
        "yaxis": {"automargin": True}, 
        "height": 600, 
        "autosize": True, 
        "showlegend": True
    }
    fig = go.Figure(data=data, layout=layout)
    fig.show()

    
def feature_importance_plot(df):
    traces = []
    trace = go.Bar(
                x=list(df.keys()),
                y=list(df.values()),
            )
    traces.append(trace)
    data = traces
    layout = go.Layout(
        title='Feature Importance',
        height=300,
        xaxis=dict(title='Features', autorange=True, gridcolor='#eeeeee'),
        yaxis=dict(title='Impact', autorange=True, gridcolor='#eeeeee'),
        barmode='group',
        paper_bgcolor='rgba(255,255,255,255)',
        plot_bgcolor='rgba(255,255,255,255)',
        font=dict(size=9, color="#333333"))
    fig = go.Figure(data=data, layout=layout)
    fig.show()
    

def draw_tree(t, df, size=70, ratio=4, precision=0):
    """ Draws a representation of a random forest in IPython.

    Parameters:
    -----------
    t: The tree you wish to draw
    df: The data used to train the tree. This is used to get the names of the features.
    example: draw_tree(m.estimators_[0], X_train, precision=2)
    """
    s=export_graphviz(t, out_file=None, feature_names=df.columns, filled=True, special_characters=True, rotate=True, precision=precision)
    IPython.display.display(graphviz.Source(re.sub('Tree {', f'Tree {{ size={size}; ratio={ratio}', s)))