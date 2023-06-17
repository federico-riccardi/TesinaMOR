#!/usr/bin/env python3

# visit http://127.0.0.1:8050/ in your web browser.

from dash import Dash, html, dcc
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
import os

app = Dash(__name__)

fig_inf = go.Figure(layout = go.Layout(title = "L_inf error Greedy vs FEM", xaxis = {'title':'parameters'}, yaxis = {'title':'error_inf'}, showlegend=True))
fig_2 = go.Figure(layout = go.Layout(title = "L_2 error Greedy vs FEM", xaxis = {'title':'parameters'}, yaxis = {'title':'error_2'}, showlegend=True))
fig_1 = go.Figure(layout = go.Layout(title = "semi_H_1 error Greedy vs FEM", xaxis = {'title':'parameters'}, yaxis = {'title':'error_H1_seminorm'}, showlegend=True))

for points in os.listdir("results_plot_Greedy"):
    df = pd.read_csv("results_plot_Greedy/{}/error.csv".format(points))
    fig_inf.add_trace(go.Scatter(x = df['parameters'], y = df['error_inf'], name="{}".format(points)))
    fig_2.add_trace(go.Scatter(x = df['parameters'], y = df['error_2'], name="{}".format(points)))
    fig_1.add_trace(go.Scatter(x = df['parameters'], y = df['error_H1'], name="{}".format(points)))



fig_2.update_layout(yaxis={'showexponent': 'all', 'exponentformat': 'e', 'rangemode': 'tozero',
    'ticks': 'outside'}, legend=dict(
    yanchor="top",
    y=0.99,
    xanchor="left",
    x=0.01
))

fig_1.update_layout(yaxis={'showexponent': 'all', 'exponentformat': 'e', 'rangemode': 'tozero',
    'ticks': 'outside'}, legend=dict(
    yanchor="top",
    y=0.99,
    xanchor="left",
    x=0.01
))

fig_inf.update_layout(yaxis={'showexponent': 'all', 'exponentformat': 'e', 'rangemode': 'tozero',
    'ticks': 'outside'}, legend=dict(
    yanchor="top",
    y=0.99,
    xanchor="left",
    x=0.01
))

app.layout = html.Div(children=[
    html.H1(children='Title 1'),
        html.Div(children='''
        Subtitle.
    '''),

    dcc.Graph(
        id='example-graph',
        figure=fig_inf
    ),
    dcc.Graph(
        id='example-graph',
        figure=fig_2
    ),
    dcc.Graph(
        id='example-graph',
        figure=fig_1
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)

    