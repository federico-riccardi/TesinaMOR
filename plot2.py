#!/usr/bin/env python3

# visit http://127.0.0.1:8050/ in your web browser.

from dash import Dash, html, dcc
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
import os

app = Dash(__name__)

fig_inf = go.Figure(layout = go.Layout(title = "L_inf error PINN vs FEM", xaxis = {'title':'iterations'}, yaxis = {'title':'error_inf'}, showlegend=True))
fig_2 = go.Figure(layout = go.Layout(title = "L_2 error PINN vs FEM", xaxis = {'title':'iterations'}, yaxis = {'title':'error_2'}, showlegend=True))

lines = ["dash", "solid"]
flag = 0
for coeff in os.listdir("results_plot2"):
    for parameters in os.listdir("results_plot2/"+coeff):
        df = pd.read_csv("results_plot2/{}/{}/error.csv".format(coeff, parameters))
        fig_inf.add_trace(go.Scatter(x = df['iterations'], y = df['error_inf'], name="{}/{}".format(coeff, parameters), line = dict(dash=lines[flag])))
        fig_2.add_trace(go.Scatter(x = df['iterations'], y = df['error_2'], name="{}/{}".format(coeff, parameters), line = dict(dash=lines[flag])))
    flag += 1

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
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)

    