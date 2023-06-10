#!/usr/bin/env python3

# visit http://127.0.0.1:8050/ in your web browser.

from dash import Dash, html, dcc
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
import os

app = Dash(__name__)

fig_inf = go.Figure(layout = go.Layout(title = "L_inf error PINN vs FEM", xaxis = {'title':'parameters'}, yaxis = {'title':'error_inf'}, showlegend=True))
fig_2 = go.Figure(layout = go.Layout(title = "L_2 error PINN vs FEM", xaxis = {'title':'parameters'}, yaxis = {'title':'error_2'}, showlegend=True))

for iterations in os.listdir("plot1"):
    for coeff in os.listdir("plot1/"+iterations):
        df = pd.read_csv("plot1/{}/{}/error.csv".format(iterations, coeff))
        fig_inf.add_trace(go.Scatter(x = df['parameters'], y = df['error_inf'], name="{}/{}".format(iterations, coeff)))
        fig_2.add_trace(go.Scatter(x = df['parameters'], y = df['error_2'], name="{}/{}".format(iterations, coeff)))

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

    