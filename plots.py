#!/usr/bin/env python3

# visit http://127.0.0.1:8050/ in your web browser.

from dash import Dash, html, dcc
import plotly.express as px
import pandas as pd
import plotly.graph_objects as go
import os

app = Dash(__name__)

#fig = go.Figure(go.Scatter(x = df['x'], y = df['y'], name="loss1"), 
#                layout = go.Layout(title = "Loss of the PINN model", xaxis = {'title':'epoch'}, yaxis = {'title':'loss'}, showlegend=True))

fig = go.Figure(layout = go.Layout(title = "Loss of the PINN model", xaxis = {'title':'epoch'}, yaxis = {'title':'loss'}, showlegend=True))

for iterations in os.listdir("results"):
    for lam in os.listdir("results/"+iterations):
        df = pd.read_csv("results/"+iterations+"/"+lam+"/loss.csv")
        fig.add_trace(go.Scatter(x = df['epoch'], y = df['loss'], name="loss "+iterations+" epochs and lambda = "+lam))
# assume you have a "long-form" data frame
# see https://plotly.com/python/px-arguments/ for more options
#df = pd.read_csv('1000/loss.csv', sep = " ")
#fig = go.Figure(go.Scatter(x = df['x'], y = df['y']))
#fig=px.line(df, x="x", y="y")



app.layout = html.Div(children=[
    html.H1(children='Title'),
        html.Div(children='''
        Subtitle.
    '''),

    dcc.Graph(
        id='example-graph',
        figure=fig
    )
])

if __name__ == '__main__':
    app.run_server(debug=True)

    