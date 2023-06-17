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
fig_1 = go.Figure(layout = go.Layout(title = '$H^1 seminorm error PINN vs FEM$', xaxis = {'title':'iterations'}, yaxis = {'title':'error_H1_seminorm'}, showlegend=True))

lines = ["dash", "solid"]
flag = 0
for coeff in os.listdir("results_plot_2"):
    for parameters in os.listdir("results_plot_2/"+coeff):
        df = pd.read_csv("results_plot_2/{}/{}/error.csv".format(coeff, parameters))
        fig_inf.add_trace(go.Scatter(x = df['iterations'], y = df['error_inf'], name="{}/{}".format(coeff, parameters), line = dict(dash=lines[flag])))
        fig_2.add_trace(go.Scatter(x = df['iterations'], y = df['error_2'], name="{}/{}".format(coeff, parameters), line = dict(dash=lines[flag])))
        fig_1.add_trace(go.Scatter(x = df['iterations'], y = df['error_semi_H1'], name="{}/{}".format(coeff, parameters), line = dict(dash=lines[flag])))
    flag += 1

x_s = []
for i in range(len(df['iterations'])):
    x_s.append(df['iterations'].to_dict().get(i))

fig_2.update_xaxes(tickvals=x_s)
fig_1.update_xaxes(tickvals=x_s)
fig_inf.update_xaxes(tickvals=x_s)

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

    