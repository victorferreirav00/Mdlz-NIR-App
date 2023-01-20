from os import stat_result
from random import randint
import dash
import dash_bootstrap_components as dbc
from dash import dcc, html, ctx
import numpy as np
import plotly.graph_objects as go
import pandas as pd
import plotly.express as px
from dash.dependencies import Input, Output, State
from scipy import signal
import dash_auth
import io
import base64
import openpyxl

app = dash.Dash(
    external_stylesheets=[__name__, dbc.themes.BOOTSTRAP]
)
server = app.server

VALID_USERNAME_PASSWORD_PAIRS = {
    'Ben Trank': 'NIR1968Bt',
    'Lucas Cava': 'NIR2022Lc',
    'Lucas Fedalto': 'NIR2022Lf',
    'Victor Ferreira': 'NIR2022Vf'
}

auth = dash_auth.BasicAuth(app, VALID_USERNAME_PASSWORD_PAIRS)
x = np.arange(400, 2500, 0.5)

fig = go.Figure(
    data=[],
    layout=go.Layout(
        width=1200,
        height=600,
        title=go.layout.Title(text='test', y=0.9, x=0.5),
        xaxis={
            'showticklabels': True,
            'title': 'Wave Length (nm)',
            'linecolor': 'black',
            'linewidth': 2,
            'showgrid': False,
            'tick0': 400,
            'dtick': 250,
            'mirror': True
        },
        yaxis={
            'showticklabels': True,
            'title': 'Abs (au)',
            'linecolor': 'black',
            'linewidth': 2,
            'showgrid': True,
            'mirror': True,
            'gridcolor': 'darkgrey'
        },
        plot_bgcolor='white'
    )
)
print(fig.layout.title.text, flush=True)
app.layout = dbc.Container([
    dcc.Store(id='dropdown-data', data=[],
              storage_type='session'),
    dcc.Store(id='trace-list', data=[],
              storage_type='session'),
    dcc.Store(id='plot-data', data=[],
              storage_type='session'),
    dbc.Row([
        dbc.Col([
            dbc.Row([
                dbc.Col(md=1),
                dbc.Col([
                    html.H1('NIR Graphing Tool', id='pagetitle'),
                    html.H5('Suggestions: Press Clear Plot in between each plot. Set up plot (title, 2nd derivative or spectrum, etc.) and then press "Plot Data. Some items can be altered dinamically, but that feature is not reliable at this point.', id='sugestion'),
                    html.H5('Guide:'),
                    html.H6('1-Load Dataframe'),
                    html.H6('2-Set Up Plot'),
                    html.H6(
                        '3-Hover mouse over plot to see more options (save as .png, zoom, pan, etc)'),
                    html.H3('Plot Options'),
                    dbc.Col(),
                    dbc.Col([
                        dcc.RadioItems(['Spectrum', 'Second Derivative'], 'Spectrum',
                                       inline=True, inputStyle={'margin': '10px'}, id='radio-items', style={'justify-content': 'center'}),
                        dcc.Input(
                            id="title-input",
                            type='text',
                            placeholder="Write the graph title"),
                        dbc.Button("Change Title", color="primary",
                                   className="me-1", id='button-title', style={'margin': '10px'}),
                        html.H5('Y Grid'),
                        dcc.RadioItems(['Active', 'Inactive'], 'Active',
                                       inline=True, inputStyle={'margin': '5px'}, id='grid-items', style={'justify-content': 'center'}
                                       )
                    ]),
                    dbc.Col(),
                    html.H6(
                        'Known issue: the plot can display results from last session when you open the app. Pressing "Clear Plot" should solve the problem.'),
                    html.H6(
                        'If the color scheme on the plot is not ideal, press "Plot Data" until the colors are in a better scheme.')
                ], md=10, id='graph-options-container'),
                dbc.Col(md=1)

            ], style={'height': '115vh'})

        ], md=3, id='sidebar-column'),
        dbc.Col([
            html.Div(id='graph-div'),
            dcc.Upload(
                id='upload-data',
                children=html.Div([
                    'Drag and Drop or ',
                    html.A('Select Dataset From Files')
                ]),
                style={
                    'width': '80%',
                    'height': '60px',
                    'lineHeight': '60px',
                    'borderWidth': '1px',
                    'borderStyle': 'dashed',
                    'borderRadius': '5px',
                    'textAlign': 'center',
                    'margin': '10px'
                },
                # Allow multiple files to be uploaded
                multiple=True
            ),
            html.Div(id='output-data-upload'),
            dbc.Button("Plot Data", color="primary",
                       className="me-1", id='button-add', style={'margin': '10px'}),
            dbc.Button("Clear plot", color="secondary",
                       className="me-1", id='button-clear'),
            dbc.Button('Download 2nd Deriv Data', color='primary',
                       id='download-button', className='me-1'),
            dcc.Download(id='download-dataframe')
        ], md=9, id='dashboard-background')
    ])
], fluid=True, id='container')


def parse_contents(contents, filename, date):
    content_type, content_string = contents.split(',')

    decoded = base64.b64decode(content_string)
    try:
        if 'csv' in filename:
            # Assume that the user uploaded a CSV file
            df = pd.read_csv(
                io.StringIO(decoded.decode('utf-8')))
            return df
        elif 'xls' in filename:
            # Assume that the user uploaded an excel file
            df = pd.read_excel(io.BytesIO(decoded))
            return df
    except Exception as e:
        print(e)
        return html.Div([
            'There was an error processing this file.'
        ])


def second_derivative(dataframe, x):
    y = np.array(dataframe.values.tolist())
    y_deriv = np.gradient(y, x)
    y_2ndDeriv = np.gradient(y_deriv, x)
    return y_2ndDeriv


@app.callback(
    Output('download-dataframe', 'data'),
    State('upload-data', 'contents'),
    State('upload-data', 'filename'),
    State('upload-data', 'last_modified'),
    Input('download-button', 'n_clicks'),
    prevent_initial_call=True
)
def download_dataframe(list_of_contents, list_of_names, list_of_dates, n_clicks):
    df_dict = {}
    df = [
        parse_contents(c, n, d) for c, n, d in
        zip(list_of_contents, list_of_names, list_of_dates)][0]
    for (columnName, columnData) in df.items():
        deriv = second_derivative(df[columnName], x)
        df_dict[columnName] = deriv
    deriv_dataframe = pd.DataFrame(df_dict)
    print(deriv_dataframe, flush=True)
    return dcc.send_data_frame(deriv_dataframe.to_excel, '2nd Derivative dataset.xlsx', sheet_name='2nd Derivative')


@ app.callback(Output('output-data-upload', 'children'),
               Input('upload-data', 'contents'),
               State('upload-data', 'filename'),
               State('upload-data', 'last_modified'))
def generate_dropdown(list_of_contents, list_of_names, list_of_dates):
    if list_of_contents is not None:
        df = [
            parse_contents(c, n, d) for c, n, d in
            zip(list_of_contents, list_of_names, list_of_dates)]
    name_list = []
    for columns in df[0].columns:
        name_list.append(columns)
    return html.Div([
        dcc.Dropdown(multi=True, options=[{'label': sheet, 'value': sheet} for sheet in name_list],
                     placeholder='Select a test', id='dropdown', style={
                         'width': '89.5%',
                         'margin': '5px'}),
    ])


@app.callback(
    Output('dropdown-data', 'data'),
    Input('dropdown', 'value')
)
def gen_test_list(dropdown_values):
    return dropdown_values


@ app.callback(
    Output(component_id='plot-data', component_property='data'),
    [
        State(component_id='dropdown-data', component_property='data'),
        Input(component_id='radio-items', component_property='value'),
        State(component_id='trace-list', component_property='data'),
        State('upload-data', 'contents'),
        State('upload-data', 'filename'),
        State('upload-data', 'last_modified'),
        Input(component_id='button-add', component_property='n_clicks'),
        Input(component_id='button-clear', component_property='n_clicks')
    ]
)
def updategraph(testname, graphtype, trace_list, contents, filename, last_modified, addclick, removeclick):
    button_clicked = ctx.triggered_id
    if button_clicked == 'button-clear':
        trace_list = []
        return trace_list

    else:
        df_list = [
            parse_contents(c, n, d) for c, n, d in
            zip(contents, filename, last_modified)]
        df = df_list[0]

        colors = ['#000000', '#2f4f4f', '#556b2f', '#8b4513', '#6b8e23', '#800000', '#708090', '#483d8b', '#008000', '#bc8f8f',
                  '#008080', '#b8860b', '#4682b4', '#d2691e', '#9acd32', '#4b0082', '#32cd32', '#8fbc8f', '#8b008b', '#48d1cc', '#9932cc',
                  '#ff4500', '#ff8c00', '#ffd700', '#c71585', '#deb887', '#00ff00', '#00ff7f', '#dc143c', '#00bfff', '#f4a460', '#9370db',
                  '#0000ff', '#a020f0', '#adff2f', '#ff6347', '#da70d6', '#ff00ff', '#db7093', '#fa8072', '#ffff54',
                  '#6495ed', '#dda0dd', '#90ee90', '#add8e6', '#7fffd4', '#ff69b4']
        for test in testname:
            dff = df[test]
            i = randint(0, len(colors))
            if graphtype == 'Second Derivative':
                y = second_derivative(dff, x)
            else:
                y = dff
            if len(testname) == 1 and fig.data == ():
                trace_list.append(
                    go.Scatter(
                        x=x,
                        y=y,
                        mode='lines',
                        line=go.scatter.Line(color='black', width=1),
                        name=test,
                    )
                )
            else:
                trace_list.append(
                    go.Scatter(
                        x=x,
                        y=y,
                        mode='lines',
                        line=go.scatter.Line(color=colors[i], width=1),
                        name=test,
                    )
                )
                colors.pop(i)
                i = i+1
    return trace_list


@app.callback(Output('graph-div', 'children'),
              State(component_id='title-input', component_property='value'),
              Input(component_id='button-title',
                    component_property='n_clicks'),
              Input(component_id='grid-items', component_property='value'),
              Input('plot-data', 'data'))
def update_figure(titletext, change_title, y_grid, data):
    fig.layout.title.text = titletext
    fig.data = []
    for trace in data:
        fig.add_trace(trace)
    if data == []:
        fig.data = []

    names = set()
    fig.for_each_trace(
        lambda trace:
        trace.update(showlegend=False)
        if (trace.name in names) else names.add(trace.name))

    fig.update_layout(title_text=titletext,
                      legend={
                          'bgcolor': 'white',
                          'bordercolor': 'black',
                          'borderwidth': 1
                      })
    if y_grid == 'Active':
        fig.update_layout(yaxis={'showgrid': True,
                                 'gridcolor': 'darkgrey'})
    else:
        fig.update_layout(yaxis={'showgrid': False})
    if titletext == '':
        fig.update_layout(title_text='')

    return dcc.Graph(figure=fig, id='graph', config={
                     'toImageButtonOptions': {'filename': fig.layout.title.text}})


if __name__ == "__main__":
    app.run_server(debug=True, port='8050')
