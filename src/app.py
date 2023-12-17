import dash
from dash import dcc, html, Input, Output, Dash, dash_table
import dash_bootstrap_components as dbc
import pandas as pd
import plotly.express as px
external_stylesheets = [
    {
        "href": (
            "https://fonts.googleapis.com/css2?"
            "family=Lato:wght@400;700&display=swap"
        ),
        "rel": "stylesheet",
        
    },'https://codepen.io/chriddyp/pen/bWLwgP.css'

]
# Sample data
#demog=pd.read_csv('./parameters/chile_demographic.csv')
#regiones=demog['Region'].unique().tolist()
#df = pd.read_csv('./src/data/rawdata_20231025.csv')
df = pd.read_csv('data/rawdata_20231025.csv')
df['DateTime'] = df['DateTime'].astype('datetime64[ns]')
df = df.set_index(['DateTime'], drop=True)
empresas=df['Empresa'].unique().tolist()
comuna=df['Comuna'].unique().tolist()
parametro=df['Parametro'].unique().tolist()

# Initialize the app
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
app.title=title = 'Water Quality Chile - SISS Data'
# Define the layout
app.layout = html.Div([
    html.Div(
        children=[
        html.Div(children=[
            html.P(children="💧", className="header-emoji"),
            html.H1(children='Water Quality Chile',className='header-title'),
            html.P(children=('Dashboard de la calidad fisicoquímica del agua potable en las diferentes comunas de Chile, de acuerdo a la información reportada por las empresas sanitarias. (SISS,2023). Creado por: Luis Plaza A.'),className='header-description')
        ])],
                className="header"),
    html.Div(
        children=[
            html.Div(
                children=[
                    html.Div(children='Seleccione una ciudad:',className='menu-title'),
                    dcc.Dropdown(
                        id='city-dropdown',
                        options=[{'label': city, 'value': city} for city in comuna],
                        value=[comuna[0]],
                        clearable=False,
                        multi=True,
                        className='dropdown'
                    ),
                ]
            ),
            html.Div(
                children=[
                    html.Div(children='Seleccione parámetro:',className='menu-title'),
                    dcc.Dropdown(
                        id='parameter-dropdown',
                        options=[{'label': str(param), 'value': param} for param in parametro],
                        value=parametro[11],
                        clearable=False,
                        className='dropdown',
                        #style={'width':'120%'}
                    ),]),
            html.Div(
                children=[
                    html.Div(children='Seleccione un rango de fechas: ',className='menu-title'),
                    dcc.DatePickerRange(
                            id="date-range",display_format='MM/YYYY',
                            min_date_allowed=df.index.min().date(),
                            max_date_allowed=df.index.max().date(),
                            start_date=df.index.min().date(),
                            end_date=df.index.max().date(),
                        ),
                ]),

        ], className='menu',),
    html.Div([
        dcc.Graph(id='line-plot',className='card'),
        html.Div([
            html.Div([
                #html.H3('Column 1'),
                dcc.Graph(id='histogram',className='card')
            ], className="six columns"),

            html.Div([
                html.H3('Estadísticas Resumen'),
                dash_table.DataTable(
                    id='table',
                    style_cell={'fontSize':20},
                )
            ], className="six columns"),
        ], className="row"),
                        
        ],className="wrapper",
        )
])


# Define the callback functions
@app.callback(
    Output('line-plot', 'figure'),
    Output('histogram', 'figure'),
    Output('table','data'),
    Input('city-dropdown', 'value'),
    Input('parameter-dropdown', 'value'),
    Input('date-range', 'start_date'),
    Input('date-range', 'end_date'),
)
def update_graph(city, parameter,start_date,end_date):
    a=[(lambda x: df['Comuna']==x)(x) for x in city]
    mask=a[0]
    for i in range(1,len(city)):
        mask=mask|a[i]
    filtered_data = df[mask & (df['Parametro'] == parameter)].copy()
    mask1=((filtered_data.index >=start_date)&(filtered_data.index<=end_date))
    filtered_data=filtered_data[mask1]
    try:
        filtered_data['Valor']=filtered_data['Valor'].astype(float)
        filtered_data.sort_index(inplace=True)
        line_plot_fig = px.line(filtered_data, x=filtered_data.index.values, y='Valor',
                                color='Comuna',title=f'{parameter}')
        line_plot_fig.add_hline(y=float(filtered_data['Limite'].unique()), line_dash="dash",
                                line_color="red", annotation_text="Límite Superior", 
                                annotation_position="bottom right")
        histogram_fig = px.histogram(filtered_data, x='Valor',color='Comuna',opacity=0.5, 
                                    histnorm='percent', title=f'{parameter} - Histograma')
        histogram_fig.add_vline(x=float(filtered_data['Limite'].unique()), line_dash="dash",
                                line_color="red", annotation_text="Límite Superior", 
                                annotation_position="top")
        df2=filtered_data.groupby('Comuna')['Valor'].describe().T.round(2)
        df2=df2.reset_index()
        data=df2.to_dict('records')
    except:
        print('Problemas durante la actualización')
    return line_plot_fig, histogram_fig, data

#app.css.append_css({
#    'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'
#})

# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)