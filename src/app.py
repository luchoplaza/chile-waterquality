import dash
from dash import dcc, html, Input, Output, dash_table
import pandas as pd
import plotly.express as px
external_stylesheets = [
    {
        "href": (
            "https://fonts.googleapis.com/css2?"
            "family=Lato:wght@400;700&display=swap"
        ),
        "rel": "stylesheet",        
    },'https://codepen.io/chriddyp/pen/bWLwgP.css',
    "https://platform.linkedin.com/in.js",

]
# Sample data
#demog=pd.read_csv('./parameters/chile_demographic.csv')
#regiones=demog['Region'].unique().tolist()
#df = pd.read_csv('./src/data/rawdata_20240409.csv')
df = pd.read_csv('data/rawdata_20240409.csv')
df['DateTime'] = df['DateTime'].astype('datetime64[ns]')
df = df.set_index(['DateTime'], drop=True)
#df.replace('SOLIDOS DISUELTOS TOTALES','SDT', inplace=True)
param_disc=["OLOR","COLOR VERDADERO", "SABOR"]
empresas=df['Empresa'].unique().tolist()
comuna=df['Comuna'].unique().tolist()
parametro=df['Parametro'].unique().tolist()[2:]
parametro=[x for x in parametro if x not in param_disc]
lims={'PH':{'lim_min':6.5,'lim_max':8.5},
      'ARSENICO':{'lim_min':0,'lim_max':0.01},
      'CLORO LIBRE RESIDUAL':{'lim_min':0.2,'lim_max':2.0}}

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
            html.P(children=('Dashboard de la calidad fisicoquímica del agua potable en las diferentes comunas de Chile, de acuerdo a la información reportada por las empresas sanitarias. (SISS,2023). Creado por: ',html.A('Luis Plaza A.',href='https://www.linkedin.com/in/lplazaalvarez/'),' Dudas y/o comentarios, contactarme !'),className='header-description'),
            
        ])],
                className="header"),
    html.Div([
        html.Script(**{"data-url": "https://platform.linkedin.com/in.js"}, type="IN/Share")
        ]),
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
                        optionHeight=52,
                        value=parametro[6],
                        clearable=False,
                        className='dropdown',
                        #style={'width':'120%'}
                    ),]),
            html.Div(
                children=[
                    html.Div(children='Seleccione un rango de fechas (MM/YYYY): ',className='menu-title'),
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
        dcc.Graph(id='line-plot',
                  className='card',
                  #style={"height": "60%", "width": "80%"}
                  ),
        html.Div([
            html.Div([
                #html.H3('Column 1'),
                dcc.Graph(id='histogram',className='card',)
            ], className="six columns"),

            html.Div([
                html.H3('Estadísticas'),
                dash_table.DataTable(
                    id='table',
                    style_cell={'fontSize':18},
                )
            ], className="six columns"),
        ], className="row"),
                        
        ],className="wrapper",
        ),   
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
    if city==[]:
        city=['COPIAPO']
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
        if filtered_data.empty:
            print('No hay valores para estas condiciones')
        else:
            unidad=filtered_data['Unidad'].unique()[0]
            line_plot_fig = px.line(filtered_data, x=filtered_data.index.values, y='Valor',
                                    color='Comuna',title=f'{parameter}', markers=True)
            line_plot_fig.update_layout(legend=dict(y=-0.15,orientation="h"),
                                        margin=dict(r=30),
                                        xaxis_title="Fecha", 
                                        yaxis_title=f"Valor ({unidad})"
                                        )
            line_plot_fig.update_xaxes(tickmode="auto",nticks=8)
            histogram_fig = px.histogram(filtered_data, x='Valor',color='Comuna',opacity=0.5, 
                                        histnorm='percent', title=f'Histograma - {parameter}')
            histogram_fig.update_layout(legend=dict(y=-0.15,orientation="h"),
                                        margin=dict(r=30),
                                        yaxis_title="Porcentaje (%)",
                                        xaxis_title=f"Valor ({unidad})"
                                        )
            histogram_fig.update_xaxes(tickmode="auto",nticks=6)
            if parameter in lims.keys():
                line_plot_fig.add_hline(y=lims[parameter]['lim_max'], line_dash="dash",
                                    line_color="red", annotation_text="Límite Superior", 
                                    annotation_position="bottom right")
                line_plot_fig.add_hline(y=lims[parameter]['lim_min'], line_dash="dash",
                                    line_color="red", annotation_text="Límite Inferior", 
                                    annotation_position="bottom right")
                histogram_fig.add_vline(x=lims[parameter]['lim_max'], line_dash="dash",
                                    line_color="red", annotation_text="Límite Superior", 
                                    annotation_position="top")
                histogram_fig.add_vline(x=lims[parameter]['lim_min'], line_dash="dash",
                                    line_color="red", annotation_text="Límite Inferior", 
                                    annotation_position="top")
            else:
                line_plot_fig.add_hline(y=float(filtered_data['Limite'].unique()), line_dash="dash",
                                        line_color="red", annotation_text="Límite Superior", 
                                        annotation_position="bottom right")
                
                histogram_fig.add_vline(x=float(filtered_data['Limite'].unique()), line_dash="dash",
                                        line_color="red", annotation_text="Límite Superior", 
                                        annotation_position="top")
            df2=filtered_data.groupby('Comuna')['Valor'].describe().T.round(2)
            df2=df2.reset_index(names='Variable')
            data=df2.to_dict('records')
            NEW_NAMES=['Conteo','Media','Desv.Est','Min','Q1','Q2','Q3','Max']
            for i in range(len(data)): 
                data[i]['Variable']=NEW_NAMES[i]
            aux={'Variable':'Last Value'}
            aux2={'Variable':'Last Date'}
            for Com in city:
                mask2=filtered_data['Comuna']==Com
                aux[Com]=filtered_data[mask2]['Valor'].iloc[-1].round(2)
                aux2[Com]=filtered_data[mask2].index[-1].strftime("%b-%Y")
            data.append(aux)
            data.append(aux2)
    except Exception:
        print('Problemas durante la actualización')
    return line_plot_fig, histogram_fig, data
# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)