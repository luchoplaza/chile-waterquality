import dash
from dash import dcc, html, Input, Output, dash_table
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

external_stylesheets = [
    {
        "href": (
            "https://fonts.googleapis.com/css2?"
            "family=Lato:wght@400;700&display=swap"
        ),
        "rel": "stylesheet",
    },
    "https://codepen.io/chriddyp/pen/bWLwgP.css",
    "https://platform.linkedin.com/in.js",
]

# Cargar datos
# demog = pd.read_csv('./parameters/chile_demographic.csv')
# regiones = demog['Region'].unique().tolist()
# df = pd.read_csv('./src/data/rawdata_20240409.csv')

# Nota: mantén esta ruta igual que en tu repo
# (en Streamlit / servidor, asegúrate de que exista data/rawdata_20240409.csv)
df = pd.read_csv("data/rawdata_20240409.csv")

df["DateTime"] = df["DateTime"].astype("datetime64[ns]")
df = df.set_index(["DateTime"], drop=True)

param_disc = ["OLOR", "COLOR VERDADERO", "SABOR"]
empresas = df["Empresa"].unique().tolist()
comuna = df["Comuna"].unique().tolist()
parametro = df["Parametro"].unique().tolist()[2:]
parametro = [x for x in parametro if x not in param_disc]

lims = {
    "PH": {"lim_min": 6.5, "lim_max": 8.5},
    "ARSENICO": {"lim_min": 0, "lim_max": 0.01},
    "CLORO LIBRE RESIDUAL": {"lim_min": 0.2, "lim_max": 2.0},
}


# Helper: figura vacía para casos sin datos
def empty_figure(title: str = "Sin datos disponibles"):
    fig = go.Figure()
    fig.add_annotation(
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        text=title,
        showarrow=False,
        font=dict(size=16),
    )
    fig.update_xaxes(visible=False)
    fig.update_yaxes(visible=False)
    fig.update_layout(margin=dict(l=40, r=40, t=40, b=40))
    return fig


# Inicializar app
app = dash.Dash(__name__, external_stylesheets=external_stylesheets)
server = app.server
app.title = "Water Quality Chile - SISS Data"


# Layout
app.layout = html.Div(
    [
        # Header
        html.Div(
            children=[
                html.Div(
                    children=[
                        html.P(children="💧", className="header-emoji"),
                        html.H1(
                            children="Water Quality Chile", className="header-title"
                        ),
                        html.P(
                            children=(
                                "Dashboard de la calidad fisicoquímica del agua potable en las diferentes comunas de Chile, de acuerdo a la información reportada por las empresas sanitarias. (SISS,2023). Creado por: ",
                                html.A(
                                    "Luis Plaza A.",
                                    href="https://www.linkedin.com/in/lplazaalvarez/",
                                ),
                                " Dudas y/o comentarios, contactarme !",
                            ),
                            className="header-description",
                        ),
                    ]
                )
            ],
            className="header",
        ),
        html.Div(
            [html.Script(**{"data-url": "https://platform.linkedin.com/in.js"}, type="IN/Share")]
        ),

        # Tabs principales
        dcc.Tabs(
            id="tabs",
            value="tab-time",
            children=[
                dcc.Tab(
                    label="Análisis temporal",
                    value="tab-time",
                    children=[
                        html.Div(
                            children=[
                                html.Div(
                                    children=[
                                        html.Div(
                                            children="Seleccione una ciudad:",
                                            className="menu-title",
                                        ),
                                        dcc.Dropdown(
                                            id="city-dropdown",
                                            options=
                                            [
                                                {"label": city, "value": city}
                                                for city in comuna
                                            ],
                                            value=[comuna[0]],
                                            clearable=False,
                                            multi=True,
                                            className="dropdown",
                                        ),
                                    ]
                                ),
                                html.Div(
                                    children=[
                                        html.Div(
                                            children="Seleccione parámetro:",
                                            className="menu-title",
                                        ),
                                        dcc.Dropdown(
                                            id="parameter-dropdown",
                                            options=[
                                                {
                                                    "label": str(param),
                                                    "value": param,
                                                }
                                                for param in parametro
                                            ],
                                            optionHeight=52,
                                            value=parametro[6]
                                            if len(parametro) > 6
                                            else parametro[0],
                                            clearable=False,
                                            className="dropdown",
                                        ),
                                    ]
                                ),
                                html.Div(
                                    children=[
                                        html.Div(
                                            children="Seleccione un rango de fechas (MM/YYYY): ",
                                            className="menu-title",
                                        ),
                                        dcc.DatePickerRange(
                                            id="date-range",
                                            display_format="MM/YYYY",
                                            min_date_allowed=df.index.min().date(),
                                            max_date_allowed=df.index.max().date(),
                                            start_date=df.index.min().date(),
                                            end_date=df.index.max().date(),
                                        ),
                                    ]
                                ),
                            ],
                            className="menu",
                        ),
                        html.Div(
                            [
                                dcc.Graph(
                                    id="line-plot",
                                    className="card",
                                ),
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                dcc.Graph(
                                                    id="histogram",
                                                    className="card",
                                                )
                                            ],
                                            className="six columns",
                                        ),
                                        html.Div(
                                            [
                                                html.H3("Estadísticas"),
                                                dash_table.DataTable(
                                                    id="table",
                                                    style_cell={"fontSize": 18},
                                                ),
                                            ],
                                            className="six columns",
                                        ),
                                    ],
                                    className="row",
                                ),
                            ],
                            className="wrapper",
                        ),
                    ],
                ),
                dcc.Tab(
                    label="Clasificación por máximos",
                    value="tab-max",
                    children=[
                        # Filtros para ranking y ficha por comuna
                        html.Div(
                            children=[
                                html.Div(
                                    children=[
                                        html.Div(
                                            children="Seleccione parámetro:",
                                            className="menu-title",
                                        ),
                                        dcc.Dropdown(
                                            id="max-parameter-dropdown",
                                            options=[
                                                {
                                                    "label": str(param),
                                                    "value": param,
                                                }
                                                for param in parametro
                                            ],
                                            value=parametro[6]
                                            if len(parametro) > 6
                                            else parametro[0],
                                            clearable=False,
                                            className="dropdown",
                                        ),
                                    ]
                                ),
                                html.Div(
                                    children=[
                                        html.Div(
                                            children="Seleccione un rango de fechas (MM/YYYY): ",
                                            className="menu-title",
                                        ),
                                        dcc.DatePickerRange(
                                            id="max-date-range",
                                            display_format="MM/YYYY",
                                            min_date_allowed=df.index.min().date(),
                                            max_date_allowed=df.index.max().date(),
                                            start_date=df.index.min().date(),
                                            end_date=df.index.max().date(),
                                        ),
                                    ]
                                ),
                                html.Div(
                                    children=[
                                        html.Div(
                                            children="Top N comunas:",
                                            className="menu-title",
                                        ),
                                        dcc.Slider(
                                            id="max-top-n",
                                            min=5,
                                            max=30,
                                            step=5,
                                            value=10,
                                            marks={
                                                5: "5",
                                                10: "10",
                                                15: "15",
                                                20: "20",
                                                25: "25",
                                                30: "30",
                                            },
                                        ),
                                    ]
                                ),
                                html.Div(
                                    children=[
                                        html.Div(
                                            children="Seleccione comuna para ficha:",
                                            className="menu-title",
                                        ),
                                        dcc.Dropdown(
                                            id="max-city-dropdown",
                                            options=[
                                                {
                                                    "label": c,
                                                    "value": c,
                                                }
                                                for c in comuna
                                            ],
                                            value=None,
                                            placeholder="(Por defecto se usará la comuna con máximo más alto)",
                                            clearable=True,
                                            className="dropdown",
                                        ),
                                    ]
                                ),
                            ],
                            className="menu",
                        ),
                        html.Div(
                            [
                                dcc.Graph(
                                    id="max-ranking-graph",
                                    className="card",
                                ),
                                html.Div(
                                    [
                                        html.Div(
                                            [
                                                dcc.Graph(
                                                    id="max-radar-graph",
                                                    className="card",
                                                )
                                            ],
                                            className="six columns",
                                        ),
                                        html.Div(
                                            [
                                                html.H3(
                                                    "Máximos por parámetro - comuna seleccionada"
                                                ),
                                                dash_table.DataTable(
                                                    id="max-table",
                                                    style_cell={"fontSize": 18},
                                                ),
                                            ],
                                            className="six columns",
                                        ),
                                    ],
                                    className="row",
                                ),
                            ],
                            className="wrapper",
                        ),
                    ],
                ),
            ],
        ),
    ]
)


# Callback: vista temporal existente
@app.callback(
    Output("line-plot", "figure"),
    Output("histogram", "figure"),
    Output("table", "data"),
    Input("city-dropdown", "value"),
    Input("parameter-dropdown", "value"),
    Input("date-range", "start_date"),
    Input("date-range", "end_date"),
)
def update_graph(city, parameter, start_date, end_date):
    if city == []:
        city = ["COPIAPO"]

    # Más legible que el list-comprehension original
    mask_comuna = df["Comuna"].isin(city)
    filtered_data = df[mask_comuna & (df["Parametro"] == parameter)].copy()

    mask_fecha = (filtered_data.index >= start_date) & (
        filtered_data.index <= end_date
    )
    filtered_data = filtered_data[mask_fecha]

    try:
        filtered_data["Valor"] = filtered_data["Valor"].astype(float)
        filtered_data.sort_index(inplace=True)

        if filtered_data.empty:
            return empty_figure("No hay valores para estas condiciones"), empty_figure(
                "No hay valores para estas condiciones"
            ), []

        unidad = filtered_data["Unidad"].unique()[0]

        line_plot_fig = px.line(
            filtered_data,
            x=filtered_data.index.values,
            y="Valor",
            color="Comuna",
            title=f"{parameter}",
            markers=True,
        )
        line_plot_fig.update_layout(
            legend=dict(y=-0.15, orientation="h"),
            margin=dict(r=30),
            xaxis_title="Fecha",
            yaxis_title=f"Valor ({unidad})",
        )
        line_plot_fig.update_xaxes(tickmode="auto", nticks=8)

        histogram_fig = px.histogram(
            filtered_data,
            x="Valor",
            color="Comuna",
            opacity=0.5,
            histnorm="percent",
            title=f"Histograma - {parameter}",
        )
        histogram_fig.update_layout(
            legend=dict(y=-0.15, orientation="h"),
            margin=dict(r=30),
            yaxis_title="Porcentaje (%)",
            xaxis_title=f"Valor ({unidad})",
        )
        histogram_fig.update_xaxes(tickmode="auto", nticks=6)

        # Límites
        if parameter in lims.keys():
            line_plot_fig.add_hline(
                y=lims[parameter]["lim_max"],
                line_dash="dash",
                line_color="red",
                annotation_text="Límite Superior",
                annotation_position="bottom right",
            )
            line_plot_fig.add_hline(
                y=lims[parameter]["lim_min"],
                line_dash="dash",
                line_color="red",
                annotation_text="Límite Inferior",
                annotation_position="bottom right",
            )
            histogram_fig.add_vline(
                x=lims[parameter]["lim_max"],
                line_dash="dash",
                line_color="red",
                annotation_text="Límite Superior",
                annotation_position="top",
            )
            histogram_fig.add_vline(
                x=lims[parameter]["lim_min"],
                line_dash="dash",
                line_color="red",
                annotation_text="Límite Inferior",
                annotation_position="top",
            )
        else:
            try:
                limite_val = float(filtered_data["Limite"].unique())
                line_plot_fig.add_hline(
                    y=limite_val,
                    line_dash="dash",
                    line_color="red",
                    annotation_text="Límite Superior",
                    annotation_position="bottom right",
                )
                histogram_fig.add_vline(
                    x=limite_val,
                    line_dash="dash",
                    line_color="red",
                    annotation_text="Límite Superior",
                    annotation_position="top",
                )
            except Exception:
                pass

        # Tabla de estadísticos por comuna
        df2 = filtered_data.groupby("Comuna")["Valor"].describe().T.round(2)
        df2 = df2.reset_index(names="Variable")
        data = df2.to_dict("records")

        NEW_NAMES = [
            "Conteo",
            "Media",
            "Desv.Est",
            "Min",
            "Q1",
            "Q2",
            "Q3",
            "Max",
        ]
        for i in range(len(data)):
            data[i]["Variable"] = NEW_NAMES[i]

        aux = {"Variable": "Last Value"}
        aux2 = {"Variable": "Last Date"}

        for Com in city:
            mask2 = filtered_data["Comuna"] == Com
            aux[Com] = filtered_data[mask2]["Valor"].iloc[-1].round(2)
            aux2[Com] = filtered_data[mask2].index[-1].strftime("%b-%Y")

        data.append(aux)
        data.append(aux2)

        return line_plot_fig, histogram_fig, data

    except Exception:
        return empty_figure("Problemas durante la actualización"), empty_figure(
            "Problemas durante la actualización"
        ), []


# Callback: clasificación por máximos (ranking + ficha comuna)
@app.callback(
    Output("max-ranking-graph", "figure"),
    Output("max-radar-graph", "figure"),
    Output("max-table", "data"),
    Input("max-parameter-dropdown", "value"),
    Input("max-date-range", "start_date"),
    Input("max-date-range", "end_date"),
    Input("max-top-n", "value"),
    Input("max-city-dropdown", "value"),
)
def update_max_views(parameter, start_date, end_date, top_n, selected_city):
    # Filtro por parámetro y fechas para el ranking
    df_param = df[(df["Parametro"] == parameter)].copy()
    df_param["Valor"] = pd.to_numeric(df_param["Valor"], errors="coerce")
    df_param = df_param.dropna(subset=["Valor"])

    mask_fecha = (df_param.index >= start_date) & (df_param.index <= end_date)
    df_param = df_param[mask_fecha]

    if df_param.empty:
        return (
            empty_figure("Sin datos para este parámetro y rango"),
            empty_figure("Sin datos para la comuna seleccionada"),
            [],
        )

    # Ranking por comuna (máximo)
    def _idxmax_datetime(s):
        # s es la serie de Valor dentro del grupo; usamos su índice datetime
        try:
            return s.idxmax()
        except Exception:
            return pd.NaT

    grouped = (
        df_param.groupby("Comuna")
        .agg(
            Valor_max=("Valor", "max"),
            Fecha_max=("Valor", _idxmax_datetime),
            Limite=("Limite", lambda x: pd.to_numeric(x, errors="coerce").max()),
        )
        .reset_index()
    )

    if grouped.empty:
        return (
            empty_figure("Sin datos para este parámetro y rango"),
            empty_figure("Sin datos para la comuna seleccionada"),
            [],
        )

    grouped["Fecha_max"] = grouped["Fecha_max"].dt.strftime("%Y-%m-%d")

    grouped = grouped.sort_values("Valor_max", ascending=False)
    top_n = int(top_n) if top_n is not None else 10
    top_grouped = grouped.head(top_n)

    ranking_fig = px.bar(
        top_grouped,
        x="Valor_max",
        y="Comuna",
        orientation="h",
        text="Valor_max",
        title=f"Ranking de comunas por máximo de {parameter}",
    )
    ranking_fig.update_layout(
        yaxis={"categoryorder": "total ascending"},
        xaxis_title="Valor máximo",
        margin=dict(l=80, r=40, t=60, b=40),
    )

    # Límite normativo (si existe)
    limites_unicos = (
        df_param["Limite"].pipe(pd.to_numeric, errors="coerce").dropna().unique()
    )
    if limites_unicos.size > 0:
        limite_val = float(limites_unicos[0])
        ranking_fig.add_vline(
            x=limite_val,
            line_dash="dash",
            line_color="red",
            annotation_text="Límite",
            annotation_position="top",
        )

    # Si no se elige comuna, usamos la primera del ranking
    if selected_city is None or selected_city == "":
        if not top_grouped.empty:
            selected_city = top_grouped.iloc[0]["Comuna"]
        else:
            return ranking_fig, empty_figure("Sin comuna seleccionada"), []

    # Ficha de comuna: perfil de máximos por parámetro (no solo el seleccionado)
    df_city = df[df["Comuna"] == selected_city].copy()
    df_city["Valor"] = pd.to_numeric(df_city["Valor"], errors="coerce")
    df_city = df_city.dropna(subset=["Valor"])

    mask_fecha_city = (df_city.index >= start_date) & (df_city.index <= end_date)
    df_city = df_city[mask_fecha_city]

    if df_city.empty:
        return ranking_fig, empty_figure("Sin datos para la comuna seleccionada"), []

    city_grouped = (
        df_city.groupby("Parametro")
        .agg(
            Valor_max=("Valor", "max"),
            Limite=("Limite", lambda x: pd.to_numeric(x, errors="coerce").max()),
        )
        .reset_index()
    )

    if city_grouped.empty:
        return ranking_fig, empty_figure("Sin datos para la comuna seleccionada"), []

    # Ratio respecto a norma (si hay límite)
    def _ratio(row):
        lim = row["Limite"]
        if pd.notnull(lim) and lim > 0:
            return row["Valor_max"] / lim
        return None

    city_grouped["Ratio_norma"] = city_grouped.apply(_ratio, axis=1)

    # Radar: usamos Ratio_norma cuando exista, si no, normalizamos por el máx global del parámetro
    city_grouped_sorted = city_grouped.sort_values("Parametro")
    categorias = city_grouped_sorted["Parametro"].tolist()

    # Normalización alternativa si falta Ratio
    ratios = []
    for _, row in city_grouped_sorted.iterrows():
        r = row["Ratio_norma"]
        if pd.notnull(r):
            ratios.append(r)
        else:
            # normalizar por el máximo global de ese parámetro en el rango
            df_param_all = df[(df["Parametro"] == row["Parametro"])].copy()
            df_param_all["Valor"] = pd.to_numeric(
                df_param_all["Valor"], errors="coerce"
            )
            df_param_all = df_param_all.dropna(subset=["Valor"])
            mask_fecha_all = (df_param_all.index >= start_date) & (
                df_param_all.index <= end_date
            )
            df_param_all = df_param_all[mask_fecha_all]
            if not df_param_all.empty:
                max_global = df_param_all["Valor"].max()
                if max_global > 0:
                    ratios.append(row["Valor_max"] / max_global)
                else:
                    ratios.append(0.0)
            else:
                ratios.append(0.0)

    if len(ratios) == 0:
        radar_fig = empty_figure("Sin datos para la comuna seleccionada")
        return ranking_fig, radar_fig, []

    # Cerrar el polígono
    categorias_cerradas = categorias + [categorias[0]]
    ratios_cerrados = ratios + [ratios[0]]

    max_ratio = max(ratios)
    radial_max = max(1.2, max_ratio * 1.1)

    radar_fig = go.Figure()
    radar_fig.add_trace(
        go.Scatterpolar(
            r=ratios_cerrados,
            theta=categorias_cerradas,
            fill="toself",
            name=selected_city,
        )
    )
    radar_fig.update_layout(
        title=f"Perfil de máximos normalizados - {selected_city}",
        polar=dict(radialaxis=dict(visible=True, range=[0, radial_max])),
        showlegend=False,
    )

    # Tabla para la comuna seleccionada
    table_df = city_grouped_sorted.copy()
    table_df.rename(
        columns={
            "Parametro": "Parámetro",
            "Valor_max": "Valor máximo",
            "Limite": "Límite",
            "Ratio_norma": "Ratio vs norma",
        },
        inplace=True,
    )

    # Formatear ratio
    table_df["Ratio vs norma"] = table_df["Ratio vs norma"].apply(
        lambda x: round(x, 2) if pd.notnull(x) else None
    )

    data_table = table_df.to_dict("records")

    return ranking_fig, radar_fig, data_table


if __name__ == "__main__":
    app.run_server(debug=True)
