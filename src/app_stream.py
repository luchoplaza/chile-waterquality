import streamlit as st
import pandas as pd
import plotly.express as px
import os

# Configuración de la página
st.set_page_config(page_title="Water Quality Chile - SISS Data", layout="wide")

# Nota: Coloca tu archivo style.css en la carpeta "assets" para que Streamlit lo cargue automáticamente.

@st.cache_data
def load_data():
    # Carga los datos
    script_dir=os.path.abspath(os.path.dirname(__file__))
    data_path = os.path.join(script_dir,'data','rawdata_20240409.csv')
    df = pd.read_csv(data_path)
    df['DateTime'] = pd.to_datetime(df['DateTime'])
    df = df.set_index('DateTime', drop=True)
    return df

df = load_data()

# Lista de parámetros y comunas
param_disc = ["OLOR", "COLOR VERDADERO", "SABOR"]
empresas = df['Empresa'].unique().tolist()
comunas = df['Comuna'].unique().tolist()
parametros = df['Parametro'].unique().tolist()[2:]
parametros = [x for x in parametros if x not in param_disc]

# Límites de referencia
lims = {
    'PH': {'lim_min': 6.5, 'lim_max': 8.5},
    'ARSENICO': {'lim_min': 0, 'lim_max': 0.01},
    'CLORO LIBRE RESIDUAL': {'lim_min': 0.2, 'lim_max': 2.0}
}

# Título principal
st.markdown("<h1 style='text-align: center;'>💧 Water Quality Chile</h1>", unsafe_allow_html=True)
st.markdown(
    """
    Dashboard de la calidad fisicoquímica del agua potable en las diferentes comunas de Chile, 
    de acuerdo a la información reportada por las empresas sanitarias (SISS, 2023).  
    Creado por: [Luis Plaza A.](https://www.linkedin.com/in/lplazaalvarez/)  
    ¡Dudas y/o comentarios, contáctame!
    """,
    unsafe_allow_html=True,
)

# Panel lateral (sidebar) con filtros
st.sidebar.header("Filtros")

selected_cities = st.sidebar.multiselect(
    "Seleccione una ciudad:",
    options=comunas,
    default=[comunas[0]] if comunas else []
)

default_index = 6 if len(parametros) > 6 else 0
selected_parameter = st.sidebar.selectbox(
    "Seleccione parámetro:",
    options=parametros,
    index=default_index
)

# Rango de fechas
min_date = df.index.min().date()
max_date = df.index.max().date()

# Inyectar CSS para ocultar el calendario emergente
st.markdown("""
<style>
/* Oculta el popover (calendario) de los widgets DatePicker */
[data-baseweb="popover"] {
    display: none !important;
}
</style>
""", unsafe_allow_html=True)

date_range = st.sidebar.date_input(
    "Seleccione un rango de fechas (YYYY/MM/DD):",
    value=(min_date, max_date),
    min_value=min_date,
    max_value=max_date
)
if isinstance(date_range, tuple) and len(date_range) == 2:
    start_date, end_date = date_range
else:
    start_date = end_date = date_range

if start_date > end_date:
    st.sidebar.error("La fecha inicio debe ser anterior a la fecha fin.")

# Filtrado de datos
if not selected_cities:
    st.warning("No se ha seleccionado ninguna ciudad. Se usará la primera disponible.")
    selected_cities = [comunas[0]] if comunas else []

mask = df['Comuna'].isin(selected_cities) & (df['Parametro'] == selected_parameter)
filtered_data = df[mask].copy()

mask_date = (filtered_data.index.date >= start_date) & (filtered_data.index.date <= end_date)
filtered_data = filtered_data[mask_date]

if filtered_data.empty:
    st.warning("No hay valores para estas condiciones.")
else:
    # Convertir a float y ordenar por fecha
    try:
        filtered_data['Valor'] = filtered_data['Valor'].astype(float)
    except Exception as e:
        st.error(f"Error al convertir los valores a float: {e}")

    filtered_data.sort_index(inplace=True)
    unidad = filtered_data['Unidad'].unique()[0] if 'Unidad' in filtered_data.columns else ""

    # ===== Gráfico de línea =====
    line_fig = px.line(
        filtered_data,
        x=filtered_data.index,
        y='Valor',
        color='Comuna',
        title=f'{selected_parameter}',
        markers=True
    )
    line_fig.update_layout(
        legend=dict(y=-0.15, orientation="h"),
        margin=dict(r=30),
        xaxis_title="Fecha",
        yaxis_title=f"Valor ({unidad})"
    )
    line_fig.update_xaxes(tickmode="auto", nticks=8)

    # Líneas de referencia si el parámetro está en lims
    if selected_parameter in lims.keys():
        line_fig.add_hline(
            y=lims[selected_parameter]['lim_max'],
            line_dash="dash",
            line_color="red",
            annotation_text="Límite Superior",
            annotation_position="bottom right"
        )
        line_fig.add_hline(
            y=lims[selected_parameter]['lim_min'],
            line_dash="dash",
            line_color="red",
            annotation_text="Límite Inferior",
            annotation_position="bottom right"
        )
    else:
        if 'Limite' in filtered_data.columns:
            try:
                limite_value = float(filtered_data['Limite'].unique()[0])
                line_fig.add_hline(
                    y=limite_value,
                    line_dash="dash",
                    line_color="red",
                    annotation_text="Límite",
                    annotation_position="bottom right"
                )
            except:
                pass

    # ===== Histograma =====
    hist_fig = px.histogram(
        filtered_data,
        x='Valor',
        color='Comuna',
        opacity=0.5,
        histnorm='percent',
        title=f'Histograma - {selected_parameter}',
    )
    hist_fig.update_layout(
        legend=dict(y=-0.15, orientation="h"),
        margin=dict(r=30),
        yaxis_title="Porcentaje (%)",
        xaxis_title=f"Valor ({unidad})"
    )
    hist_fig.update_xaxes(tickmode="auto", nticks=6)

    if selected_parameter in lims.keys():
        hist_fig.add_vline(
            x=lims[selected_parameter]['lim_max'],
            line_dash="dash",
            line_color="red",
            annotation_text="Límite Superior",
            annotation_position="top"
        )
        hist_fig.add_vline(
            x=lims[selected_parameter]['lim_min'],
            line_dash="dash",
            line_color="red",
            annotation_text="Límite Inferior",
            annotation_position="top"
        )
    else:
        if 'Limite' in filtered_data.columns:
            try:
                limite_value = float(filtered_data['Limite'].unique()[0])
                hist_fig.add_vline(
                    x=limite_value,
                    line_dash="dash",
                    line_color="red",
                    annotation_text="Límite",
                    annotation_position="top"
                )
            except:
                pass

    # ===== Estadísticas numéricas =====
    # Agrupamos por comuna y calculamos describe()
    # Esto devuelve estadísticos en filas, por lo que usamos .T para transponer
    # y renombramos ejes para facilitar la lectura.
    stats_numeric = filtered_data.groupby('Comuna')['Valor'].describe().T.round(2)
    stats_numeric = stats_numeric.rename_axis("Variable").reset_index()

    # Renombrar las filas que produce describe()
    rename_map = {
        'count': 'Conteo',
        'mean': 'Media',
        'std': 'Desv.Est',
        'min': 'Min',
        '25%': 'Q1',
        '50%': 'Q2',
        '75%': 'Q3',
        'max': 'Max'
    }
    stats_numeric['Variable'] = stats_numeric['Variable'].map(rename_map)

    # ===== Último valor y fecha por comuna =====
    last_values_list = []
    for city in selected_cities:
        city_data = filtered_data[filtered_data['Comuna'] == city]
        if not city_data.empty:
            last_val = round(city_data['Valor'].iloc[-1], 2)
            last_date = city_data.index[-1].strftime("%b-%Y")
            last_values_list.append({
                "Comuna": city,
                "Último Valor": last_val,
                "Última Fecha": last_date
            })

    last_values_df = pd.DataFrame(last_values_list)

    # ===== Visualización final =====
    st.plotly_chart(line_fig, use_container_width=True)
    
    col1, col2 = st.columns(2)
    with col1:
        st.plotly_chart(hist_fig, use_container_width=True)

    with col2:
        st.subheader("Estadísticas Numéricas")
        st.dataframe(stats_numeric)

        if not last_values_df.empty:
            st.subheader("Última Medición")
            st.dataframe(last_values_df)
