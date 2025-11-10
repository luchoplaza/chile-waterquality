import os

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Configuración de la página
st.set_page_config(page_title="Water Quality Chile - SISS Data", layout="wide")


@st.cache_data
def load_data():
    """Carga los datos originales desde la carpeta data."""
    script_dir = os.path.abspath(os.path.dirname(__file__))
    data_path = os.path.join(script_dir, "data", "rawdata_20240409.csv")
    df = pd.read_csv(data_path)
    df["DateTime"] = pd.to_datetime(df["DateTime"])
    df = df.set_index("DateTime", drop=True)
    return df


df = load_data()

# Lista de parámetros y comunas
param_disc = ["OLOR", "COLOR VERDADERO", "SABOR"]  # Se ignoran estos parámetros discretos
empresas = df["Empresa"].unique().tolist()
comunas = df["Comuna"].unique().tolist()
parametros = df["Parametro"].unique().tolist()[2:]
parametros = [x for x in parametros if x not in param_disc]

# Límites de referencia
lims = {
    "PH": {"lim_min": 6.5, "lim_max": 8.5},
    "ARSENICO": {"lim_min": 0, "lim_max": 0.01},
    "CLORO LIBRE RESIDUAL": {"lim_min": 0.2, "lim_max": 2.0},
}

# Título principal
st.markdown(
    "<h1 style='text-align: center;'>💧 Calidad de agua potable Chile</h1>",
    unsafe_allow_html=True,
)
st.markdown(
    """
    Dashboard de la calidad fisicoquímica del agua potable en las diferentes comunas de Chile, 
    de acuerdo a la información reportada por las empresas sanitarias (SISS, 2023).  
    Creado por: [Luis Plaza A.](https://www.linkedin.com/in/lplazaalvarez/)  
    ¡Dudas y/o comentarios, contáctame!.  
    Desde celulares, para seleccionar comuna/parametro/fechas desplegar menú lateral en la esquina superior izquierda.
    """,
    unsafe_allow_html=True,
)

# Panel lateral (sidebar) con filtros generales
st.sidebar.header("Filtros")

selected_cities = st.sidebar.multiselect(
    "Seleccione una ciudad:",
    options=comunas,
    default=[comunas[0]] if comunas else [],
)

default_index = 6 if len(parametros) > 6 else 0
selected_parameter = st.sidebar.selectbox(
    "Seleccione parámetro:",
    options=parametros,
    index=default_index,
)

# Rango de fechas
min_date = df.index.min().date()
max_date = df.index.max().date()
st.sidebar.write("Ingrese rango de fechas en formato YYYY-MM-DD:")
start_str = st.sidebar.text_input("Fecha inicio:", value=min_date)
end_str = st.sidebar.text_input("Fecha fin:", value=max_date)

# Convertir las cadenas a tipo fecha
try:
    start_date = pd.to_datetime(start_str).date()
    end_date = pd.to_datetime(end_str).date()
    if start_date > end_date:
        st.sidebar.error("La fecha de inicio no puede ser mayor que la fecha fin.")
except ValueError:
    st.sidebar.error("Formato de fecha inválido. Usa YYYY-MM-DD.")
    # Si falla, fijamos algunos valores por defecto:
    start_date = df.index.min().date()
    end_date = df.index.max().date()

# Controles adicionales para clasificación por máximos
st.sidebar.header("Clasificación por máximos")

top_n = st.sidebar.slider(
    "Top N comunas:",
    min_value=5,
    max_value=30,
    step=5,
    value=10,
)

opciones_comuna_ficha = ["(Usar comuna con máximo más alto)"] + comunas
selected_city_max_label = st.sidebar.selectbox(
    "Comuna para ficha de máximos:",
    options=opciones_comuna_ficha,
    index=0,
)
selected_city_max = (
    None if selected_city_max_label == "(Usar comuna con máximo más alto)" else selected_city_max_label
)

# Disclaimer de comunas
st.sidebar.markdown("\n💧 Notas:\n", unsafe_allow_html=True)
st.sidebar.markdown(
    """
    Algunas comunas pueden estar embebidas en servicios más grandes como "Gran Santiago" 
    que contiene a Santiago Centro, Ñuñoa, Providencia, etc.  
    Actualmente no se tiene el detalle de qué comunas componen cada servicio/concesión de agua potable.
    """,
    unsafe_allow_html=True,
)

# =============================================================================
# 1) Análisis temporal - filtrado por ciudades seleccionadas
# =============================================================================

if not selected_cities:
    st.warning("No se ha seleccionado ninguna ciudad. Se usará la primera disponible.")
    selected_cities = [comunas[0]] if comunas else []

mask_time = (df.index.date >= start_date) & (df.index.date <= end_date)

mask_temporal = df["Comuna"].isin(selected_cities) & (df["Parametro"] == selected_parameter)
filtered_data = df[mask_temporal & mask_time].copy()

# Intentar convertir a float
if not filtered_data.empty:
    try:
        filtered_data["Valor"] = filtered_data["Valor"].astype(float)
    except Exception as e:
        st.error(f"Error al convertir los valores a float: {e}")

# =============================================================================
# 2) Clasificación por máximos - ranking global por parámetro en el rango
# =============================================================================

# Filtrar solo por parámetro y rango de fechas (todas las comunas)
df_param = df[(df["Parametro"] == selected_parameter) & mask_time].copy()

ranking_fig = None
maxima_table = None

if not df_param.empty:
    df_param["Valor"] = pd.to_numeric(df_param["Valor"], errors="coerce")
    df_param = df_param.dropna(subset=["Valor"])  # quitamos NaN en Valor

    if not df_param.empty:
        # Índice de la fila con máximo por comuna
        try:
            idxmax_series = df_param.groupby("Comuna")["Valor"].idxmax()
            grouped = df_param.loc[idxmax_series, ["Comuna", "Valor", "Limite"]].copy()
            grouped.rename(columns={"Valor": "Valor máximo"}, inplace=True)
            grouped["Fecha máximo"] = grouped.index.strftime("%Y-%m-%d")

            # Asegurar columna límite numérica
            grouped["Limite"] = pd.to_numeric(grouped["Limite"], errors="coerce")

            # Ordenar y tomar top N
            grouped = grouped.sort_values("Valor máximo", ascending=False)
            top_n_int = int(top_n) if top_n is not None else 10
            top_grouped = grouped.head(top_n_int)

            # Figura de ranking
            ranking_fig = px.bar(
                top_grouped,
                x="Valor máximo",
                y="Comuna",
                orientation="h",
                text="Valor máximo",
                title=f"Ranking de comunas por máximo de {selected_parameter}",
            )
            ranking_fig.update_layout(
                yaxis={"categoryorder": "total ascending"},
                xaxis_title="Valor máximo",
                margin=dict(l=80, r=40, t=60, b=40),
            )

            # Línea de límite normativo si existe
            limites_unicos = (
                pd.to_numeric(df_param["Limite"], errors="coerce").dropna().unique()
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

            maxima_table = grouped.reset_index(drop=True)
        except Exception as e:  # protección defensiva
            st.error(f"No se pudo calcular el ranking de máximos: {e}")

# =============================================================================
# 3) Ficha de comuna - perfil de máximos por parámetro (Radar)
# =============================================================================

radar_fig = None
city_max_table = None
city_for_profile = selected_city_max

if maxima_table is not None and not maxima_table.empty:
    # Si no se selecciona comuna explícita, usar la de máximo más alto
    if city_for_profile is None:
        city_for_profile = maxima_table.iloc[0]["Comuna"]

    df_city = df[(df["Comuna"] == city_for_profile) & mask_time].copy()

    if not df_city.empty:
        df_city["Valor"] = pd.to_numeric(df_city["Valor"], errors="coerce")
        df_city = df_city.dropna(subset=["Valor"])

        if not df_city.empty:
            # Agrupar por parámetro: máximo y límite
            city_grouped = (
                df_city.groupby("Parametro")
                .agg(
                    Valor_max=("Valor", "max"),
                    Limite=(
                        "Limite",
                        lambda x: pd.to_numeric(x, errors="coerce").max(),
                    ),
                )
                .reset_index()
            )

            # Excluir parámetros discretos del radar
            city_grouped = city_grouped[~city_grouped["Parametro"].isin(param_disc)]

            if not city_grouped.empty:
                # Ratio contra norma (si existe límite > 0)
                def _ratio(row):
                    lim = row["Limite"]
                    if pd.notnull(lim) and lim > 0:
                        return row["Valor_max"] / lim
                    return None

                city_grouped["Ratio_norma"] = city_grouped.apply(_ratio, axis=1)

                # Ordenar por nombre de parámetro para un radar estable
                city_grouped_sorted = city_grouped.sort_values("Parametro")
                categorias = city_grouped_sorted["Parametro"].tolist()

                ratios = []
                for _, row in city_grouped_sorted.iterrows():
                    r = row["Ratio_norma"]
                    if pd.notnull(r):
                        ratios.append(r)
                    else:
                        # Normalización alternativa: respecto al máximo global
                        df_param_all = df[
                            (df["Parametro"] == row["Parametro"]) & mask_time
                        ].copy()
                        df_param_all["Valor"] = pd.to_numeric(
                            df_param_all["Valor"], errors="coerce"
                        )
                        df_param_all = df_param_all.dropna(subset=["Valor"])
                        if not df_param_all.empty:
                            max_global = df_param_all["Valor"].max()
                            ratios.append(
                                row["Valor_max"] / max_global if max_global > 0 else 0.0
                            )
                        else:
                            ratios.append(0.0)

                if ratios:
                    # Cerrar el polígono del radar
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
                            name=city_for_profile,
                        )
                    )
                    radar_fig.update_layout(
                        title=f"Perfil de máximos normalizados - {city_for_profile}",
                        polar=dict(
                            radialaxis=dict(visible=True, range=[0, radial_max])
                        ),
                        showlegend=False,
                    )

                    # Tabla para la comuna
                    city_max_table = city_grouped_sorted.copy()
                    city_max_table.rename(
                        columns={
                            "Parametro": "Parámetro",
                            "Valor_max": "Valor máximo",
                            "Limite": "Límite",
                            "Ratio_norma": "Ratio vs norma",
                        },
                        inplace=True,
                    )
                    city_max_table["Ratio vs norma"] = city_max_table[
                        "Ratio vs norma"
                    ].apply(lambda x: round(x, 2) if pd.notnull(x) else None)

# =============================================================================
# Layout principal con pestañas
# =============================================================================

tab1, tab2 = st.tabs(["Análisis temporal", "Clasificación por máximos"])

with tab1:
    if filtered_data.empty:
        st.warning("No hay valores para estas condiciones.")
    else:
        filtered_data = filtered_data.sort_index()
        unidad = (
            filtered_data["Unidad"].unique()[0]
            if "Unidad" in filtered_data.columns
            else ""
        )

        # ===== Gráfico de línea =====
        line_fig = px.line(
            filtered_data,
            x=filtered_data.index,
            y="Valor",
            color="Comuna",
            title=f"{selected_parameter}",
            markers=True,
        )
        line_fig.update_layout(
            legend=dict(y=-0.15, orientation="h"),
            margin=dict(r=30),
            xaxis_title="Fecha",
            yaxis_title=f"Valor ({unidad})",
        )
        line_fig.update_xaxes(tickmode="auto", nticks=8)

        # Líneas de referencia si el parámetro está en lims
        if selected_parameter in lims.keys():
            line_fig.add_hline(
                y=lims[selected_parameter]["lim_max"],
                line_dash="dash",
                line_color="red",
                annotation_text="Límite Superior",
                annotation_position="bottom right",
            )
            line_fig.add_hline(
                y=lims[selected_parameter]["lim_min"],
                line_dash="dash",
                line_color="red",
                annotation_text="Límite Inferior",
                annotation_position="bottom right",
            )
        else:
            if "Limite" in filtered_data.columns:
                try:
                    limite_value = float(filtered_data["Limite"].unique()[0])
                    line_fig.add_hline(
                        y=limite_value,
                        line_dash="dash",
                        line_color="red",
                        annotation_text="Límite",
                        annotation_position="bottom right",
                    )
                except Exception:
                    pass

        # ===== Histograma =====
        hist_fig = px.histogram(
            filtered_data,
            x="Valor",
            color="Comuna",
            opacity=0.5,
            histnorm="percent",
            title=f"Histograma - {selected_parameter}",
        )
        hist_fig.update_layout(
            legend=dict(y=-0.15, orientation="h"),
            margin=dict(r=30),
            yaxis_title="Porcentaje (%)",
            xaxis_title=f"Valor ({unidad})",
        )
        hist_fig.update_xaxes(tickmode="auto", nticks=6)

        if selected_parameter in lims.keys():
            hist_fig.add_vline(
                x=lims[selected_parameter]["lim_max"],
                line_dash="dash",
                line_color="red",
                annotation_text="Límite Superior",
                annotation_position="top",
            )
            hist_fig.add_vline(
                x=lims[selected_parameter]["lim_min"],
                line_dash="dash",
                line_color="red",
                annotation_text="Límite Inferior",
                annotation_position="top",
            )
        else:
            if "Limite" in filtered_data.columns:
                try:
                    limite_value = float(filtered_data["Limite"].unique()[0])
                    hist_fig.add_vline(
                        x=limite_value,
                        line_dash="dash",
                        line_color="red",
                        annotation_text="Límite",
                        annotation_position="top",
                    )
                except Exception:
                    pass

        # ===== Estadísticas numéricas =====
        stats_numeric = (
            filtered_data.groupby("Comuna")["Valor"].describe().T.round(2)
        )
        stats_numeric = stats_numeric.rename_axis("Variable").reset_index()

        rename_map = {
            "count": "Conteo",
            "mean": "Media",
            "std": "Desv.Est",
            "min": "Min",
            "25%": "Q1",
            "50%": "Q2",
            "75%": "Q3",
            "max": "Max",
        }
        stats_numeric["Variable"] = stats_numeric["Variable"].map(rename_map)

        # ===== Último valor y fecha por comuna =====
        last_values_list = []
        for city in selected_cities:
            city_data = filtered_data[filtered_data["Comuna"] == city]
            if not city_data.empty:
                last_val = round(city_data["Valor"].iloc[-1], 2)
                last_date = city_data.index[-1].strftime("%b-%Y")
                last_values_list.append(
                    {
                        "Comuna": city,
                        "Último Valor": last_val,
                        "Última Fecha": last_date,
                    }
                )

        last_values_df = pd.DataFrame(last_values_list)

        # ===== Visualización final pestaña 1 =====
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

with tab2:
    st.subheader(f"Ranking por máximos - parámetro: {selected_parameter}")

    if ranking_fig is None:
        st.info(
            "No se pudo construir el ranking de máximos para el parámetro y rango de fechas seleccionados."
        )
    else:
        st.plotly_chart(ranking_fig, use_container_width=True)

        # Mostrar tabla de máximos globales por comuna (opcional)
        with st.expander("Ver tabla de máximos por comuna"):
            st.dataframe(maxima_table)

    st.markdown("---")
    st.subheader("Ficha de comuna - perfil de máximos por parámetro")

    if radar_fig is None or city_max_table is None:
        st.info(
            "No hay datos suficientes para construir la ficha de comuna en el rango seleccionado."
        )
    else:
        col1, col2 = st.columns(2)

        with col1:
            st.plotly_chart(radar_fig, use_container_width=True)

        with col2:
            st.subheader(f"Máximos por parámetro - {city_for_profile}")
            st.dataframe(city_max_table)
