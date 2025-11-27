import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from datetime import timedelta

# ==========================================
# CONFIGURACIÓN DE LA PÁGINA
# ==========================================
st.set_page_config(
    page_title="Monitor de Calidad del Agua - Chile",
    layout="wide",
    page_icon="💧",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados para métricas
st.markdown("""
<style>
    [data-testid="stMetricValue"] {
        font-size: 24px;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# CARGA DE DATOS
# ==========================================
@st.cache_data
def load_data():
    """Carga los datos originales, formatea fechas y limpia numéricos."""
    script_dir = os.path.abspath(os.path.dirname(__file__))
    # Intenta cargar desde carpeta data (estructura repo) o raíz
    data_path_folder = os.path.join(script_dir, "data", "rawdata_20240409.csv")
    data_path_root = "rawdata_20240409 (1).csv" 
    
    if os.path.exists(data_path_folder):
        df = pd.read_csv(data_path_folder)
    elif os.path.exists(data_path_root):
        df = pd.read_csv(data_path_root)
    else:
        # Intento final nombre genérico
        try:
            df = pd.read_csv("rawdata_20240409.csv")
        except:
            st.error("⚠️ Error Crítico: No se encontró el archivo de datos 'rawdata_20240409.csv'.")
            return pd.DataFrame()

    # Conversión de fechas y numéricos
    df["DateTime"] = pd.to_datetime(df["DateTime"])
    
    if df["Valor"].dtype == object:
        df["Valor"] = df["Valor"].astype(str).str.replace(",", ".").replace("Ausencia", "0")
        df["Valor"] = pd.to_numeric(df["Valor"], errors="coerce")
    
    return df.sort_values("DateTime")

df_raw = load_data()

if df_raw.empty:
    st.stop()

# ==========================================
# REFERENCIAS Y CONSTANTES
# ==========================================
LIMS = {
    "PH": {"lim_min": 6.5, "lim_max": 8.5, "ref": 8.5},
    "ARSENICO": {"lim_max": 0.01, "ref": 0.01},
    "CLORO LIBRE RESIDUAL": {"lim_min": 0.2, "lim_max": 2.0, "ref": 2.0},
    "COLIFORMES TOTALES": {"lim_max": 0, "ref": 1},
    "TURBIEDAD": {"lim_max": 2.0, "ref": 2.0},
    "FLUORURO": {"lim_max": 1.5, "ref": 1.5},
    "RAZON NITRATOS + NITRITOS": {"lim_max": 1.0, "ref": 1.0},
    "TRIHALOMETANOS": {"lim_max": 1.0, "ref": 1.0},
    "SULFATOS": {"lim_max": 250, "ref": 250},
    "HIERRO": {"lim_max": 0.3, "ref": 0.3},
    "MANGANESO": {"lim_max": 0.1, "ref": 0.1},
    "NITRATOS": {"lim_max": 50, "ref": 50},
    "CINC": {"lim_max": 3.0, "ref": 3.0},
    "COBRE": {"lim_max": 2.0, "ref": 2.0},
}

param_disc = ["OLOR", "COLOR VERDADERO", "SABOR"]
all_params = [p for p in df_raw["Parametro"].unique() if p not in param_disc]
all_comunas = sorted(df_raw["Comuna"].unique())

# ==========================================
# BARRA LATERAL (Navegación y Filtros)
# ==========================================
st.sidebar.title("🚰 Monitor SISS")
modo_visualizacion = st.sidebar.radio(
    "Seleccione análisis:",
    options=["Análisis Temporal", "Radar de Riesgos (Comparativo)"],
    index=0
)

st.sidebar.markdown("---")

# ==========================================
# VISTA 1: ANÁLISIS TEMPORAL
# ==========================================
if modo_visualizacion == "Análisis Temporal":
    
    st.sidebar.subheader("🛠️ Filtros")
    selected_param = st.sidebar.selectbox("Parámetro", all_params, index=0)
    
    # Multiselect inteligente: si hay una sola comuna en data, la selecciona por defecto
    default_comuna = [all_comunas[0]] if all_comunas else []
    selected_comunas = st.sidebar.multiselect("Comunas", all_comunas, default=default_comuna)
    
    min_date = df_raw["DateTime"].min()
    max_date = df_raw["DateTime"].max()
    date_range = st.sidebar.date_input(
        "Rango de Fechas",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    st.title(f"📈 Evolución: {selected_param}")
    
    if not selected_comunas:
        st.warning("Seleccione al menos una comuna en la barra lateral.")
    else:
        # Filtrado
        mask = (
            (df_raw["Parametro"] == selected_param) & 
            (df_raw["Comuna"].isin(selected_comunas)) &
            (df_raw["DateTime"] >= pd.to_datetime(date_range[0])) &
            (df_raw["DateTime"] <= pd.to_datetime(date_range[1]))
        )
        df_filtered = df_raw[mask].copy()
        
        if df_filtered.empty:
            st.info("No hay datos disponibles para esta selección.")
        else:
            # 1. Gráfico Principal
            fig_line = px.line(
                df_filtered, x="DateTime", y="Valor", color="Comuna", 
                markers=True, title=f"Serie Temporal - {selected_param}",
                labels={"Valor": f"Valor ({df_filtered['Unidad'].iloc[0]})"}
            )
            
            # Líneas de límite normativo
            if selected_param in LIMS:
                limits = LIMS[selected_param]
                if "lim_max" in limits and limits["lim_max"] > 0:
                    fig_line.add_hline(y=limits["lim_max"], line_dash="dash", line_color="red", annotation_text="Límite Max")
                if "lim_min" in limits:
                    fig_line.add_hline(y=limits["lim_min"], line_dash="dash", line_color="orange", annotation_text="Límite Min")

            st.plotly_chart(fig_line, use_container_width=True)
            
            # 2. Métricas y KPIs (Mejora visual)
            st.subheader("📊 Estado Actual (Último Registro)")
            cols_kpi = st.columns(len(selected_comunas))
            
            # Mostrar tarjeta KPI para cada comuna seleccionada (hasta 4 columnas para no saturar)
            # Si son muchas, se mostrarán en filas
            for idx, comuna in enumerate(selected_comunas):
                col_idx = idx % 4
                if idx > 0 and idx % 4 == 0:
                    cols_kpi = st.columns(4) # Nueva fila
                
                df_c = df_filtered[df_filtered["Comuna"] == comuna].sort_values("DateTime")
                if not df_c.empty:
                    last_val = df_c.iloc[-1]["Valor"]
                    last_date_str = df_c.iloc[-1]["DateTime"].strftime("%d/%m/%Y")
                    
                    # Calcular delta si hay más de 1 dato
                    delta = None
                    if len(df_c) > 1:
                        prev_val = df_c.iloc[-2]["Valor"]
                        delta = round(last_val - prev_val, 4)
                    
                    with cols_kpi[col_idx]:
                        st.metric(
                            label=f"{comuna}",
                            value=f"{last_val} {df_c.iloc[0]['Unidad']}",
                            delta=delta,
                            help=f"Fecha medición: {last_date_str}"
                        )

            # 3. Estadísticas y Descarga
            col1, col2 = st.columns([1, 1])
            with col1:
                st.subheader("Resumen Estadístico")
                stats = df_filtered.groupby("Comuna")["Valor"].describe()[['count', 'mean', 'max', 'min']]
                st.dataframe(stats.style.format("{:.3f}"), use_container_width=True)
                
            with col2:
                st.subheader("Exportar Datos")
                st.write("Descargue los datos filtrados para su propio análisis.")
                csv = df_filtered.to_csv(index=False).encode('utf-8')
                st.download_button(
                    "📥 Descargar CSV Filtrado",
                    data=csv,
                    file_name=f"datos_{selected_param}.csv",
                    mime="text/csv",
                    key='download-csv'
                )

# ==========================================
# VISTA 2: RADAR DE RIESGOS
# ==========================================
elif modo_visualizacion == "Radar de Riesgos (Comparativo)":
    
    st.sidebar.subheader("⚙️ Configuración Radar")
    st.sidebar.info(
        "Muestra el **máximo histórico del último año móvil** (últimos 365 días de datos) "
        "normalizado respecto a la norma. Valor > 1.0 indica fuera de norma."
    )
    
    radar_comunas = st.sidebar.multiselect(
        "Comunas a comparar", all_comunas, 
        default=all_comunas[:2] if len(all_comunas) > 1 else all_comunas
    )
    
    params_avail = [p for p in all_params if p in LIMS]
    radar_params = st.sidebar.multiselect(
        "Parámetros", params_avail, default=params_avail
    )

    st.title("🕸️ Radar de Riesgos Multidimensional")
    
    if not radar_comunas or not radar_params:
        st.warning("Seleccione al menos una comuna y parámetros.")
    else:
        fig_radar = go.Figure()
        data_exists = False
        
        # Guardar datos para tabla resumen
        summary_data = []

        for comuna in radar_comunas:
            r_values = []
            theta_values = []
            
            df_comuna = df_raw[df_raw["Comuna"] == comuna]
            if df_comuna.empty: continue
                
            # Buscar último año de datos de ESA comuna
            last_date = df_comuna["DateTime"].max()
            start_date = last_date - timedelta(days=365)
            df_last_year = df_comuna[df_comuna["DateTime"] >= start_date]
            
            if df_last_year.empty: continue
            
            # Calcular scores
            comuna_has_data = False
            for param in radar_params:
                df_p = df_last_year[df_last_year["Parametro"] == param]
                if not df_p.empty:
                    max_val = df_p["Valor"].max()
                    ref_val = LIMS[param]["ref"]
                    norm_score = max_val / ref_val if ref_val > 0 else 0
                    
                    r_values.append(norm_score)
                    theta_values.append(param)
                    comuna_has_data = True
                    
                    # Agregar a resumen
                    summary_data.append({
                        "Comuna": comuna,
                        "Parámetro": param,
                        "Max Valor (Año)": max_val,
                        "Límite Ref": ref_val,
                        "Indice Riesgo": round(norm_score, 2),
                        "Última Fecha": last_date.date()
                    })
                else:
                    r_values.append(0)
                    theta_values.append(param)
            
            if comuna_has_data:
                r_values.append(r_values[0])
                theta_values.append(theta_values[0])
                fig_radar.add_trace(go.Scatterpolar(
                    r=r_values, theta=theta_values, fill='toself', name=f"{comuna}"
                ))
                data_exists = True

        if data_exists:
            # Línea de límite = 1.0
            line_ref = [1.0] * (len(radar_params) + 1)
            line_theta = radar_params + [radar_params[0]]
            fig_radar.add_trace(go.Scatterpolar(
                r=line_ref, theta=line_theta, mode='lines', 
                line=dict(color='red', dash='dash', width=2),
                name='Límite (100%)', hoverinfo='skip'
            ))

            # --- CORRECCIÓN APLICADA AQUÍ ---
            # Calcular rango dinámico de manera segura
            max_val_found = 0
            for trace in fig_radar.data:
                # trace.r devuelve una tupla/lista de valores
                if hasattr(trace, 'r') and trace.r:
                    current_max = max(trace.r) # Maximo de la lista de esta traza
                    if current_max > max_val_found:
                        max_val_found = current_max
            
            upper_range = max(2.0, max_val_found * 1.1)

            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, upper_range])),
                height=600, showlegend=True
            )
            # --------------------------------
            
            st.plotly_chart(fig_radar, use_container_width=True)
            
            # Tabla de detalles debajo del radar
            st.subheader("Detalle de valores críticos")
            df_summary = pd.DataFrame(summary_data)
            if not df_summary.empty:
                # Ordenar por riesgo descendente
                df_summary = df_summary.sort_values("Indice Riesgo", ascending=False)
                
                # Función para colorear filas peligrosas
                def highlight_risk(val):
                    color = 'red' if val > 1.0 else 'orange' if val > 0.8 else 'green'
                    return f'color: {color}; font-weight: bold'

                st.dataframe(
                    df_summary.style.applymap(highlight_risk, subset=['Indice Riesgo'])
                    .format({"Max Valor (Año)": "{:.4f}", "Indice Riesgo": "{:.2f}"}),
                    use_container_width=True, hide_index=True
                )
        else:
            st.info("No hay datos suficientes en el último año móvil para generar el radar.")

# ==========================================
# FOOTER / ACERCA DE
# ==========================================
st.sidebar.markdown("---")
with st.sidebar.expander("ℹ️ Acerca de"):
    st.markdown("""
    **Monitor de Calidad de Agua**
    
    Esta herramienta visualiza datos públicos de calidad del agua.
    
    - **Código Fuente:** [GitHub: chile-waterquality](https://github.com/luchoplaza/chile-waterquality)
    - **Datos:** Extraídos de reportes SISS.
    - **Desarrollador:** Lucho Plaza.
    """)
    st.caption("v1.2 - Streamlit Edition")
