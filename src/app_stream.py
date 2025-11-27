import os
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from datetime import timedelta

# Librerías de Machine Learning
try:
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA
    from sklearn.cluster import KMeans
except ImportError:
    st.error("Falta instalar scikit-learn. Por favor agrégalo a requirements.txt")

# ==========================================
# CONFIGURACIÓN DE LA PÁGINA
# ==========================================
st.set_page_config(
    page_title="Water Quality Chile - SISS Data",
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
# DICCIONARIO DE REGIONES (MAPEO)
# ==========================================
def get_comuna_region_map():
    """Retorna un diccionario para mapear Comuna -> Región."""
    # Nota: Este es un mapeo simplificado para las principales comunas/sistemas.
    # En un entorno real, idealmente cargarías esto desde un CSV maestro de geografía.
    return {
        # XV Arica y Parinacota
        "ARICA": "Arica y Parinacota", "PUTRE": "Arica y Parinacota",
        # I Tarapacá
        "IQUIQUE": "Tarapacá", "ALTO HOSPICIO": "Tarapacá", "PICA": "Tarapacá", "POZO ALMONTE": "Tarapacá",
        # II Antofagasta
        "ANTOFAGASTA": "Antofagasta", "CALAMA": "Antofagasta", "TOCOPILLA": "Antofagasta", "MEJILLONES": "Antofagasta", "TALTAL": "Antofagasta", "SAN PEDRO DE ATACAMA": "Antofagasta",
        # III Atacama
        "COPIAPO": "Atacama", "VALLENAR": "Atacama", "CALDERA": "Atacama", "CHAÑARAL": "Atacama", "DIEGO DE ALMAGRO": "Atacama", "HUASCO": "Atacama", "TIERRA AMARILLA": "Atacama",
        # IV Coquimbo
        "LA SERENA": "Coquimbo", "COQUIMBO": "Coquimbo", "OVALLE": "Coquimbo", "ILLAPEL": "Coquimbo", "VICUÑA": "Coquimbo", "SALAMANCA": "Coquimbo", "LOS VILOS": "Coquimbo", "ANDACOLLO": "Coquimbo",
        # V Valparaíso
        "VALPARAISO": "Valparaíso", "VIÑA DEL MAR": "Valparaíso", "QUILPUE": "Valparaíso", "VILLA ALEMANA": "Valparaíso", "SAN ANTONIO": "Valparaíso", "QUILLOTA": "Valparaíso", "LOS ANDES": "Valparaíso", "SAN FELIPE": "Valparaíso", "LA LIGUA": "Valparaíso", "LIMACHE": "Valparaíso", "CONCON": "Valparaíso", "QUINTERO": "Valparaíso", "PUCHUNCAVI": "Valparaíso", "CASABLANCA": "Valparaíso",
        # RM Metropolitana
        "SANTIAGO": "Metropolitana", "PUENTE ALTO": "Metropolitana", "MAIPU": "Metropolitana", "LA FLORIDA": "Metropolitana", "LAS CONDES": "Metropolitana", "SAN BERNARDO": "Metropolitana", "PEÑALOLEN": "Metropolitana", "QUILICURA": "Metropolitana", "SANTIAGO CENTRO": "Metropolitana", "PROVIDENCIA": "Metropolitana", "ÑUÑOA": "Metropolitana", "COLINA": "Metropolitana", "LAMPA": "Metropolitana", "TIL TIL": "Metropolitana", "BUIN": "Metropolitana", "PAINE": "Metropolitana", "TALAGANTE": "Metropolitana", "MELIPILLA": "Metropolitana", "CURACAVI": "Metropolitana", "HACIENDA BATUCO": "Metropolitana", "REINA NORTE": "Metropolitana", "EL COLORADO": "Metropolitana", "LA PARVA": "Metropolitana", "VALLE NEVADO": "Metropolitana",
        # VI O'Higgins
        "RANCAGUA": "O'Higgins", "MACHALI": "O'Higgins", "SAN FERNANDO": "O'Higgins", "RENGO": "O'Higgins", "SAN VICENTE": "O'Higgins", "SANTA CRUZ": "O'Higgins", "CHIMBARONGO": "O'Higgins", "PICHILEMU": "O'Higgins",
        # VII Maule
        "TALCA": "Maule", "CURICO": "Maule", "LINARES": "Maule", "CONSTITUCION": "Maule", "CAUQUENES": "Maule", "MOLINA": "Maule", "PARRAL": "Maule", "SAN JAVIER": "Maule",
        # VIII Biobío
        "CONCEPCION": "Biobío", "TALCAHUANO": "Biobío", "CHIGUAYANTE": "Biobío", "SAN PEDRO DE LA PAZ": "Biobío", "LOS ANGELES": "Biobío", "CORONEL": "Biobío", "HUALPEN": "Biobío", "LOTA": "Biobío", "PENCO": "Biobío", "TOME": "Biobío", "ARAUCO": "Biobío",
        # XVI Ñuble
        "CHILLAN": "Ñuble", "CHILLAN VIEJO": "Ñuble", "SAN CARLOS": "Ñuble", "COELECURE": "Ñuble",
        # IX Araucanía
        "TEMUCO": "Araucanía", "PADRE LAS CASAS": "Araucanía", "VILLARRICA": "Araucanía", "ANGOL": "Araucanía", "PUCON": "Araucanía", "VICTORIA": "Araucanía", "LAUTARO": "Araucanía",
        # XIV Los Ríos
        "VALDIVIA": "Los Ríos", "LA UNION": "Los Ríos", "RIO BUENO": "Los Ríos", "PANGUIPULLI": "Los Ríos",
        # X Los Lagos
        "PUERTO MONTT": "Los Lagos", "OSORNO": "Los Lagos", "PUERTO VARAS": "Los Lagos", "CASTRO": "Los Lagos", "ANCUD": "Los Lagos", "FRUTILLAR": "Los Lagos",
        # XI Aysén
        "COYHAIQUE": "Aysén", "AYSEN": "Aysén", "PUERTO AYSEN": "Aysén", "CHILE CHICO": "Aysén",
        # XII Magallanes
        "PUNTA ARENAS": "Magallanes", "PUERTO NATALES": "Magallanes", "PORVENIR": "Magallanes"
    }

# ==========================================
# CARGA DE DATOS
# ==========================================
@st.cache_data
def load_data():
    """Carga los datos originales, formatea fechas y agrega regiones."""
    script_dir = os.path.abspath(os.path.dirname(__file__))
    data_path_folder = os.path.join(script_dir, "data", "rawdata_20240409.csv")
    data_path_root = "rawdata_20240409 (1).csv" 
    
    if os.path.exists(data_path_folder):
        df = pd.read_csv(data_path_folder)
    elif os.path.exists(data_path_root):
        df = pd.read_csv(data_path_root)
    else:
        try:
            df = pd.read_csv("rawdata_20240409.csv")
        except:
            st.error("⚠️ Error Crítico: No se encontró el archivo de datos.")
            return pd.DataFrame()

    df["DateTime"] = pd.to_datetime(df["DateTime"])
    
    if df["Valor"].dtype == object:
        df["Valor"] = df["Valor"].astype(str).str.replace(",", ".").replace("Ausencia", "0")
        df["Valor"] = pd.to_numeric(df["Valor"], errors="coerce")
    
    # --- NUEVO: Agregar columna Región ---
    region_map = get_comuna_region_map()
    # Normalizar a mayúsculas para el mapeo y limpiar espacios
    df["Comuna_Norm"] = df["Comuna"].str.upper().str.strip()
    df["Region"] = df["Comuna_Norm"].map(region_map).fillna("Otra / No Identificada")
    # Limpiamos la columna auxiliar
    df = df.drop(columns=["Comuna_Norm"])
    
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
    "SOLIDOS DISUELTOS TOTALES": {"lim_max": 1500, "ref": 1500} 
}

param_disc = ["OLOR", "COLOR VERDADERO", "SABOR"]
all_params = [p for p in df_raw["Parametro"].unique() if p not in param_disc]
all_regions = sorted(df_raw["Region"].unique()) # Lista de regiones

# ==========================================
# BARRA LATERAL (Navegación y Filtros Globales de Geo)
# ==========================================
st.sidebar.title("Water Quality Chile")

# --- NUEVO: Filtro de Región en Sidebar (Afecta la lista de comunas disponibles) ---
st.sidebar.header("📍 Ubicación Geográfica")
# Por defecto seleccionamos todas o una específica si prefieres
selected_regions = st.sidebar.multiselect(
    "Filtrar por Región",
    all_regions,
    default=all_regions, # Por defecto todas seleccionadas para no ocultar datos
    help="Seleccione una o más regiones para filtrar la lista de comunas."
)

# Filtramos la lista de comunas basada en la región seleccionada
if selected_regions:
    comunas_filtered = sorted(df_raw[df_raw["Region"].isin(selected_regions)]["Comuna"].unique())
else:
    comunas_filtered = sorted(df_raw["Comuna"].unique())

st.sidebar.markdown("---")

modo_visualizacion = st.sidebar.radio(
    "Seleccione análisis:",
    options=[
        "Análisis Temporal", 
        "Radar de Riesgos (Comparativo)",
        "Clustering IA (Perfiles de Agua)"
    ],
    index=0
)

st.sidebar.markdown("---")

# ==========================================
# VISTA 1: ANÁLISIS TEMPORAL
# ==========================================
if modo_visualizacion == "Análisis Temporal":
    
    st.sidebar.subheader("🛠️ Filtros de Análisis")
    
    default_param_name = "SOLIDOS DISUELTOS TOTALES"
    try:
        default_param_index = all_params.index(default_param_name)
    except ValueError:
        default_param_index = 0

    selected_param = st.sidebar.selectbox("Parámetro", all_params, index=default_param_index)
    
    # Lógica de defaults para comunas (debe estar dentro de las filtradas por región)
    default_target = "COPIAPO"
    default_selection = []
    if default_target in comunas_filtered:
        default_selection = [default_target]
    elif comunas_filtered:
        default_selection = [comunas_filtered[0]]

    selected_comunas = st.sidebar.multiselect(
        "Comunas", 
        comunas_filtered, 
        default=default_selection
    )
    
    min_date = df_raw["DateTime"].min()
    max_date = df_raw["DateTime"].max()
    date_range = st.sidebar.date_input(
        "Rango de Fechas",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    st.title("Dashboard Calidad de agua potable Chile")
    
    # Mostrar región en subtítulo si hay 1 comuna seleccionada
    if len(selected_comunas) == 1:
        reg = df_raw[df_raw["Comuna"] == selected_comunas[0]]["Region"].iloc[0]
        st.markdown(f"**Comuna:** {selected_comunas[0]} ({reg}) | **Parámetro:** {selected_param}")
    else:
        st.markdown(f"**Parámetro analizado:** {selected_param}")
    
    if not selected_comunas:
        st.warning("Seleccione al menos una comuna en la barra lateral.")
    else:
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
            # 1. Gráfico
            fig_line = px.line(
                df_filtered, x="DateTime", y="Valor", color="Comuna", 
                markers=True, title=f"Serie Temporal - {selected_param}",
                labels={"Valor": f"Valor ({df_filtered['Unidad'].iloc[0]})"},
                # Agregamos Región al hover
                hover_data={"Region": True}
            )
            
            if selected_param in LIMS:
                limits = LIMS[selected_param]
                if "lim_max" in limits and limits["lim_max"] > 0:
                    fig_line.add_hline(y=limits["lim_max"], line_dash="dash", line_color="red", annotation_text="Límite Max")
                if "lim_min" in limits:
                    fig_line.add_hline(y=limits["lim_min"], line_dash="dash", line_color="orange", annotation_text="Límite Min")

            st.plotly_chart(fig_line, use_container_width=True)
            
            # 2. KPIs
            st.subheader("📊 Estado Actual (Último Registro)")
            cols_kpi = st.columns(len(selected_comunas))
            
            for idx, comuna in enumerate(selected_comunas):
                col_idx = idx % 4
                if idx > 0 and idx % 4 == 0:
                    cols_kpi = st.columns(4) 
                
                df_c = df_filtered[df_filtered["Comuna"] == comuna].sort_values("DateTime")
                if not df_c.empty:
                    last_val = df_c.iloc[-1]["Valor"]
                    last_date_str = df_c.iloc[-1]["DateTime"].strftime("%d/%m/%Y")
                    
                    delta = None
                    if len(df_c) > 1:
                        prev_val = df_c.iloc[-2]["Valor"]
                        delta = round(last_val - prev_val, 1)
                    
                    with cols_kpi[col_idx]:
                        st.metric(
                            label=f"{comuna}",
                            value=f"{last_val:.1f} {df_c.iloc[0]['Unidad']}",
                            delta=delta,
                            help=f"Fecha medición: {last_date_str}"
                        )

            # 3. Histograma y Stats
            col1, col2 = st.columns([1, 1])
            with col1:
                st.subheader("Histograma")
                fig_hist = px.histogram(
                    df_filtered, x="Valor", color="Comuna", barmode="overlay",
                    title="Histograma", opacity=0.7
                )
                st.plotly_chart(fig_hist, use_container_width=True)
                
            with col2:
                st.subheader("Resumen Estadístico")
                stats = df_filtered.groupby("Comuna")["Valor"].describe()[['count', 'mean', 'max', 'min', 'std']]
                last_dates = df_filtered.groupby("Comuna")["DateTime"].max().dt.strftime('%m-%Y')
                # Agregamos región a la tabla resumen
                regions_summary = df_filtered.groupby("Comuna")["Region"].first()
                
                stats["Región"] = regions_summary
                stats["Último Dato"] = last_dates
                
                st.dataframe(
                    stats.style.format("{:.1f}", subset=['mean', 'max', 'min', 'std']), 
                    use_container_width=True
                )
                
            st.subheader("Exportar Datos")
            csv = df_filtered.to_csv(index=False).encode('utf-8')
            st.download_button(
                "📥 Descargar CSV Filtrado", data=csv, file_name=f"datos_{selected_param}.csv", mime="text/csv"
            )

# ==========================================
# VISTA 2: RADAR MULTIPARÁMETRO
# ==========================================
elif modo_visualizacion == "Radar de Riesgos (Comparativo)":
    
    st.sidebar.subheader("⚙️ Configuración Radar")
    st.sidebar.info("Comparación multidimensional normalizada (Valor / Límite).")
    
    # Defaults considerando el filtro de regiones
    default_radar_comunas = ["COPIAPO", "VALPARAISO"]
    # Solo mantener los defaults si están en la lista filtrada actual
    default_radar_comunas = [c for c in default_radar_comunas if c in comunas_filtered]
    
    # Si los defaults no están disponibles (ej: filtraste solo Región Metropolitana), tomar los primeros disponibles
    if len(default_radar_comunas) < 2 and len(comunas_filtered) >= 2:
        default_radar_comunas = comunas_filtered[:2]

    radar_comunas = st.sidebar.multiselect(
        "Comunas a comparar", comunas_filtered, 
        default=default_radar_comunas
    )
    
    params_avail = [p for p in all_params if p in LIMS]
    
    default_radar_params = []
    excluded_params = ["COLIFORMES TOTALES", "TURBIEDAD"]
    for p in params_avail:
        if p not in excluded_params:
            default_radar_params.append(p)
    if "SOLIDOS DISUELTOS TOTALES" in params_avail and "SOLIDOS DISUELTOS TOTALES" not in default_radar_params:
        default_radar_params.append("SOLIDOS DISUELTOS TOTALES")

    radar_params = st.sidebar.multiselect("Parámetros", params_avail, default=default_radar_params)

    st.title("🕸️ Radar multipárametro")
    
    if not radar_comunas or not radar_params:
        st.warning("Seleccione al menos una comuna y parámetros.")
    else:
        fig_radar = go.Figure()
        data_exists = False
        summary_data = []

        for comuna in radar_comunas:
            r_values = []
            theta_values = []
            
            df_comuna = df_raw[df_raw["Comuna"] == comuna]
            if df_comuna.empty: continue
            
            # Obtener región para mostrar en tabla
            region_name = df_comuna["Region"].iloc[0]
                
            last_date = df_comuna["DateTime"].max()
            start_date = last_date - timedelta(days=365)
            df_last_year = df_comuna[df_comuna["DateTime"] >= start_date]
            
            if df_last_year.empty: continue
            
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
                    
                    summary_data.append({
                        "Comuna": comuna,
                        "Región": region_name,
                        "Parámetro": param,
                        "Max Valor (Año)": max_val,
                        "Límite Ref": ref_val,
                        "Indice Riesgo": round(norm_score, 1),
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
            line_ref = [1.0] * (len(radar_params) + 1)
            line_theta = radar_params + [radar_params[0]]
            fig_radar.add_trace(go.Scatterpolar(
                r=line_ref, theta=line_theta, mode='lines', 
                line=dict(color='red', dash='dash', width=2),
                name='Límite (100%)', hoverinfo='skip'
            ))

            max_val_found = 0
            for trace in fig_radar.data:
                if hasattr(trace, 'r') and trace.r:
                    current_max = max(trace.r)
                    if current_max > max_val_found:
                        max_val_found = current_max
            upper_range = max(2.0, max_val_found * 1.1)

            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, upper_range])),
                height=600, showlegend=True
            )
            st.plotly_chart(fig_radar, use_container_width=True)
            
            st.subheader("Detalle de valores críticos")
            df_summary = pd.DataFrame(summary_data)
            if not df_summary.empty:
                df_summary = df_summary.sort_values("Indice Riesgo", ascending=False)
                def highlight_risk(val):
                    color = 'red' if val > 1.0 else 'orange' if val > 0.8 else 'green'
                    return f'color: {color}; font-weight: bold'

                st.dataframe(
                    df_summary.style.applymap(highlight_risk, subset=['Indice Riesgo'])
                    .format({"Max Valor (Año)": "{:.1f}", "Indice Riesgo": "{:.1f}"}),
                    use_container_width=True, hide_index=True
                )
        else:
            st.info("No hay datos suficientes en el último año móvil.")

# ==========================================
# VISTA 3: CLUSTERING IA
# ==========================================
elif modo_visualizacion == "Clustering IA (Perfiles de Agua)":
    st.title("🤖 Clustering de Perfiles de Agua (IA)")
    st.markdown("""
    Agrupación automática de comunas según huella química (K-Means).
    **Nota:** El modelo usará solo las comunas de las regiones seleccionadas en la barra lateral.
    """)

    st.sidebar.subheader("⚙️ Configuración del Modelo")
    n_clusters = st.sidebar.slider("Número de Grupos (Clusters)", 2, 6, 3)
    
    # Filtrar data por tiempo Y por región seleccionada
    last_date = df_raw["DateTime"].max()
    start_date = last_date - timedelta(days=365*2)
    
    # Aplicar filtro de regiones (usamos comunas_filtered que ya viene filtrado arriba)
    df_recent = df_raw[
        (df_raw["DateTime"] >= start_date) & 
        (df_raw["Comuna"].isin(comunas_filtered))
    ].copy()
    
    if df_recent.empty:
        st.error("No hay suficientes datos recientes en las regiones seleccionadas.")
    else:
        df_numeric = df_recent[~df_recent["Parametro"].isin(param_disc)]
        df_pivot = df_numeric.pivot_table(index="Comuna", columns="Parametro", values="Valor", aggfunc="mean")
        
        limit_nan = len(df_pivot) * 0.3
        df_pivot = df_pivot.dropna(thresh=len(df_pivot) - limit_nan, axis=1)
        df_pivot_filled = df_pivot.fillna(df_pivot.median())
        df_pivot_final = df_pivot_filled.dropna()
        
        if df_pivot_final.shape[0] < n_clusters:
            st.warning(f"No hay suficientes comunas ({len(df_pivot_final)}) para formar {n_clusters} clusters. Intenta seleccionar más regiones.")
        else:
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(df_pivot_final)
            
            pca = PCA(n_components=2)
            components = pca.fit_transform(X_scaled)
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
            clusters = kmeans.fit_predict(X_scaled)
            
            df_results = df_pivot_final.copy()
            df_results["Cluster"] = clusters.astype(str)
            df_results["PC1"] = components[:, 0]
            df_results["PC2"] = components[:, 1]
            
            # Unir info de región para el tooltip
            region_map_info = df_raw.groupby("Comuna")["Region"].first()
            df_results["Region"] = df_results.index.map(region_map_info)

            st.subheader("Mapa de Similitud (PCA)")
            fig_pca = px.scatter(
                df_results.reset_index(),
                x="PC1", y="PC2", color="Cluster",
                hover_name="Comuna",
                hover_data={"Region": True},
                title=f"Agrupación de Comunas (K={n_clusters})",
                color_discrete_sequence=px.colors.qualitative.Bold
            )
            st.plotly_chart(fig_pca, use_container_width=True)
            
            # Radar de Clusters
            st.subheader("Interpretación de los Grupos (Perfil Relativo)")
            cluster_means = df_results.drop(["PC1", "PC2", "Region"], axis=1).groupby("Cluster").mean()
            global_mean = df_pivot_final.mean()
            relative_means = (cluster_means / global_mean) - 1
            
            variances = df_pivot_final.var().sort_values(ascending=False)
            top_vars = variances.index[:10].tolist() 
            
            fig_radar_clusters = go.Figure()
            for cluster_id in sorted(df_results["Cluster"].unique()):
                values = relative_means.loc[cluster_id, top_vars]
                fig_radar_clusters.add_trace(go.Scatterpolar(
                    r=values, theta=top_vars, fill='toself', name=f"Cluster {cluster_id}"
                ))
            fig_radar_clusters.add_trace(go.Scatterpolar(
                r=[0]*len(top_vars), theta=top_vars, mode='lines',
                line=dict(color='black', dash='dash'), name='Promedio Nacional'
            ))
            fig_radar_clusters.update_layout(
                polar=dict(radialaxis=dict(visible=True, tickformat=".0%")),
                title="Perfil Químico Relativo", height=600
            )
            st.plotly_chart(fig_radar_clusters, use_container_width=True)
            
            st.subheader("Listado de Comunas por Cluster")
            cols = st.columns(n_clusters)
            for i, col in enumerate(cols):
                cluster_id = str(i)
                if cluster_id in df_results["Cluster"].unique():
                    comunas_in_cluster = df_results[df_results["Cluster"] == cluster_id].index.tolist()
                    with col:
                        st.success(f"**Cluster {cluster_id}** ({len(comunas_in_cluster)} comunas)")
                        for c in comunas_in_cluster[:15]:
                            # Mostrar región junto al nombre
                            reg = df_results.loc[c, "Region"]
                            st.write(f"- {c} <span style='color:gray;font-size:0.8em'>({reg})</span>", unsafe_allow_html=True)
                        if len(comunas_in_cluster) > 15:
                            st.caption(f"... y {len(comunas_in_cluster)-15} más")

# ==========================================
# FOOTER
# ==========================================
st.sidebar.markdown("---")
with st.sidebar.expander("ℹ️ Acerca de"):
    st.markdown("""
    **Monitor de Calidad de Agua**
    v1.7 - Geo Edition
    [GitHub: chile-waterquality](https://github.com/luchoplaza/chile-waterquality)
    """)
