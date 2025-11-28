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
    st.error("⚠️ Falta instalar scikit-learn. Por favor agrega 'scikit-learn' a tu archivo requirements.txt")

# ==========================================
# CONFIGURACIÓN DE LA PÁGINA
# ==========================================
st.set_page_config(
    page_title="Water Quality Chile - SISS Data",
    layout="wide",
    page_icon="💧",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados
st.markdown("""
<style>
    [data-testid="stMetricValue"] {
        font-size: 24px;
    }
</style>
""", unsafe_allow_html=True)

# ==========================================
# DICCIONARIO DE REGIONES (MAPEO ROBUSTO)
# ==========================================
def get_comuna_region_map():
    """Retorna un diccionario extendido para mapear Comuna/Localidad -> Región."""
    # Este diccionario mapea nombres de comunas y localidades (en mayúsculas) a su Región Administrativa.
    return {
        # XV Arica y Parinacota
        "ARICA": "Arica y Parinacota", "CAMARONES": "Arica y Parinacota", 
        "PUTRE": "Arica y Parinacota", "GENERAL LAGOS": "Arica y Parinacota",

        # I Tarapacá
        "IQUIQUE": "Tarapacá", "ALTO HOSPICIO": "Tarapacá", "POZO ALMONTE": "Tarapacá", 
        "CAMIÑA": "Tarapacá", "COLCHANE": "Tarapacá", "HUARA": "Tarapacá", "PICA": "Tarapacá",

        # II Antofagasta
        "ANTOFAGASTA": "Antofagasta", "MEJILLONES": "Antofagasta", "SIERRA GORDA": "Antofagasta", 
        "TALTAL": "Antofagasta", "CALAMA": "Antofagasta", "OLLAGUE": "Antofagasta", 
        "SAN PEDRO DE ATACAMA": "Antofagasta", "TOCOPILLA": "Antofagasta", "MARIA ELENA": "Antofagasta",

        # III Atacama
        "COPIAPO": "Atacama", "CALDERA": "Atacama", "TIERRA AMARILLA": "Atacama", 
        "CHAÑARAL": "Atacama", "DIEGO DE ALMAGRO": "Atacama", "VALLENAR": "Atacama", 
        "ALTO DEL CARMEN": "Atacama", "FREIRINA": "Atacama", "HUASCO": "Atacama",

        # IV Coquimbo
        "LA SERENA": "Coquimbo", "COQUIMBO": "Coquimbo", "ANDACOLLO": "Coquimbo", 
        "LA HIGUERA": "Coquimbo", "PAIGUANO": "Coquimbo", "VICUÑA": "Coquimbo", 
        "ILLAPEL": "Coquimbo", "CANELA": "Coquimbo", "LOS VILOS": "Coquimbo", 
        "SALAMANCA": "Coquimbo", "OVALLE": "Coquimbo", "COMBARBALA": "Coquimbo", 
        "MONTE PATRIA": "Coquimbo", "PUNITAQUI": "Coquimbo", "RIO HURTADO": "Coquimbo",

        # V Valparaíso
        "VALPARAISO": "Valparaíso", "CASABLANCA": "Valparaíso", "CONCON": "Valparaíso", 
        "JUAN FERNANDEZ": "Valparaíso", "PUCHUNCAVI": "Valparaíso", "QUINTERO": "Valparaíso", 
        "VIÑA DEL MAR": "Valparaíso", "ISLA DE PASCUA": "Valparaíso", "LOS ANDES": "Valparaíso", 
        "CALLE LARGA": "Valparaíso", "RINCONADA": "Valparaíso", "SAN ESTEBAN": "Valparaíso", 
        "LA LIGUA": "Valparaíso", "CABILDO": "Valparaíso", "PAPUDO": "Valparaíso", 
        "PETORCA": "Valparaíso", "ZAPALLAR": "Valparaíso", "QUILLOTA": "Valparaíso", 
        "CALERA": "Valparaíso", "HIJUELAS": "Valparaíso", "LA CRUZ": "Valparaíso", 
        "NOGALES": "Valparaíso", "SAN ANTONIO": "Valparaíso", "ALGARROBO": "Valparaíso", 
        "CARTAGENA": "Valparaíso", "EL QUISCO": "Valparaíso", "EL TABO": "Valparaíso", 
        "SANTO DOMINGO": "Valparaíso", "SAN FELIPE": "Valparaíso", "CATEMU": "Valparaíso", 
        "LLAILLAY": "Valparaíso", "PANQUEHUE": "Valparaíso", "PUTAENDO": "Valparaíso", 
        "SANTA MARIA": "Valparaíso", "QUILPUE": "Valparaíso", "LIMACHE": "Valparaíso", 
        "OLMUE": "Valparaíso", "VILLA ALEMANA": "Valparaíso",

        # RM Metropolitana (Incluyendo localidades específicas del dataset)
        "SANTIAGO": "Metropolitana", "CERRILLOS": "Metropolitana", "CERRO NAVIA": "Metropolitana", 
        "CONCHALI": "Metropolitana", "EL BOSQUE": "Metropolitana", "ESTACION CENTRAL": "Metropolitana", 
        "HUECHURABA": "Metropolitana", "INDEPENDENCIA": "Metropolitana", "LA CISTERNA": "Metropolitana", 
        "LA FLORIDA": "Metropolitana", "LA GRANJA": "Metropolitana", "LA PINTANA": "Metropolitana", 
        "LA REINA": "Metropolitana", "LAS CONDES": "Metropolitana", "LO BARNECHEA": "Metropolitana", 
        "LO ESPEJO": "Metropolitana", "LO PRADO": "Metropolitana", "MACUL": "Metropolitana", 
        "MAIPU": "Metropolitana", "NUNOA": "Metropolitana", "ÑUÑOA": "Metropolitana", 
        "PEDRO AGUIRRE CERDA": "Metropolitana", "PENALOLEN": "Metropolitana", "PEÑALOLEN": "Metropolitana", 
        "PROVIDENCIA": "Metropolitana", "PUDAHUEL": "Metropolitana", "QUILICURA": "Metropolitana", 
        "QUINTA NORMAL": "Metropolitana", "RECOLETA": "Metropolitana", "RENCA": "Metropolitana", 
        "SAN JOAQUIN": "Metropolitana", "SAN MIGUEL": "Metropolitana", "SAN RAMON": "Metropolitana", 
        "VITACURA": "Metropolitana", "PUENTE ALTO": "Metropolitana", "PIRQUE": "Metropolitana", 
        "SAN JOSE DE MAIPO": "Metropolitana", "COLINA": "Metropolitana", "LAMPA": "Metropolitana", 
        "TIL TIL": "Metropolitana", "SAN BERNARDO": "Metropolitana", "BUIN": "Metropolitana", 
        "CALERA DE TANGO": "Metropolitana", "PAINE": "Metropolitana", "MELIPILLA": "Metropolitana", 
        "ALHUE": "Metropolitana", "CURACAVI": "Metropolitana", "MARIA PINTO": "Metropolitana", 
        "SAN PEDRO": "Metropolitana", "TALAGANTE": "Metropolitana", "EL MONTE": "Metropolitana", 
        "ISLA DE MAIPO": "Metropolitana", "PADRE HURTADO": "Metropolitana", "PEÑAFLOR": "Metropolitana", 
        "SANTIAGO CENTRO": "Metropolitana",
        # Localidades específicas SISS
        "HACIENDA BATUCO": "Metropolitana", "REINA NORTE": "Metropolitana", 
        "EL COLORADO": "Metropolitana", "LA PARVA": "Metropolitana", "VALLE NEVADO": "Metropolitana",
        "SANTA MARIA DE MANQUEHUE": "Metropolitana", "LO CURRO": "Metropolitana",

        # VI O'Higgins
        "RANCAGUA": "O'Higgins", "CODEGUA": "O'Higgins", "COINCO": "O'Higgins", 
        "COLTAUCO": "O'Higgins", "DOÑIHUE": "O'Higgins", "GRANEROS": "O'Higgins", 
        "LAS CABRAS": "O'Higgins", "MACHALI": "O'Higgins", "MALLOA": "O'Higgins", 
        "MOSTAZAL": "O'Higgins", "OLIVAR": "O'Higgins", "PEUMO": "O'Higgins", 
        "PICHIDEGUA": "O'Higgins", "QUINTA DE TILCOCO": "O'Higgins", "RENGO": "O'Higgins", 
        "REQUINOA": "O'Higgins", "SAN VICENTE": "O'Higgins", "PICHILEMU": "O'Higgins", 
        "LA ESTRELLA": "O'Higgins", "LITUECHE": "O'Higgins", "MARCHIGUE": "O'Higgins", 
        "NAVIDAD": "O'Higgins", "PAREDONES": "O'Higgins", "SAN FERNANDO": "O'Higgins", 
        "CHEPICA": "O'Higgins", "CHIMBARONGO": "O'Higgins", "LOLOL": "O'Higgins", 
        "NANCAGUA": "O'Higgins", "PALMILLA": "O'Higgins", "PERALILLO": "O'Higgins", 
        "PLACILLA": "O'Higgins", "PUMANQUE": "O'Higgins", "SANTA CRUZ": "O'Higgins",

        # VII Maule
        "TALCA": "Maule", "CONSTITUCION": "Maule", "CUREPTO": "Maule", 
        "EMPEDRADO": "Maule", "MAULE": "Maule", "PELARCO": "Maule", 
        "PENCAHUE": "Maule", "RIO CLARO": "Maule", "SAN CLEMENTE": "Maule", 
        "SAN RAFAEL": "Maule", "CAUQUENES": "Maule", "CHANCO": "Maule", 
        "PELLUHUE": "Maule", "CURICO": "Maule", "HUALAÑE": "Maule", 
        "LICANTEN": "Maule", "MOLINA": "Maule", "RAUCO": "Maule", 
        "ROMERAL": "Maule", "SAGRADA FAMILIA": "Maule", "TENO": "Maule", 
        "VICHUQUEN": "Maule", "LINARES": "Maule", "COLBUN": "Maule", 
        "LONGAVI": "Maule", "PARRAL": "Maule", "RETIRO": "Maule", 
        "SAN JAVIER": "Maule", "VILLA ALEGRE": "Maule", "YERBAS BUENAS": "Maule",

        # XVI Ñuble
        "COBQUECURA": "Ñuble", "COELECURE": "Ñuble", "NINHUE": "Ñuble", 
        "PORTEZUELO": "Ñuble", "QUIRIHUE": "Ñuble", "RANQUIL": "Ñuble", 
        "TREHUACO": "Ñuble", "CHILLAN": "Ñuble", "BULNES": "Ñuble", 
        "CHILLAN VIEJO": "Ñuble", "EL CARMEN": "Ñuble", "PEMUCO": "Ñuble", 
        "PINTO": "Ñuble", "QUILLON": "Ñuble", "SAN IGNACIO": "Ñuble", 
        "YUNGAY": "Ñuble", "SAN CARLOS": "Ñuble", "COIHUECO": "Ñuble", 
        "ÑIQUEN": "Ñuble", "SAN FABIAN": "Ñuble", "SAN NICOLAS": "Ñuble",

        # VIII Biobío
        "CONCEPCION": "Biobío", "CORONEL": "Biobío", "CHIGUAYANTE": "Biobío", 
        "FLORIDA": "Biobío", "HUALQUI": "Biobío", "LOTA": "Biobío", 
        "PENCO": "Biobío", "SAN PEDRO DE LA PAZ": "Biobío", "SANTA JUANA": "Biobío", 
        "TALCAHUANO": "Biobío", "TOME": "Biobío", "HUALPEN": "Biobío", 
        "LEBU": "Biobío", "ARAUCO": "Biobío", "CAÑETE": "Biobío", 
        "CONTULMO": "Biobío", "CURANILAHUE": "Biobío", "LOS ALAMOS": "Biobío", 
        "TIRUA": "Biobío", "LOS ANGELES": "Biobío", "ANTUCO": "Biobío", 
        "CABRERO": "Biobío", "LAJA": "Biobío", "MULCHEN": "Biobío", 
        "NACIMIENTO": "Biobío", "NEGRETE": "Biobío", "QUILLACO": "Biobío", 
        "QUILLECO": "Biobío", "SAN ROSENDO": "Biobío", "SANTA BARBARA": "Biobío", 
        "TUCAPEL": "Biobío", "YUMBEL": "Biobío", "ALTO BIOBIO": "Biobío",

        # IX Araucanía
        "TEMUCO": "Araucanía", "CARAHUE": "Araucanía", "CUNCO": "Araucanía", 
        "CURARREHUE": "Araucanía", "FREIRE": "Araucanía", "GALVARINO": "Araucanía", 
        "GORBEA": "Araucanía", "LAUTARO": "Araucanía", "LONCOCHE": "Araucanía", 
        "MELIPEUCO": "Araucanía", "NUEVA IMPERIAL": "Araucanía", "PADRE LAS CASAS": "Araucanía", 
        "PERQUENCO": "Araucanía", "PITRUFQUEN": "Araucanía", "PUCON": "Araucanía", 
        "SAAVEDRA": "Araucanía", "TEODORO SCHMIDT": "Araucanía", "TOLTEN": "Araucanía", 
        "VILCUN": "Araucanía", "VILLARRICA": "Araucanía", "CHOLCHOL": "Araucanía", 
        "ANGOL": "Araucanía", "COLLIPULLI": "Araucanía", "CURACAUTIN": "Araucanía", 
        "ERCILLA": "Araucanía", "LONQUIMAY": "Araucanía", "LOS SAUCES": "Araucanía", 
        "LUMACO": "Araucanía", "PUREN": "Araucanía", "RENAICO": "Araucanía", 
        "TRAIGUEN": "Araucanía", "VICTORIA": "Araucanía",

        # XIV Los Ríos
        "VALDIVIA": "Los Ríos", "CORRAL": "Los Ríos", "LANCO": "Los Ríos", 
        "LOS LAGOS": "Los Ríos", "MAFIL": "Los Ríos", "MARIQUINA": "Los Ríos", 
        "PAILLACO": "Los Ríos", "PANGUIPULLI": "Los Ríos", "LA UNION": "Los Ríos", 
        "FUTRONO": "Los Ríos", "LAGO RANCO": "Los Ríos", "RIO BUENO": "Los Ríos",

        # X Los Lagos
        "PUERTO MONTT": "Los Lagos", "CALBUCO": "Los Lagos", "COCHAMO": "Los Lagos", 
        "FRESIA": "Los Lagos", "FRUTILLAR": "Los Lagos", "LOS MUERMOS": "Los Lagos", 
        "LLANQUIHUE": "Los Lagos", "MAULLIN": "Los Lagos", "PUERTO VARAS": "Los Lagos", 
        "CASTRO": "Los Lagos", "ANCUD": "Los Lagos", "CHONCHI": "Los Lagos", 
        "CURACO DE VELEZ": "Los Lagos", "DALCAHUE": "Los Lagos", "PUQUELDON": "Los Lagos", 
        "QUEILEN": "Los Lagos", "QUELLON": "Los Lagos", "QUEMCHI": "Los Lagos", 
        "QUINCHAO": "Los Lagos", "OSORNO": "Los Lagos", "PUERTO OCTAY": "Los Lagos", 
        "PURRANQUE": "Los Lagos", "PUYEHUE": "Los Lagos", "RIO NEGRO": "Los Lagos", 
        "SAN JUAN DE LA COSTA": "Los Lagos", "SAN PABLO": "Los Lagos", 
        "CHAITEN": "Los Lagos", "FUTALEUFU": "Los Lagos", "HUALAIHUE": "Los Lagos", 
        "PALENA": "Los Lagos",

        # XI Aysén
        "COYHAIQUE": "Aysén", "LAGO VERDE": "Aysén", "AYSEN": "Aysén", 
        "PUERTO AYSEN": "Aysén", "CISNES": "Aysén", "GUAITECAS": "Aysén", 
        "COCHRANE": "Aysén", "O'HIGGINS": "Aysén", "TORTEL": "Aysén", 
        "CHILE CHICO": "Aysén", "RIO IBAÑEZ": "Aysén",

        # XII Magallanes
        "PUNTA ARENAS": "Magallanes", "LAGUNA BLANCA": "Magallanes", "RIO VERDE": "Magallanes", 
        "SAN GREGORIO": "Magallanes", "CABO DE HORNOS": "Magallanes", "ANTARTICA": "Magallanes", 
        "PORVENIR": "Magallanes", "PRIMAVERA": "Magallanes", "TIMAUKEL": "Magallanes", 
        "NATALES": "Magallanes", "PUERTO NATALES": "Magallanes", "TORRES DEL PAINE": "Magallanes"
    }

# ==========================================
# CARGA DE DATOS (ROBUSTA)
# ==========================================
@st.cache_data
def load_data():
    """Carga los datos originales, formatea fechas y agrega regiones."""
    script_dir = os.path.abspath(os.path.dirname(__file__))
    
    possible_paths = [
        os.path.join(script_dir, "data", "rawdata_20240409.csv"),
        os.path.join(script_dir, "..", "rawdata_20240409.csv"), 
        "rawdata_20240409.csv",
        "rawdata_20240409 (1).csv"
    ]
    
    df = None
    
    for path in possible_paths:
        if os.path.exists(path):
            try:
                if os.stat(path).st_size == 0:
                    continue 
            except OSError:
                continue

            try:
                temp_df = pd.read_csv(path)
                if not temp_df.empty:
                    df = temp_df
                    break 
            except Exception:
                continue
    
    if df is None:
        st.error("⚠️ Error Crítico: No se encontró ningún archivo de datos válido (CSV no vacío).")
        return pd.DataFrame(columns=["DateTime", "Comuna", "Parametro", "Valor", "Unidad", "Empresa", "Region"])

    df["DateTime"] = pd.to_datetime(df["DateTime"])
    
    if df["Valor"].dtype == object:
        df["Valor"] = df["Valor"].astype(str).str.replace(",", ".").replace("Ausencia", "0")
        df["Valor"] = pd.to_numeric(df["Valor"], errors="coerce")
    
    region_map = get_comuna_region_map()
    df["Comuna_Norm"] = df["Comuna"].str.upper().str.strip()
    df["Region"] = df["Comuna_Norm"].map(region_map).fillna("Otra / No Identificada")
    df = df.drop(columns=["Comuna_Norm"])
    
    return df.sort_values("DateTime")

df_raw = load_data()

if df_raw.empty:
    st.warning("Esperando datos... Por favor verifica que el archivo CSV esté cargado correctamente.")
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
all_regions = sorted(df_raw["Region"].unique())

# ==========================================
# BARRA LATERAL (Navegación)
# ==========================================
st.sidebar.title("Water Quality Chile")

# Selector de Modo (Ahora al principio)
modo_visualizacion = st.sidebar.radio(
    "Seleccione análisis:",
    options=[
        "Análisis Temporal", 
        "Radar multipárametro", 
        "Perfiles de agua (KM+PCA)"
    ],
    index=0
)

st.sidebar.markdown("---")

# Función auxiliar para renderizar filtro de región de manera consistente
def render_region_filter():
    st.sidebar.markdown("**Ubicación Geográfica**")
    selected = st.sidebar.multiselect(
        "Filtrar por Región",
        all_regions,
        default=[], # Por defecto vacío (no filtra nada)
        help="Dejar vacío para ver todas las regiones.",
        key="region_filter_global" # Misma key para mantener estado entre pestañas si se desea
    )
    # Lógica: Si está vacío, devuelve TODAS las comunas. Si tiene selección, filtra.
    if selected:
        return sorted(df_raw[df_raw["Region"].isin(selected)]["Comuna"].unique())
    else:
        return sorted(df_raw["Comuna"].unique())

# ==========================================
# VISTA 1: ANÁLISIS TEMPORAL
# ==========================================
if modo_visualizacion == "Análisis Temporal":
    
    st.sidebar.subheader("🛠️ Filtros de Análisis")
    
    # 1. Filtro Región (Ubicado bajo Filtros de Análisis)
    comunas_filtered = render_region_filter()
    
    # 2. Filtro Parámetro
    default_param_name = "SOLIDOS DISUELTOS TOTALES"
    try:
        default_param_index = all_params.index(default_param_name)
    except ValueError:
        default_param_index = 0
    selected_param = st.sidebar.selectbox("Parámetro", all_params, index=default_param_index)
    
    # 3. Filtro Comunas (Depende de región)
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
    
    # 4. Filtro Fechas
    min_date = df_raw["DateTime"].min()
    max_date = df_raw["DateTime"].max()
    date_range = st.sidebar.date_input(
        "Rango de Fechas",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date
    )

    st.title("Dashboard Calidad de agua potable Chile")
    
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
            # Gráfico Línea
            fig_line = px.line(
                df_filtered, x="DateTime", y="Valor", color="Comuna", 
                markers=True, title=f"Serie Temporal - {selected_param}",
                labels={"Valor": f"Valor ({df_filtered['Unidad'].iloc[0]})"},
                hover_data={"Region": True}
            )
            
            if selected_param in LIMS:
                limits = LIMS[selected_param]
                if "lim_max" in limits and limits["lim_max"] > 0:
                    fig_line.add_hline(y=limits["lim_max"], line_dash="dash", line_color="red", annotation_text="Límite Max")
                if "lim_min" in limits:
                    fig_line.add_hline(y=limits["lim_min"], line_dash="dash", line_color="orange", annotation_text="Límite Min")

            st.plotly_chart(fig_line, use_container_width=True)
            
            # KPIs
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

            # Histograma y Tabla
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
# VISTA 2: RADAR MULTIPARÁMETRO (RENOMBRADO)
# ==========================================
elif modo_visualizacion == "Radar multipárametro":
    
    st.sidebar.subheader("⚙️ Configuración Radar")
    
    # 1. Filtro Región
    comunas_filtered = render_region_filter()
    
    st.sidebar.info("Comparación multidimensional normalizada (Valor / Límite).")
    
    default_radar_comunas = ["COPIAPO", "VALPARAISO"]
    default_radar_comunas = [c for c in default_radar_comunas if c in comunas_filtered]
    
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
# VISTA 3: PERFILES DE AGUA (RENOMBRADO)
# ==========================================
elif modo_visualizacion == "Perfiles de agua (KM+PCA)":
    st.title("🤖 Perfiles de agua (KM+PCA)")
    st.markdown("""
    Agrupación automática de comunas según huella química (K-Means).
    **Nota:** El modelo usará las comunas de las regiones seleccionadas.
    """)

    st.sidebar.subheader("⚙️ Configuración del Modelo")
    
    # 1. Filtro Región
    comunas_filtered = render_region_filter()
    
    n_clusters = st.sidebar.slider("Número de Grupos (Clusters)", 2, 6, 3)
    
    # Filtrar data
    last_date = df_raw["DateTime"].max()
    start_date = last_date - timedelta(days=365*2)
    
    # Aplicar filtro de regiones (usando la lista comunas_filtered que ya fue procesada)
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
            st.info("💡 **Colores = Región** | **Formas = Cluster**. Observa si las comunas de una misma región se agrupan juntas.")

            fig_pca = px.scatter(
                df_results.reset_index(),
                x="PC1", y="PC2", 
                color="Region",  # Colorear por Región Geográfica
                symbol="Cluster", # Forma según el Cluster IA
                hover_name="Comuna",
                hover_data={"Region": True, "Cluster": True},
                title=f"Relación Geografía vs Química del Agua (K={n_clusters})",
                color_discrete_sequence=px.colors.qualitative.Bold
            )
            
            fig_pca.update_traces(marker=dict(size=12, line=dict(width=1, color='DarkSlateGrey')))
            st.plotly_chart(fig_pca, use_container_width=True)
            
            # Radar de Clusters (Modificado: Respecto a Límite Real)
            st.subheader("Interpretación de los Grupos (Riesgo Relativo)")
            st.markdown("""
            **Valores normalizados (Promedio del Cluster / Límite Normativo).**
            - **1.0 (Línea punteada):** El promedio del grupo alcanza el límite permitido.
            - **< 0.5:** Valores seguros.
            """)
            
            # Calculamos promedios por cluster
            cluster_means = df_results.drop(["PC1", "PC2", "Region"], axis=1).groupby("Cluster").mean()
            
            # Filtramos solo las variables que tienen Límite definido en LIMS
            # y que además tengan cierta varianza para que el gráfico sea interesante
            vars_with_limits = [col for col in cluster_means.columns if col in LIMS]
            
            if not vars_with_limits:
                st.warning("No se encontraron parámetros con límites definidos para graficar el radar.")
            else:
                # Seleccionamos Top 10 con mayor varianza dentro de las que tienen límites
                variances = df_pivot_final[vars_with_limits].var().sort_values(ascending=False)
                top_vars = variances.index[:10].tolist() 
                
                fig_radar_clusters = go.Figure()
                
                for cluster_id in sorted(df_results["Cluster"].unique()):
                    # Calcular Score = Valor Promedio / Limite Referencia
                    scores = []
                    for var in top_vars:
                        val = cluster_means.loc[cluster_id, var]
                        ref = LIMS[var]["ref"]
                        scores.append(val / ref if ref > 0 else 0)
                    
                    fig_radar_clusters.add_trace(go.Scatterpolar(
                        r=scores,
                        theta=top_vars,
                        fill='toself',
                        name=f"Cluster {cluster_id}"
                    ))
                
                # Línea de referencia (Límite = 1.0)
                fig_radar_clusters.add_trace(go.Scatterpolar(
                    r=[1.0]*len(top_vars),
                    theta=top_vars,
                    mode='lines',
                    line=dict(color='red', dash='dash'),
                    name='Límite Normativo (100%)'
                ))

                fig_radar_clusters.update_layout(
                    polar=dict(
                        radialaxis=dict(visible=True, tickformat=".1f") # Formato decimal
                    ),
                    title="Perfil de Riesgo Promedio por Cluster (Normalizado)",
                    height=600
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
                            reg = df_results.loc[c, "Region"]
                            st.write(f"- {c} <span style='color:gray;font-size:0.8em'>({reg})</span>", unsafe_allow_html=True)
                        if len(comunas_in_cluster) > 15:
                            st.caption(f"... y {len(comunas_in_cluster)-15} más")

            # --- SECCION NUEVA: EXPORTAR DATOS DEL MODELO ---
            st.markdown("---")
            st.subheader("📥 Exportar Resultados del Modelo")
            st.write("Descarga la clasificación de comunas y sus perfiles químicos promedio.")
            
            csv_clusters = df_results.to_csv(index=True).encode('utf-8')
            st.download_button(
                label="Descargar Reporte de Clusters (CSV)",
                data=csv_clusters,
                file_name="clusters_perfiles_agua.csv",
                mime="text/csv"
            )

# ==========================================
# FOOTER
# ==========================================
st.sidebar.markdown("---")
with st.sidebar.expander("ℹ️ Acerca de"):
    st.markdown("""
    **Monitor de Calidad de Agua**
    v1.8 - ML & UX Update
    [GitHub: chile-waterquality](https://github.com/luchoplaza/chile-waterquality)
    """)
