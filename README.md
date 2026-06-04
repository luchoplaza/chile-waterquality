# Chile Water Quality Dashboard

Dashboard interactivo en **Streamlit** para visualizar y analizar parámetros de calidad
del agua potable en Chile, a partir de datos de la SISS (Superintendencia de Servicios
Sanitarios). Incluye series de tiempo, comparación entre comunas/empresas, gráficos de
radar y análisis de *clustering* (PCA + KMeans).

> La versión anterior en Dash (para Render) se separó a su propio repositorio.

## Datos
Fuente: SISS — https://www.siss.gob.cl/586/w3-propertyvalue-6405.html

La app lee `src/data/rawdata.csv`. Este archivo se actualiza automáticamente mediante un
GitHub Action alojado en el repositorio de datos (mensual), que copia el CSV procesado a
`src/data/rawdata.csv` en la rama `main`.

## Uso local

### Con [uv](https://docs.astral.sh/uv/) (recomendado)
```bash
uv venv
uv pip install -r requirements.txt
uv run streamlit run src/app.py
```
O en un solo paso, sin crear el entorno a mano:
```bash
uv run --with-requirements requirements.txt streamlit run src/app.py
```

### Con pip
```bash
pip install -r requirements.txt
streamlit run src/app.py
```

La app queda disponible en http://localhost:8501/

## Despliegue
Desplegado en [Streamlit Community Cloud](https://streamlit.io/cloud):
- **Rama:** `main`
- **Main file path:** `src/app.py`

## Funcionalidades
- **Series de tiempo** por parámetro, comuna y empresa.
- **Filtros** por región, comuna, empresa, parámetro y rango de fechas.
- **Gráfico de radar** con rangos dinámicos.
- **Clustering** (StandardScaler → PCA → KMeans) para perfilar comunas.
- **Exportación** de datos filtrados y perfiles de clusters a CSV.

## Licencia
MIT.

## Contacto
lplazaalvarez@gmail.com
