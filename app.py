import streamlit as st
import pandas as pd
import plotly.express as px
import numpy as np 

# --- Configuración inicial y carga de datos para evitar que haya problemas para hacer la gráficas ---
st.set_page_config(
    page_title="Analysis of Vehicle Ads in US",
    layout="wide",
)


@st.cache_data
def load_and_clean_data(file_path):
    """Carga, limpia y prepara el dataset, creando la columna 'base_model'."""
    try:
        df = pd.read_csv(file_path)
    except FileNotFoundError:
        st.error(f"Error: El archivo '{file_path}' no fue encontrado.")
        return pd.DataFrame()
    
    # 1. Limpieza para evitar bloqueos
    cols_critical = ['price', 'model_year', 'odometer', 'condition', 'type', 'days_listed', 'model']
    df.dropna(subset=cols_critical, inplace=True)
    
    # 2. Creación de la columna 'base_model'
    # Extrae la primera palabra del nombre del modelo (ej. 'chevrolet camaro' -> 'chevrolet')
    df['base_model'] = df['model'].apply(lambda x: x.split(' ')[0].lower())
    
    # 3. Conversión segura de tipos de datos
    df['price'] = pd.to_numeric(df['price']).astype(int)
    df['model_year'] = pd.to_numeric(df['model_year']).astype(int)
    df['days_listed'] = pd.to_numeric(df['days_listed']).astype(int)
    df['odometer'] = pd.to_numeric(df['odometer']).astype(int)

    # 4. Filtrar valores atípicos (opcional pero recomendado)
    df = df[(df['price'] > 500) & (df['price'] < 100000)]
    df = df[df['days_listed'] < df['days_listed'].quantile(0.99)] # Quitar anuncios listados por mucho tiempo
    
    return df

# Cargar dataset (llamando a la función de limpieza)
car_data = load_and_clean_data('vehicles_us.csv')

if car_data.empty:
    st.stop()

# --- Definición de Rangos para Sliders
# Gráfica 2
min_price = int(car_data['price'].min())
max_price = int(car_data['price'].max())
# Gráfica 3
min_year = int(car_data['model_year'].min())
max_year = int(car_data['model_year'].max())


# --- Encabezado título
st.header('Analysis of Vehicle Ads in US')
st.write(
    """
    Welcome to the analysis of vehicle ads. 
    Use the bottoms to explore the distribution and relationship of price, condition, 
    model, mileage, model year, vehicle type and average of Ads's days listed.
    """
)

st.write('---')
#---Encabezado gráfica 1---
st.subheader('Histogram: Distribution of price and condition ')
st.markdown(""" 
    *How is the price and condition of vechicle distributed?*
""")

# Controles de usuario
col_check_hist, col_slider_hist = st.columns([1, 3])

with col_check_hist:
    # Casilla de verificación para controlar la visibilidad del gráfico
    show_hist_chart = st.checkbox(
        'Show graph',
        value=True,
        key='hist_chart_show'
    )

with col_slider_hist:
    st.markdown("Adjust number of bins in the histogram:")
    bin_size = st.slider(
    'Number of Bins (Bars)', 
    min_value=20, 
    max_value=80, 
    value=40, 
    step=20, key='bin_slider')

#Creación del histograma
if show_hist_chart:
    st.subheader(f'Distribution of prices segmented by Condition (N={len(car_data)})')
    
    # Creación del Histograma Plotly Express
    fig_hist = px.histogram(
        car_data, 
        x="price", 
        color="condition",
        title=f'Frecuency Vehicle prices by Condition (Total: {len(car_data):,})',
        labels={'price': 'Price (USD)', 'condition': 'Condición del Vehículo'}, # Corregido 'Condition' a 'condition'
        hover_data=['model', 'model_year'],
        template="plotly_white", 
        barmode='overlay', 
        opacity=0.7, 
        nbins=bin_size 
    )
    
    # Ajustes del layout para mejor lectura
    fig_hist.update_layout(
        xaxis_title="Vechicle price (USD)",
        yaxis_title="Ads count",
        legend_title="Condition"
    )

    # Mostrar el gráfico en Streamlit
    st.plotly_chart(fig_hist, use_container_width=True)
else:
    st.info('Activate the checkbox to show the graph.')

st.write('---')

#---Encabezado gráfica 2---
st.subheader('Dispersion graph: Price and Mileage by vehicle type')
st.markdown("""
    *Is there a relationship between a car's mileage and its price?*
""")

# Controles de usuario
col_check, col_slider = st.columns([1, 3])

with col_check:
    # Casilla de verificación para controlar la visibilidad del gráfico
    show_age_scatter = st.checkbox(
        'Show dispersion graph',
        value=True,
        key='age_scatter_show'
    )

with col_slider:
    # 2 Barra Deslizadora (Slider) para seleccionar el rango de precios
    price_range = st.slider(
        'Selecciona un Rango de Precio (USD)',
        min_value=min_price,
        max_value=max_price,
        value=(min_price, max_price), # Valor inicial: el rango completo
        step=500,
        key='price_slider'
    )

# Filtrado de datos según el rango de precios seleccionado por el usuario
min_selected_price = price_range[0]
max_selected_price = price_range[1]

filtered_data_scatter = car_data[ # Cambiado a filtered_data_scatter para evitar conflicto
    (car_data['price'] >= min_selected_price) & 
    (car_data['price'] <= max_selected_price)
]


# Mostrar gráfico
if show_age_scatter:
    
    if filtered_data_scatter.empty:
        st.warning("No data that matches the price range.")
    else:
        st.write(f'Total filtered data: **{len(filtered_data_scatter):,}** Ads')

        # Creación del gráfico
        fig_age_scatter = px.scatter(
            filtered_data_scatter, 
            x="odometer", 
            y="price", 
            color="type", 
            title=f'Price and Mileage by Vehicle Type (Range: ${min_selected_price:,.0f} - ${max_selected_price:,.0f})',
            labels={'odometer': 'Mileage', 'price': 'Price (USD)'},
            hover_data=['model', 'odometer', 'condition'],
            template="plotly_white",
            opacity=0.6,
            height=600
        )
        
        # Ajustes del eje X (se recomienda log_x para kilometraje)
        fig_age_scatter.update_xaxes(type='log') # Se ha cambiado el ajuste de tick a log scale

        st.plotly_chart(fig_age_scatter, use_container_width=True)
else:
    st.info('Activate the checkbox to show the graph.')

st.write('---')

  
#---Encabezado gráfico 3 (Modelo y Días Listados)---
st.subheader('Horizontal bar chart: Ads by listed days, type and model  of vehicles.')
st.markdown("""
    *Which model and type pf vehicle take the longest to sell, based on the average number of days listed?*
""")

# Controles de usuario
col_check_bar, col_slider_bar = st.columns([1, 3])

with col_check_bar:
    show_bar_chart = st.checkbox(
        'Show graph',
        value=True,
        key='bar_chart_show_model'
    )

with col_slider_bar:
    # Barra deslizadora (Slider) para seleccionar el rango de Año del Modelo
    year_range = st.slider(
        'Selection of model year',
        min_value=min_year,
        max_value=max_year,
        value=(max_year - 5, max_year), # Valor inicial: últimos 5 años para mejor enfoque
        step=1,
        key='year_slider_model_bar'
    )

# Filtrado y Agregación de Datos 
if show_bar_chart:
    
    min_selected_year = year_range[0]
    max_selected_year = year_range[1]

    # 1. Filtrar el Dataframe por el rango de años
    filtered_data_bar = car_data[ 
        (car_data['model_year'] >= min_selected_year) & 
        (car_data['model_year'] <= max_selected_year)
    ]

    # 2. AGREGACIÓN CLAVE: Promedio de 'days_listed' por 'base_model' y 'type'
    pivot_data_model = filtered_data_bar.groupby(['base_model', 'type'])['days_listed'].mean().reset_index()
    pivot_data_model.rename(columns={'days_listed': 'promedio_dias_listados'}, inplace=True)
    
    # Filtrar los modelos con pocos anuncios para evitar sesgos
    if len(pivot_data_model) > 50:
        st.warning(f"Showed Top 50 modelos for more clarity (models: {len(pivot_data_model)}).")
        # Ordenar los modelos con más días listados
        pivot_data_model = pivot_data_model.sort_values(
            'promedio_dias_listados', ascending=False
        ).head(50) 


    if pivot_data_model.empty:
        st.warning("No data that matches the selected year range.")
    else:
        st.write(f'Model year range: {min_selected_year} - {max_selected_year}')

        # Creación del Gráfico de Barras Horizontales Plotly Express
        fig_bar_model = px.bar(
            pivot_data_model, 
            x="promedio_dias_listados", 
            y="base_model", # Eje Y: El modelo base
            color="type", # Color: Segmentación por Tipo de Vehículo
            orientation='h', 
            title=f'Average of days listed by model and type of vechicle ({min_selected_year}-{max_selected_year})',
            labels={'promedio_dias_listados': 'Average of days listed', 'base_model': 'Model'},
            template="plotly_white",
            height=800,
            hover_data=['type']
        )
        
        # Ajustes de diseño para ordenar por promedio de días (del más lento al más rápido)
        fig_bar_model.update_yaxes(categoryorder='total ascending') # Ordena de menor a mayor (bottom-up)

        st.plotly_chart(fig_bar_model, use_container_width=True)
else:
    st.info('Activate the checkbox to show the graph.')

st.write('---')
