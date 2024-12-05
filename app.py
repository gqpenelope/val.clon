import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import scipy.optimize as sco
from streamlit_option_menu import option_menu
from streamlit_extras.metric_cards import style_metric_cards

# Configuración de la página
st.set_page_config(page_title="Análisis de Portafolios", layout="wide")
st.markdown(
    """
    <style>
    body {
        background-color: #1D1E2C;
        color: white;
    }
    h1, h2, h3, h4, h5, h6, p, span, div {
        color: white !important;
    }
    .stApp {
        background-color: #1D1E2C;
    }
    </style>
    """,
    unsafe_allow_html=True
)
st.title("Análisis y Optimización de Portafolios")





# Definición de ETFs y ventanas de tiempo
etfs = ['LQD', 'EMB', 'ACWI', 'SPY', 'WMT']
descripciones_etfs = {
    "LQD": {
        "nombre": "iShares iBoxx $ Investment Grade Corporate Bond ETF",
        "exposicion": "Bonos corporativos denominados en dólares de EE.UU. y con grado de inversión.",
        "indice": "iBoxx $ Liquid Investment Grade Index",
        "moneda": "USD",
        "principales": ["JPMorgan Chase & Co", "Bank of America Corp", "Morgan Stanley"],
        "paises": "Estados Unidos y empresas multinacionales",
        "estilo": "Value",
        "costos": "Comisión de administración: 0.14%"
    },
    "EMB": {
        "nombre": "iShares J.P. Morgan USD Emerging Markets Bond ETF",
        "exposicion": "Bonos de gobierno denominados en USD emitidos por países de mercados emergentes.",
        "indice": "JPMorgan EMBI Global Core Index",
        "moneda": "USD",
        "principales": ["Turkey (Republic of)", "Saudi Arabia (Kingdom of)", "Brazil Federative Republic of"],
        "paises": "América Latina, Medio Oriente, África y Asia",
        "estilo": "Value",
        "costos": "Comisión de administración: 0.39%"
    },
    "ACWI": {
        "nombre": "iShares MSCI ACWI ETF",
        "exposicion": "Empresas internacionales de mercados desarrollados y emergentes de alta y mediana capitalización.",
        "indice": "MSCI ACWI Index",
        "moneda": "USD",
        "principales": ["Apple Inc", "NVIDIA Corp", "Microsoft Corp"],
        "paises": "Estados Unidos y mercados desarrollados/emergentes",
        "estilo": "Growth",
        "costos": "Comisión de administración: 0.32%"
    },
    "SPY": {
        "nombre": "iShares Core S&P 500 ETF",
        "exposicion": "Empresas de alta capitalización en Estados Unidos.",
        "indice": "S&P 500 Index (USD)",
        "moneda": "USD",
        "principales": ["Apple Inc", "NVIDIA Corp", "Microsoft Corp"],
        "paises": "Estados Unidos",
        "estilo": "Mix (Growth/Value)",
        "costos": "Comisión de administración: 0.03%"
    },
    "WMT": {
        "nombre": "Walmart Inc",
        "exposicion": "Retailer global enfocado en mercados de Estados Unidos.",
        "indice": "N/A",
        "moneda": "USD",
        "principales": ["N/A"],
        "paises": "Estados Unidos y mercados internacionales",
        "estilo": "Mix (Growth/Value)",
        "costos": "N/A"
    }
}
ventanas = {
    "2010-2023": ("2010-01-01", "2023-12-31"),
    "2010-2020": ("2010-01-01", "2020-12-31"),
    "2021-2023": ("2021-01-01", "2023-12-31")
}

# Funcion para crear un menu 
with st.sidebar:
    selected = option_menu(
        menu_title="Ventana",
        options=list(ventanas.keys()),
        icons=["calendar", "calendar-range", "calendar3"],
        menu_icon="gear",
        default_index=0,
        styles={
            "container": {"padding": "5px", "background-color": "#1D1E2C"},
            "icon": {"color": "white", "font-size": "25px"},  
            "nav-link": {
                "font-size": "16px",
                "text-align": "left",
                "margin": "0px",
                "color": "white",
                "background-color": "#1D1E2C",
            },
            "nav-link-selected": {"background-color": "#C4F5FC", "color": "white"},
        },
    )
    
start_date, end_date = ventanas[selected]


# Tabs de la aplicación
st.markdown(
    """
    <style>
    div[data-baseweb="tab-highlight"] {
        background-color: white !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)
tab1, tab2, tab3, tab4 = st.tabs([
    "Análisis de Activos Individuales",
    "Portafolios Óptimos",
    "Comparación de Portafolios",
    "Black-Litterman"
])


# Función para descargar datos de Yahoo Finance
@st.cache_data
def obtener_datos(etfs, start_date, end_date):
    try:
        data = yf.download(etfs, start=start_date, end=end_date)['Close']
        if data.empty:
            st.error(f"No se encontraron datos para los ETFs: {etfs} entre {start_date} y {end_date}.")
        return data
    except Exception as e:
        st.error(f"Error al descargar datos: {e}")
        return pd.DataFrame()

# Descargar datos
datos = obtener_datos(etfs, start_date, end_date)

# Validar datos
if datos.empty:
    st.error("No se encontraron datos para los parámetros seleccionados.")
else:
    # Calcular rendimientos
    rendimientos = datos.pct_change().dropna()

    # Función para calcular métricas
    def calcular_metricas(rendimientos):
        media = rendimientos.mean() * 252  # Rendimiento anualizado
        volatilidad = rendimientos.std() * np.sqrt(252)  # Volatilidad anualizada
        sharpe = media / volatilidad  # Ratio Sharpe
        sesgo = rendimientos.skew()  # Sesgo de los rendimientos
        curtosis = rendimientos.kurt()  # Curtosis de los rendimientos
        return {
            "Media": media,
            "Volatilidad": volatilidad,
            "Sharpe": sharpe,
            "Sesgo": sesgo,
            "Curtosis": curtosis
        }

    # Calcular métricas para cada ETF
    metricas = {etf: calcular_metricas(rendimientos[etf]) for etf in etfs}
    metricas_df = pd.DataFrame(metricas).T  # Convertir a DataFrame para análisis tabular

    # Tab 1: Análisis de Activos Individuales
    with tab1:
        st.markdown(
        """
        <div style="
            background-color: #C4F5FC;
            padding: 8px;
            border-radius: 20px;
            color: black;
            text-align: center;
        ">
            <h1 style="margin: 0; color: #black; font-size: 25px; ">Análisis de Activos Individuales</h1>
        </div>
        """,
        unsafe_allow_html=True,
        )
        
        # Selección del ETF para análisis
        etfs = ['LQD', 'EMB', 'ACWI', 'SPY', 'WMT']
        etf_seleccionado = st.selectbox("Selecciona un ETF para análisis:", options=etfs)
        st.markdown(
            """
            <style>
            .card {
                background-color: #1F2C56;
                border-radius: 10px;
                padding: 20px;
                margin: 10px;#497076
                color: white;
                text-align: center;
                box-shadow: 2px 2px 10px rgba(0, 0, 0, 0.3);
            }
            .card-title {
                font-size: 20px;
                font-weight: bold;
                margin-bottom: 10px;
            }
            .card-value {
                font-size: 28px;
                font-weight: bold;
            }
            </style>
            """,
            unsafe_allow_html=True
        )
    
        # Dividir en dos columnas
        col1, col2 = st.columns(2)
    
        # Características del ETF
        with col1:
            st.subheader("Características del ETF")
            data = descripciones_etfs[etf_seleccionado]

            # Crear la tabla con las características
            tabla_caracteristicas = pd.DataFrame({
                "Características": ["Nombre", "Exposición", "Índice", "Moneda", "Principales Contribuyentes", "Países", "Estilo", "Costos"],
                "Detalles": [
                    data["nombre"],
                    data["exposicion"],
                    data["indice"],
                    data["moneda"],
                    ", ".join(data["principales"]),
                    data["paises"],
                    data["estilo"],
                    data["costos"]
                ]
            })  

            # Estilizar la tabla con HTML para letras blancas y eliminar índices
            st.markdown(
                """
                <style>
                table {
                    color: white;
                    background-color: transparent;
                    border-collapse: collapse;
                    width: 100%;
                th {
                    background-color: #2CA58D; /* Fondo amarillo para la fila del encabezado */
                    color: black; /* Texto en negro para contraste */
                    font-weight: bold;
                    text-align: center;
                    vertical-align: middle;
                }
                td {
                    border: 1px solid white;
                    padding: 8px;
                    text-align: center;
                }
                </style>
                """,
                unsafe_allow_html=True
            )

            # Convertir el DataFrame a HTML 
            tabla_html = tabla_caracteristicas.to_html(index=False, escape=False)

            # Renderizar la tabla estilizada
            st.markdown(tabla_html, unsafe_allow_html=True)
        # CSS para aumentar el tamaño de la fuente en las métricas
        st.markdown(
            """
            <style>
            .stMetric {
                font-size: 24px !important; 
            }
            .stMetric label {
                font-size: 28px !important; 
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        # Métricas calculadas
        with col2:
            st.subheader("Métricas Calculadas")
        
            # Mostrar las métricas en recuadros
            st.metric("Media", value=f"{metricas[etf_seleccionado]['Media']:.2f}")
            st.metric("Volatilidad", value=f"{metricas[etf_seleccionado]['Volatilidad']:.2f}")
            st.metric("Sharpe", value=f"{metricas[etf_seleccionado]['Sharpe']:.2f}") 
            st.metric("Sesgo", value=f"{metricas[etf_seleccionado]['Sesgo']:.2f}") 
            st.metric("Curtosis", value=f"{metricas[etf_seleccionado]['Curtosis']:.2f}")
                
            style_metric_cards(background_color="#84BC9C", border_left_color="#F46197")

        # Gráfica de precios normalizados con fondo oscuro y color de línea personalizado
        st.subheader("Serie de Tiempo de Precios Normalizados")
        precios_normalizados = datos[etf_seleccionado] / datos[etf_seleccionado].iloc[0] * 100
        fig = go.Figure(go.Scatter(
            x=precios_normalizados.index,
            y=precios_normalizados,
            mode='lines',
            name=etf_seleccionado,
            line=dict(color='#F46197')  
        ))

        fig.update_layout(
            title=dict(
                text="Precio Normalizado",
                font=dict(color='white'),
            ),
            xaxis=dict(
                title="Fecha",
                titlefont=dict(color='white'),  # Etiqueta del eje x en blanco
                tickfont=dict(color='white')  # Etiquetas de los ticks en blanco
            ),
            yaxis=dict(
                title="Precio Normalizado",
                titlefont=dict(color='white'),  # Etiqueta del eje y en blanco
                tickfont=dict(color='white')  # Etiquetas de los ticks en blanco
            ),
            hovermode="x unified",
            plot_bgcolor='#1D1E2C',  # Fondo del área de la gráfica
            paper_bgcolor='#1D1E2C',  # Fondo del área completa de la gráfica
            font=dict(color='white')  # Color del texto general
        )

        fig.update_xaxes(showgrid=False)  # Oculta líneas de cuadrícula verticales
        fig.update_yaxes(showgrid=False)  # Oculta líneas de cuadrícula horizontales

        st.plotly_chart(fig)


    # Tab 2: Portafolios Óptimos
    with tab2:
        st.markdown(
        """
        <div style="
            background-color: #C4F5FC;
            padding: 8px;
            border-radius: 20px;
            color: black;
            text-align: center;
        ">
            <h1 style="margin: 0; color: #black; font-size: 25px; ">Portafolios Óptimos</h1>
        </div>
        """,
        unsafe_allow_html=True,
        )

        # Optimización de portafolios
        def optimizar_portafolio(rendimientos, objetivo="sharpe", rendimiento_objetivo=None):
            media = rendimientos.mean() * 252
            covarianza = rendimientos.cov() * 252
            num_activos = len(media)
            pesos_iniciales = np.ones(num_activos) / num_activos
            limites = [(0, 1) for _ in range(num_activos)]  # Restricción: pesos entre 0 y 1 (sin posiciones cortas)
            restricciones = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]  # Restricción: suma de pesos = 1

            if objetivo == "sharpe":
                def objetivo_func(pesos):
                    rendimiento = np.dot(pesos, media)
                    riesgo = np.sqrt(np.dot(pesos.T, np.dot(covarianza, pesos)))
                    return -rendimiento / riesgo
            elif objetivo == "volatilidad":
                def objetivo_func(pesos):
                    return np.sqrt(np.dot(pesos.T, np.dot(covarianza, pesos)))
            elif objetivo == "rendimiento":
                restricciones.append({'type': 'eq', 'fun': lambda x: np.dot(x, media) - rendimiento_objetivo})
                def objetivo_func(pesos):
                    return np.sqrt(np.dot(pesos.T, np.dot(covarianza, pesos)))

            resultado = sco.minimize(objetivo_func, pesos_iniciales, method='SLSQP', bounds=limites, constraints=restricciones)
            return resultado.x


        # Optimización de los tres portafolios
        pesos_sharpe = optimizar_portafolio(rendimientos, objetivo="sharpe")
        pesos_volatilidad = optimizar_portafolio(rendimientos, objetivo="volatilidad")
        pesos_rendimiento = optimizar_portafolio(rendimientos, objetivo="rendimiento", rendimiento_objetivo=0.10)

        # Resultados en DataFrame
        pesos_df = pd.DataFrame({
            "Máximo Sharpe": pesos_sharpe,
            "Mínima Volatilidad": pesos_volatilidad,
            "Mínima Volatilidad (Rendimiento 10%)": pesos_rendimiento
        }, index=etfs)

        # Selección de portafolio
        portafolio_seleccionado = st.selectbox(
            "Selecciona el portafolio a visualizar:",
            ["Máximo Sharpe", "Mínima Volatilidad", "Mínima Volatilidad (Rendimiento 10%)"]
        )

        # Gráfico de barras para los pesos
        st.subheader(f"Pesos del Portafolio: {portafolio_seleccionado}")

        # Crear gráfica personalizada con Plotly
        fig_barras = go.Figure(data=[
            go.Bar(
                x=pesos_df.index,
                y=pesos_df[portafolio_seleccionado],
                marker_color='#2CA58D'  # Cambia el color de las barras si lo deseas
            )
        ])

        # Configuración de diseño para eliminar fondo y texto blanco
        fig_barras.update_layout(
            title=dict(
                text=f"Pesos del Portafolio: {portafolio_seleccionado}",
                font=dict(color='white')
            ),
            xaxis=dict(
                title="ETFs",
                titlefont=dict(color='white'),
                tickfont=dict(color='white')  # Texto blanco para etiquetas del eje X
            ),
            yaxis=dict(
                title="Pesos",
                titlefont=dict(color='white'),
                tickfont=dict(color='white')  # Texto blanco para etiquetas del eje Y
            ),
            plot_bgcolor='rgba(0,0,0,0)',  # Fondo transparente
            paper_bgcolor='rgba(0,0,0,0)',  # Fondo transparente
            font=dict(color='white')  # Texto general en blanco
        )
        
        # Mostrar el gráfico
        st.plotly_chart(fig_barras)
        
        
        
        # Gráfico de pastel para la composición del portafolio
        st.subheader("Composición del Portafolio")

        # Ajustar valores y etiquetas
        valores_redondeados = [round(peso, 6) if peso > 1e-6 else 0 for peso in pesos_df[portafolio_seleccionado]]
        etiquetas = [
            f"{etf} ({peso:.6f})" if peso > 0 else f"{etf} (<1e-6)"
            for etf, peso in zip(etfs, pesos_df[portafolio_seleccionado])
        ]

        # Crear gráfica de pastel
        fig_pastel = go.Figure(data=[
            go.Pie(
                labels=etiquetas,
                values=valores_redondeados,
                hoverinfo='label+percent+value',
                textinfo='percent',  # Muestra porcentajes en la gráfica
                marker=dict(colors=['#2CA58D', '#F46197', '#84BC9C', '#FFD700', '#497076'])  # Colores personalizados
            )
        ])

        # Configuración del diseño para eliminar fondo y texto blanco
        fig_pastel.update_layout(
            title=dict(
                text=f"Distribución del Portafolio ({portafolio_seleccionado})",
                font=dict(color='white')  # Título en blanco
            ),
            legend=dict(
                font=dict(color='white'),  # Texto blanco para la leyenda
                bgcolor='rgba(0,0,0,0)'  # Fondo transparente para la leyenda
            ),
            plot_bgcolor='rgba(0,0,0,0)',  # Fondo transparente
            paper_bgcolor='rgba(0,0,0,0)',  # Fondo transparente
            font=dict(color='white')  # Texto general en blanco
        )

        #Mostrar el gráfico
        st.plotly_chart(fig_pastel)


    # Tab 3: Comparación de Portafolios
    with tab3:
        st.header("Comparación de Portafolios")
        st.subheader("Precios Normalizados de los ETFs Seleccionados")

        # Normalización de precios
        precios_normalizados = datos / datos.iloc[0] * 100

        # Gráfico de comparación
        fig = go.Figure()
        for etf in etfs:
            fig.add_trace(go.Scatter(
                x=precios_normalizados.index,
                y=precios_normalizados[etf],
                mode='lines',
                name=etf
            ))

        fig.update_layout(
            title="Comparación de Precios Normalizados",
            xaxis_title="Fecha",
            yaxis_title="Precio Normalizado",
            hovermode="x unified"
        )
        st.plotly_chart(fig)

    # Tab 4: Black-Litterman
    with tab4:
        st.header("Modelo de Optimización Black-Litterman")

        # Definición de la matriz P (opiniones de mercado) y Q (rendimientos esperados por esas opiniones)
        P = np.array([
            [1, 0, 0, 0, 0],  # LQD tiene un rendimiento esperado de 3%
            [0, 1, 0, 0, 0],  # EMB tiene un rendimiento esperado de 6%
            [0, 0, 1, 0, 0],  # ACWI tiene un rendimiento esperado de 8%
            [0, 0, 0, 1, 0],  # SPY tiene un rendimiento esperado de 10%
            [0, 0, 0, 0, 1]   # WMT tiene un rendimiento esperado de 5%
        ])
        Q = np.array([0.03, 0.06, 0.08, 0.10, 0.05])  # Rendimientos esperados

        # Datos necesarios para el modelo
        media_rendimientos = rendimientos.mean() * 252
        covarianza_rendimientos = rendimientos.cov() * 252

        # Implementación del modelo Black-Litterman
        def black_litterman_optimizar(media_rendimientos, covarianza_rendimientos, P, Q, tau=0.05):
            """
            Calcula los pesos óptimos ajustados con el modelo Black-Litterman.

            media_rendimientos: Rendimientos esperados implícitos del mercado.
            covarianza_rendimientos: Matriz de covarianza de rendimientos.
            P: Matriz que representa las opiniones del mercado.
            Q: Vector de rendimientos esperados por las opiniones.
            tau: Escalar que ajusta la incertidumbre de la covarianza.
            """
            # Rendimientos implícitos del mercado
            pi = media_rendimientos
            omega = np.diag(np.diag(P @ covarianza_rendimientos @ P.T)) * tau  # Incertidumbre de las opiniones

            # Cálculo de los rendimientos ajustados
            medio_ajustado = np.linalg.inv(
                np.linalg.inv(tau * covarianza_rendimientos) + P.T @ np.linalg.inv(omega) @ P
            ) @ (
                np.linalg.inv(tau * covarianza_rendimientos) @ pi + P.T @ np.linalg.inv(omega) @ Q
            )

            # Optimización de los pesos ajustados
            num_activos = len(media_rendimientos)
            limites = [(0, 1) for _ in range(num_activos)]
            restricciones = [{'type': 'eq', 'fun': lambda x: np.sum(x) - 1}]

            def objetivo_func(pesos):
                return -np.dot(pesos, medio_ajustado) / np.sqrt(np.dot(pesos.T, np.dot(covarianza_rendimientos, pesos)))

            resultado = sco.minimize(objetivo_func, np.ones(num_activos) / num_activos, bounds=limites, constraints=restricciones)
            return resultado.x

        # Cálculo de los pesos ajustados
        pesos_black_litterman = black_litterman_optimizar(media_rendimientos, covarianza_rendimientos, P, Q)

        # Mostrar resultados
        st.subheader("Pesos del Portafolio Ajustado con el Modelo Black-Litterman")
        for etf, peso in zip(etfs, pesos_black_litterman):
            st.write(f"{etf}: {peso:.2%}")

        # Gráfico de barras para los pesos
        fig_black_litterman = go.Figure(data=[
            go.Bar(x=etfs, y=pesos_black_litterman, text=[f"{p:.2%}" for p in pesos_black_litterman], textposition='auto')
        ])
        fig_black_litterman.update_layout(
            title="Pesos Ajustados - Black-Litterman",
            xaxis_title="ETF",
            yaxis_title="Peso",
            template="plotly_white"
        )
        st.plotly_chart(fig_black_litterman)
