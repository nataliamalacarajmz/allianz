import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import os

# Lista de ETFs disponibles con sus símbolos de Yahoo Finance
etfs = {
    "AZ China": "ASHR",
    "AZ MSCI TAIWAN INDEX FD": "EWT",
    "AZ RUSSELL 2000": "IWM",
    "AZ Brasil": "EWZ",
    "AZ MSCI UNITED KINGDOM": "EWU",
    # Agrega los demás ETFs aquí...
}

# Aplicar el estilo de Seaborn a los gráficos
sns.set(style="whitegrid")
sns.set_palette("muted")

# Título principal
st.markdown("<h1 style='text-align: center; color: #003366;'>Simulador de Inversión Allianz</h1>", unsafe_allow_html=True)

# Instrucción amigable
st.markdown("<h3 style='color: #336699;'>Selecciona uno, dos o tres ETFs para comparar su rendimiento y simular la inversión:</h3>", unsafe_allow_html=True)

# Selector de ETF
etf_nombres = list(etfs.keys())
seleccion_etfs = st.multiselect('Selecciona uno, dos o tres ETFs para comparar', etf_nombres, default=[etf_nombres[0]])

# Verificar selección de ETFs
if 1 <= len(seleccion_etfs) <= 3:
    try:
        # Descargar los datos del S&P 500
        sp500 = yf.download('^GSPC', period='10y')['Close']
        if sp500.empty:
            st.error("No se pudieron obtener los datos del S&P 500.")
    except Exception as e:
        st.error(f"Error al descargar datos del S&P 500: {e}")

    # Selector de periodos de tiempo
    periodos = ['1mo', '3mo', '6mo', '1y', 'ytd', '5y', '10y']
    seleccion_periodo = st.selectbox('Selecciona el periodo de tiempo', periodos)

    # Monto de inversión
    monto_inicial = st.number_input("Introduce el monto inicial de inversión ($)", min_value=100.0, value=1000.0)
    tasa_libre_riesgo = 2.0

    # Barra de progreso para indicar la descarga de datos
    if st.button('Simular inversión y comparar ETFs'):
        st.write("Descargando datos...")
        progress_bar = st.progress(0)

        # Descargar los datos de cada ETF seleccionado
        datos_list = []
        for idx, etf_nombre in enumerate(seleccion_etfs):
            simbolo = etfs[etf_nombre]
            try:
                datos = yf.download(simbolo, period=seleccion_periodo)
                st.write(f"Datos descargados para {etf_nombre}:")
                st.write(datos.head())  # Mostrar las primeras filas para depuración

                if datos.empty:
                    st.error(f"No se pudieron obtener datos para el ETF {etf_nombre}.")
                    continue

                # Verificar si 'Close' está en las columnas
                if 'Close' not in datos.columns:
                    st.error(f"El ETF {etf_nombre} no tiene datos de cierre ('Close').")
                    continue

                datos_list.append((etf_nombre, datos))
                progress_bar.progress((idx + 1) / len(seleccion_etfs))
            except Exception as e:
                st.error(f"Error al descargar datos para {etf_nombre}: {e}")

        # Si no hay datos válidos, no se continúa con los cálculos
        if not datos_list:
            st.error("No se encontraron datos válidos para los ETFs seleccionados.")
        else:
            # Función de cálculo
            def calcular_rendimiento_riesgo(datos, tasa_libre_riesgo, sp500):
                if len(datos) > 1:
                    try:
                        # Calcular rendimiento como un valor numérico (extraemos un único valor en lugar de una serie)
                        rendimiento = ((datos['Close'].iloc[-1] - datos['Close'].iloc[0]) / datos['Close'].iloc[0]) * 100
                        rendimiento = float(rendimiento)  # Aseguramos que sea un número
                        
                        # Calcular volatilidad (anualizada) como un valor numérico
                        rendimientos_diarios = datos['Close'].pct_change().dropna()
                        volatilidad = rendimientos_diarios.std() * np.sqrt(252)
                        volatilidad = float(volatilidad)  # Aseguramos que sea un número

                        # Calcular la Beta comparando con S&P 500
                        retornos_mercado = sp500.pct_change().dropna()
                        datos_alineados = pd.concat([rendimientos_diarios, retornos_mercado], axis=1, join="inner").dropna()
                        beta = np.cov(datos_alineados.iloc[:, 0], datos_alineados.iloc[:, 1])[0, 1] / np.var(datos_alineados.iloc[:, 1])
                        beta = float(beta)  # Aseguramos que sea un número

                        # Calcular el máximo drawdown como un valor numérico
                        max_value = datos['Close'].cummax()
                        drawdown = (datos['Close'] - max_value) / max_value
                        max_drawdown = drawdown.min() * 100
                        max_drawdown = float(max_drawdown)  # Aseguramos que sea un número

                        # Calcular el Alpha como un valor numérico
                        rendimiento_mercado = ((sp500.iloc[-1] - sp500.iloc[0]) / sp500.iloc[0]) * 100
                        alpha = rendimiento - (tasa_libre_riesgo + beta * (rendimiento_mercado - tasa_libre_riesgo))
                        alpha = float(alpha)  # Aseguramos que sea un número

                        return rendimiento, volatilidad, beta, max_drawdown, alpha
                    except KeyError as e:
                        st.error(f"Error de columna faltante: {e}")
                        return None, None, None, None, None
                    except Exception as e:
                        st.error(f"Ocurrió un error al calcular el rendimiento y riesgo: {e}")
                        return None, None, None, None, None
                else:
                    st.error("Los datos no son suficientes para calcular el rendimiento.")
                    return None, None, None, None, None


            # Calcular resultados para cada ETF
            resultados = []
            for etf_nombre, datos in datos_list:
                rendimiento, volatilidad, beta, max_drawdown, alpha = calcular_rendimiento_riesgo(datos, tasa_libre_riesgo, sp500)
                if rendimiento is not None:
                    resultados.append((etf_nombre, rendimiento, volatilidad, beta, max_drawdown, alpha))

            # Mostrar resultados de cada ETF
            for nombre, rendimiento, volatilidad, beta, max_drawdown, alpha in resultados:
                st.markdown(f"<h4 style='color: #336699;'>Resultados para {nombre}:</h4>", unsafe_allow_html=True)
                st.write(f"Volatilidad: {volatilidad:.2f}")
                st.write(f"Beta: {beta:.2f}")
                st.write(f"Máximo drawdown: {max_drawdown:.2f}%")
                st.write(f"Alpha: {alpha:.2f}")
else:
    st.error("Por favor selecciona entre uno y tres ETFs para realizar la simulación.")