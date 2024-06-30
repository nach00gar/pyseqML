# Herramienta Python para pipeline inteligent de datos de secuenciación RNASeq

Autor: Ignacio Garach Vélez

Máster en Ciencia de Datos e Ingeniería de Computadores por la Universidad de Granada

Este repositorio contiene una herramienta desarrollada en Python para llevar a cabo un pipeline completo de datos de secuenciación RNASeq. La herramienta incluye funcionalidades para el análisis de expresión génica y clasificación en datos transcriptómicos, facilitando la búsqueda de biomarcadores de enfermedades como el cáncer.

## Instrucciones de Uso
Clona este repositorio en tu máquina local.
Instala los requisitos del proyecto ejecutando pip install -r requirements.txt.
Ejecuta la aplicación desde el archivo app.py.
Accede a la interfaz web desde tu navegador, por defecto en el puerto 8050.

## Funcionalidades Principales
Carga de datos desde ficheros Count o a partir de una matriz de expresión y etiquetas de muestras.
Normalización CQN (Conditional Quantile Normalization).
Detección de Outliers.
Limpieza de efectos Batch (Surrogate Variable Analysis).
Implementación del core de la librería limma de R.
Análisis de expresión génica diferencial. Posibilidad de resolver problemas multiclase mediante cobertura.
Feature Selection y Clasificación para enriquecer los resultados y encontrar conjunto reducido de marcadores.
