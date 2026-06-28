# Predicción de intención de compra inmobiliaria usando Redes Neuronales Profundas


## Objetivo

Predecir si un cliente tiene intención de comprar una propiedad utilizando sus características financieras y del inmueble.


## Dataset

Global House Purchase Decision Dataset - Kaggle


Variables utilizadas:

- salario
- precio propiedad
- crédito
- gastos mensuales
- satisfacción
- ubicación
- características del inmueble


## Estrategia

Se realizó:

1. Limpieza y análisis exploratorio
2. Codificación de variables categóricas
3. Normalización de variables numéricas
4. División:

- 70% entrenamiento
- 15% validación
- 15% prueba


## Modelo

Red neuronal profunda:

- Dense layers
- ReLU
- Dropout
- Adam optimizer
- Binary Crossentropy


## Optimización

Se utilizó Keras Tuner para evaluar diferentes arquitecturas:

- número de capas
- neuronas
- dropout
- learning rate


## Manejo del desbalance

Se aplicó class_weight para dar mayor importancia a la clase comprador.


## Resultados

Se evaluaron:

- Accuracy
- Precision
- Recall
- F1-score


Debido al problema de negocio, se priorizó la detección de compradores potenciales.


## Estructura

src/
- preprocessing
- training
- tuning
- evaluation


results/
- gráficas de entrenamiento
- modelos