# Resumen de experimentos

Proyecto: Predicción de intención de compra inmobiliaria mediante red neuronal profunda


| Experimento | Estrategia | Accuracy | Precision | Recall | F1 |
|---|---|---:|---:|---:|---:|
|03|Red profunda + regularización + class weights|0.65|0.40|0.99|0.57|
|04|Keras Tuner optimizando Recall|0.65|0.40|0.99|0.57|
|05|Keras Tuner + ajuste de hiperparámetros|0.76|0.39|0.01|0.07|


## Observaciones

El dataset presenta desbalance de clases, por lo que accuracy no representa completamente el desempeño.

Los experimentos mostraron que:

- Optimizar recall permite identificar más compradores potenciales.
- Incrementar precisión reduce falsos positivos pero puede perder compradores.
- Se evaluaron diferentes configuraciones de red neuronal profunda mediante Keras Tuner.


El modelo seleccionado considera el equilibrio entre métricas según el objetivo del problema.