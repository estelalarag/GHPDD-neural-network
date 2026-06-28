# Resumen de experimentos

Proyecto: Predicción de intención de compra inmobiliaria mediante red neuronal profunda


| Experimento | Estrategia | Resultado |
|---|---|---|
|01|Modelo inicial de red neuronal profunda|Baseline inicial|
|02|Preprocesamiento y ajuste de variables|Mejora del modelo base|
|03|Red profunda + regularización + class weights|Recall elevado para compradores|
|04|Keras Tuner optimizando recall|Mayor sensibilidad hacia clase comprador|
|05|Keras Tuner + evaluación de precision/recall|Análisis del balance entre métricas|


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

## Conclusión

Debido al desbalance del dataset, accuracy por sí sola no fue suficiente. 
Los experimentos demostraron la importancia de evaluar precision, recall y F1-score para seleccionar una estrategia adecuada.