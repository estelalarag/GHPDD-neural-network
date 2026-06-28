import tensorflow as tf
import matplotlib.pyplot as plt

from pathlib import Path

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score
)

from preprocessing_v2 import load_data, preprocess_data


# =========================
# RUTAS
# =========================

BASE_DIR = Path(__file__).resolve().parent.parent


DATA_PATH = BASE_DIR / "data" / "house_purchase.csv"

MODEL_PATH = (
    BASE_DIR /
    "results" /
    "experiment_04_tuned.keras"
)


RESULTS_PATH = (
    BASE_DIR /
    "results" /
    "evaluation_tuned"
)


RESULTS_PATH.mkdir(
    parents=True,
    exist_ok=True
)


# =========================
# CARGAR DATOS
# =========================

df = load_data(DATA_PATH)


X_train, X_val, X_test, y_train, y_val, y_test, _ = preprocess_data(df)



# =========================
# CARGAR MODELO
# =========================

model = tf.keras.models.load_model(
    MODEL_PATH
)


model.summary()



# =========================
# EVALUACIÓN
# =========================

results = model.evaluate(
    X_test,
    y_test
)

print("\nTEST RESULTS")

for name, value in zip(
    model.metrics_names,
    results
):
    print(
        name,
        ":",
        value
    )

#print("\nTEST RESULTS")

#print("Loss:", loss)

#print("Accuracy:", accuracy)



# =========================
# PREDICCIONES
# =========================

y_prob = model.predict(
    X_test
)


threshold = 0.5


y_pred = (
    y_prob > threshold
).astype(int)



# =========================
# MÉTRICAS
# =========================

precision = precision_score(
    y_test,
    y_pred
)


recall = recall_score(
    y_test,
    y_pred
)


f1 = f1_score(
    y_test,
    y_pred
)


print("\nMETRICS")

print(
    "Precision:",
    precision
)

print(
    "Recall:",
    recall
)

print(
    "F1:",
    f1
)



print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        y_pred
    )
)



# =========================
# MATRIZ CONFUSIÓN
# =========================

cm = confusion_matrix(
    y_test,
    y_pred
)


plt.figure()

plt.imshow(cm)

plt.title(
    "Matriz de Confusión - Modelo Optimizado"
)

plt.xlabel(
    "Predicción"
)

plt.ylabel(
    "Real"
)


plt.colorbar()


plt.savefig(
    RESULTS_PATH /
    "confusion_matrix.png"
)


plt.close()



# =========================
# GUARDAR RESULTADOS
# =========================

print("\nEvaluación finalizada")