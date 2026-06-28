import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.metrics import confusion_matrix, classification_report

import tensorflow as tf

from model import build_model
from preprocessing_v2 import load_data, preprocess_data


# =========================
# RUTAS
# =========================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "house_purchase.csv"
RESULTS_PATH = BASE_DIR / "results" / "experiment_03"

RESULTS_PATH.mkdir(parents=True, exist_ok=True)


# =========================
# CARGA Y PREPROCESAMIENTO
# =========================

df = load_data(DATA_PATH)

X_train, X_val, X_test, y_train, y_val, y_test, _ = preprocess_data(df)

input_shape = X_train.shape[1]


# =========================
# MODELO
# =========================

model = build_model(input_shape)

model.summary()


# =========================
# CALLBACKS IMPORTANTES
# =========================

early_stop = tf.keras.callbacks.EarlyStopping(
    monitor="val_loss",
    patience=8,
    restore_best_weights=True
)


from sklearn.utils.class_weight import compute_class_weight


class_weights = compute_class_weight(

    class_weight="balanced",

    classes=np.unique(y_train),

    y=y_train

)


class_weights = dict(
    enumerate(class_weights)
)


print(class_weights)

# =========================
# ENTRENAMIENTO
# =========================

history = model.fit(

    X_train,
    y_train,

    validation_data=(X_val,y_val),

    epochs=50,

    batch_size=64,

    callbacks=[early_stop],

    class_weight=class_weights

)


# =========================
# EVALUACIÓN
# =========================

test_results = model.evaluate(
    X_test,
    y_test
)


print("\nTEST RESULTS")

print("Loss:", test_results[0])

print("Accuracy:", test_results[1])

print("Precision:", test_results[2])

print("Recall:", test_results[3])


# =========================
# GRÁFICAS LOSS / ACCURACY
# =========================

plt.figure()
plt.plot(history.history["loss"], label="Train Loss")
plt.plot(history.history["val_loss"], label="Val Loss")
plt.legend()
plt.title("Loss durante entrenamiento")
plt.savefig(RESULTS_PATH / "loss_curve.png")
plt.close()


plt.figure()
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Val Accuracy")
plt.legend()
plt.title("Accuracy durante entrenamiento")
plt.savefig(RESULTS_PATH / "accuracy_curve.png")
plt.close()


# =========================
# MATRIZ DE CONFUSIÓN
# =========================

y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5).astype(int)

cm = confusion_matrix(y_test, y_pred)

print("\nClassification Report:\n")
print(classification_report(y_test, y_pred))


# =========================
# GUARDAR MODELO
# =========================

model.save(
    RESULTS_PATH / "model_experiment_03.keras"
)

print("\nModelo guardado correctamente")