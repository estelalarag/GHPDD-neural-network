from pathlib import Path

import tensorflow as tf
import matplotlib.pyplot as plt


from preprocessing_v2 import (
    load_data,
    preprocess_data
)


BASE_DIR = Path(__file__).resolve().parent.parent


DATA_PATH = BASE_DIR / "data" / "house_purchase.csv"


RESULTS = BASE_DIR / "results" / "experiment_05"

RESULTS.mkdir(
    parents=True,
    exist_ok=True
)


df = load_data(DATA_PATH)


X_train, X_val, X_test, y_train, y_val, y_test, _ = preprocess_data(df)



model = tf.keras.models.load_model(

    RESULTS /
    "model_experiment_05.keras"

)



history = model.fit(

    X_train,

    y_train,

    validation_data=(

        X_val,
        y_val

    ),

    epochs=20,

    batch_size=64

)



plt.figure()

plt.plot(
    history.history["loss"],
    label="train"
)

plt.plot(
    history.history["val_loss"],
    label="validation"
)


plt.title(
    "Loss Experiment 05"
)

plt.legend()


plt.savefig(
    RESULTS /
    "loss_curve.png"
)


plt.close()



plt.figure()


plt.plot(
    history.history["accuracy"],
    label="train"
)


plt.plot(
    history.history["val_accuracy"],
    label="validation"
)


plt.title(
    "Accuracy Experiment 05"
)


plt.legend()


plt.savefig(

    RESULTS /
    "accuracy_curve.png"

)


plt.close()



print("Graficas generadas")