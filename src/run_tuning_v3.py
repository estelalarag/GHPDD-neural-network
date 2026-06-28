from tuning_v3 import tuner

from preprocessing_v2 import (
    load_data,
    preprocess_data
)


from pathlib import Path

import tensorflow as tf



BASE_DIR = Path(__file__).resolve().parent.parent


DATA_PATH = (
    BASE_DIR /
    "data" /
    "house_purchase.csv"
)



RESULT_PATH = (

    BASE_DIR /
    "results" /
    "experiment_05"

)


RESULT_PATH.mkdir(
    parents=True,
    exist_ok=True
)



df = load_data(
    DATA_PATH
)



X_train, X_val, X_test, y_train, y_val, y_test, _ = preprocess_data(df)

from sklearn.utils.class_weight import compute_class_weight
import numpy as np

class_weights = compute_class_weight(

    class_weight="balanced",

    classes=np.unique(y_train),

    y=y_train

)


class_weights = dict(
    enumerate(class_weights)
)


print(class_weights)

early_stop = tf.keras.callbacks.EarlyStopping(

    monitor="val_loss",

    patience=3,

    restore_best_weights=True

)


tuner.search(

    X_train,

    y_train,

    epochs=15,

    validation_data=(X_val,y_val),

    batch_size=64,

    callbacks=[early_stop],

    class_weight=class_weights

)



best_model = tuner.get_best_models(1)[0]



best_model.save(

    RESULT_PATH /
    "model_experiment_05.keras",

    include_optimizer=False

)



print("Modelo experimento 05 guardado")