from tuning_v2 import tuner
from preprocessing_v2 import load_data, preprocess_data

from pathlib import Path

import tensorflow as tf


BASE_DIR = Path(__file__).resolve().parent.parent


DATA_PATH = BASE_DIR / "data" / "house_purchase.csv"


df = load_data(DATA_PATH)


X_train, X_val, X_test, y_train, y_val, y_test, _ = preprocess_data(df)



early_stop = tf.keras.callbacks.EarlyStopping(

    monitor="val_loss",

    patience=3,

    restore_best_weights=True

)



tuner.search(

    X_train,
    y_train,

    epochs=15,

    validation_data=(

        X_val,
        y_val

    ),

    batch_size=64,

    callbacks=[early_stop]

)



best_model = tuner.get_best_models(1)[0]



best_model.save(

    BASE_DIR /

    "results" /

    "experiment_04_tuned.keras",

    include_optimizer=False

)