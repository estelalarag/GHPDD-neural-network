import numpy as np
import keras_tuner as kt
import tensorflow as tf

from pathlib import Path

from preprocessing_v2 import load_data, preprocess_data


# =========================
# RUTAS
# =========================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "house_purchase.csv"



# =========================
# DATOS
# =========================


df = load_data(DATA_PATH)


X_train, X_val, X_test, y_train, y_val, y_test, _ = preprocess_data(df)


input_shape = X_train.shape[1]



# =========================
# MODELO PARA TUNER
# =========================


def build_model(hp):


    model = tf.keras.Sequential()



    # cantidad de capas ocultas
    num_layers = hp.Int(
        "num_layers",
        min_value=2,
        max_value=4
    )



    for i in range(num_layers):


        model.add(

            tf.keras.layers.Dense(

                units=hp.Int(

                    f"units_{i}",

                    min_value=32,

                    max_value=256,

                    step=32

                ),

                activation="relu"

            )

        )


        model.add(

            tf.keras.layers.Dropout(

                hp.Float(

                    f"dropout_{i}",

                    min_value=0.1,

                    max_value=0.5,

                    step=0.1

                )

            )

        )



    model.add(

        tf.keras.layers.Dense(

            1,

            activation="sigmoid"

        )

    )



    model.compile(

        optimizer=tf.keras.optimizers.Adam(

            learning_rate=hp.Choice(

                "learning_rate",

                [

                    0.001,

                    0.0005,

                    0.0001

                ]

            )

        ),


        loss="binary_crossentropy",


        metrics=[

            "accuracy",

            tf.keras.metrics.Precision(),

            tf.keras.metrics.Recall()

        ]

    )


    return model




# =========================
# CLASS WEIGHTS
# =========================


from sklearn.utils.class_weight import compute_class_weight


weights = compute_class_weight(

    class_weight="balanced",

    classes=np.unique(y_train),

    y=y_train

)


class_weights = dict(

    enumerate(weights)

)



print(class_weights)



# =========================
# TUNER
# =========================


tuner = kt.RandomSearch(

    build_model,


    objective="val_recall",


    max_trials=5,


    executions_per_trial=1,


    directory="tuner_results",


    project_name="house_purchase_v2"

)



# =========================
# BUSQUEDA
# =========================


tuner.search(

    X_train,

    y_train,

    epochs=30,

    validation_data=(

        X_val,

        y_val

    ),

    batch_size=64,

    class_weight=class_weights

)



# =========================
# MEJOR MODELO
# =========================


best_model = tuner.get_best_models(

    num_models=1

)[0]



best_model.summary()



best_model.save(

    BASE_DIR /

    "results" /

    "experiment_04_tuned.keras"

)


print("Modelo optimizado guardado")