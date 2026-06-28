import keras_tuner as kt
import tensorflow as tf


from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout



def build_model(hp):


    model = Sequential()



    for i in range(
        hp.Int(
            "num_layers",
            min_value=2,
            max_value=4
        )
    ):


        model.add(

            Dense(

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

            Dropout(

                hp.Float(
                    f"dropout_{i}",
                    0.1,
                    0.5,
                    step=0.1
                )

            )

        )



    model.add(

        Dense(
            1,
            activation="sigmoid"
        )

    )



    model.compile(

        optimizer=tf.keras.optimizers.Adam(

            learning_rate=

            hp.Choice(

                "learning_rate",

                [
                    0.001,
                    0.0001
                ]

            )

        ),


        loss="binary_crossentropy",


        metrics=[
        "accuracy",
        tf.keras.metrics.Precision(name="precision"),
        tf.keras.metrics.Recall(name="recall")
        ]

    )


    return model





tuner = kt.RandomSearch(


    build_model,


        objective=kt.Objective(
            "val_precision",
            direction="max"
        ),


    max_trials=5,


    directory="tuner_results",


    project_name="house_purchase_f1_v2"

)