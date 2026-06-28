import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    BatchNormalization
)

from tensorflow.keras.regularizers import l2



def build_model(input_shape):


    model = Sequential([


        # Primera capa profunda
        Dense(
            256,
            activation="relu",
            input_shape=(input_shape,),
            kernel_regularizer=l2(0.001)
        ),

        BatchNormalization(),

        Dropout(0.3),



        # Segunda capa
        Dense(
            128,
            activation="relu",
            kernel_regularizer=l2(0.001)
        ),

        BatchNormalization(),

        Dropout(0.3),



        # Tercera capa
        Dense(
            64,
            activation="relu"
        ),

        Dropout(0.2),



        # Cuarta capa
        Dense(
            32,
            activation="relu"
        ),



        # salida binaria
        Dense(
            1,
            activation="sigmoid"
        )

    ])



    model.compile(

        optimizer=tf.keras.optimizers.Adam(
            learning_rate=0.001
        ),

        loss="binary_crossentropy",

        metrics=[

        "accuracy",

        tf.keras.metrics.Precision(
            name="precision"
        ),

        tf.keras.metrics.Recall(
            name="recall"
        )
    ]
    )


    return model



if __name__ == "__main__":


    model = build_model(81)

    model.summary()