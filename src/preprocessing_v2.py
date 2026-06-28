import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer


def load_data(path):

    df = pd.read_csv(path)

    print("Dimensiones:")
    print(df.shape)

    print("\nColumnas:")
    print(df.columns)

    print("\nPrimeras filas:")
    print(df.head())

    return df



def preprocess_data(df):

    # ===============================
    # 1. Separar variable objetivo
    # ===============================

    target = "decision"


    X = df.drop(
        target,
        axis=1
    )

# Eliminamos variables que pueden introducir
# información posterior a la decisión

    columns_to_remove = [
        "property_id",
        "satisfaction_score"
    ]


    X = X.drop(
        columns_to_remove,
        axis=1
    )


    y = df[target]


    # ===============================
    # 2. Identificar columnas
    # ===============================

    numerical_features = X.select_dtypes(
        include=["int64", "float64"]
    ).columns


    categorical_features = X.select_dtypes(
        include=["object"]
    ).columns


    print("\nVariables numéricas:")
    print(list(numerical_features))

    print("\nVariables categóricas:")
    print(list(categorical_features))


    # ===============================
    # 3. Pipeline numérico
    # ===============================

    numeric_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="median")
            ),
            (
                "scaler",
                StandardScaler()
            )
        ]
    )


    # ===============================
    # 4. Pipeline categórico
    # ===============================

    categorical_pipeline = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(strategy="most_frequent")
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore"
                )
            )
        ]
    )


    # ===============================
    # 5. Transformador completo
    # ===============================

    preprocessor = ColumnTransformer(
        transformers=[
            (
                "numeric",
                numeric_pipeline,
                numerical_features
            ),

            (
                "categorical",
                categorical_pipeline,
                categorical_features
            )
        ]
    )


    # ===============================
    # 6. Train / Validation / Test
    # ===============================


    X_train, X_temp, y_train, y_temp = train_test_split(
        X,
        y,
        test_size=0.30,
        random_state=42,
        stratify=y
    )


    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=0.50,
        random_state=42,
        stratify=y_temp
    )


    # aplicar transformación
    X_train = preprocessor.fit_transform(X_train)

    X_val = preprocessor.transform(X_val)

    X_test = preprocessor.transform(X_test)


    print("\nDatos después del procesamiento:")

    print("Train:", X_train.shape)
    print("Validation:", X_val.shape)
    print("Test:", X_test.shape)



    return (
        X_train,
        X_val,
        X_test,
        y_train,
        y_val,
        y_test,
        preprocessor
    )



if __name__ == "__main__":


    df = load_data(
        "../data/house_purchase.csv"
    )


    data = preprocess_data(df)
