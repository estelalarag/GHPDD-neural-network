import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from pathlib import Path


# rutas
BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = BASE_DIR / "data" / "house_purchase.csv"

RESULTS_PATH = BASE_DIR / "results" / "plots"

RESULTS_PATH.mkdir(
    exist_ok=True,
    parents=True
)


# cargar datos

df = pd.read_csv(DATA_PATH)


print("Dimensiones:")
print(df.shape)


print("\nBalance de clases:")
print(df["decision"].value_counts())


# ==========================
# 1. Balance de clases
# ==========================

plt.figure(figsize=(6,4))

sns.countplot(
    data=df,
    x="decision"
)

plt.title(
    "Distribución de decisión de compra"
)

plt.xlabel(
    "Decisión (0 = No compra, 1 = Compra)"
)

plt.ylabel(
    "Cantidad de clientes"
)


plt.savefig(
    RESULTS_PATH / "balance_classes.png",
    dpi=300,
    bbox_inches="tight"
)

plt.close()



# ==========================
# 2. Correlaciones
# ==========================


corr = df.corr(
    numeric_only=True
)


plt.figure(
    figsize=(12,10)
)


sns.heatmap(
    corr,
    cmap="coolwarm",
    center=0
)


plt.title(
    "Mapa de correlaciones variables numéricas"
)


plt.savefig(
    RESULTS_PATH / "correlation_matrix.png",
    dpi=300,
    bbox_inches="tight"
)


plt.close()



# ==========================
# 3. Salario vs decisión
# ==========================


plt.figure(figsize=(7,5))


sns.boxplot(
    data=df,
    x="decision",
    y="customer_salary"
)


plt.title(
    "Relación entre salario y decisión de compra"
)


plt.savefig(
    RESULTS_PATH / "salary_vs_decision.png",
    dpi=300,
    bbox_inches="tight"
)


plt.close()



# ==========================
# 4. Precio vs decisión
# ==========================


plt.figure(figsize=(7,5))


sns.boxplot(
    data=df,
    x="decision",
    y="price"
)


plt.title(
    "Relación entre precio de propiedad y decisión"
)


plt.savefig(
    RESULTS_PATH / "price_vs_decision.png",
    dpi=300,
    bbox_inches="tight"
)


plt.close()



print("\nGráficas generadas correctamente")

decision_corr = corr["decision"].sort_values(
    ascending=False
)

print(decision_corr)

df.describe().to_csv(
    RESULTS_PATH / "dataset_summary.csv"
)

print(
    df.groupby("decision").mean(numeric_only=True)
)

for col in df.columns:

    print("\nVARIABLE:", col)

    print(
        df.groupby("decision")[col].nunique()
    )

    from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


X = df.drop("decision", axis=1)

X = pd.get_dummies(X)

y = df["decision"]


model = RandomForestClassifier(
    n_estimators=100,
    random_state=42
)


model.fit(X,y)


pred = model.predict(X)


print(
    accuracy_score(y,pred)
)