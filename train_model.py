import pandas as pd
import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

print(">>> A gerar dados sintéticos...")

np.random.seed(42)
n_samples = 500

idades = np.random.randint(18, 71, n_samples)
rendimentos = np.random.lognormal(mean=10.5, sigma=0.8, size=n_samples)
rendimentos = np.round(rendimentos, -2)
historico = np.random.randint(0, 16, n_samples)

df = pd.DataFrame(
    {"idade": idades, "rendimento": rendimentos, "historico_emprestimos": historico}
)


def classificar_risco(row):
    score = 0

    if row["idade"] < 22 or row["idade"] > 60:
        score += 2

    if row["rendimento"] < 30000:
        score += 3
    elif row["rendimento"] < 80000:
        score += 2
    elif row["rendimento"] < 200000:
        score += 1

    if row["historico_emprestimos"] > 5:
        score -= 2
    elif row["historico_emprestimos"] == 0:
        score += 1

    if score <= 0:
        return "BAIXO"
    elif score <= 2:
        return "MEDIO"
    elif score <= 4:
        return "ALTO"
    else:
        return "MUITO_ALTO"


df["risco_label"] = df.apply(classificar_risco, axis=1)

print("\n>>> Exemplo dos dados gerados:")
print(df.head())
print(f"\n>>> Distribuição das Classes:\n{df['risco_label'].value_counts()}")

X = df[["idade", "rendimento", "historico_emprestimos"]]
y = df["risco_label"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\n>>> A treinar modelo com {len(X_train)} exemplos...")

modelo_rf = RandomForestClassifier(n_estimators=100, random_state=42)
modelo_rf.fit(X_train, y_train)

y_pred = modelo_rf.predict(X_test)

acuracia = accuracy_score(y_test, y_pred)
print(f"\n>>> Acurácia do Modelo: {acuracia * 100:.2f}%")
print("\n>>> Relatório de Classificação:")
print(classification_report(y_test, y_pred))

arquivo_modelo = "modelo_rf.pkl"
joblib.dump(modelo_rf, arquivo_modelo)
print(f"\n>>> Modelo salvo com sucesso em '{arquivo_modelo}'!")
