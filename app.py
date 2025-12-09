import pandas as pd
import numpy as np
import joblib
import os
from flask import Flask, request, jsonify

app = Flask(__name__)

MODEL_FILE = "modelo_rf.pkl"
modelo_rf = None

if os.path.exists(MODEL_FILE):
    print(f">>> A carregar modelo treinado de '{MODEL_FILE}'...")
    try:
        modelo_rf = joblib.load(MODEL_FILE)
        print(">>> Modelo carregado com sucesso! IA pronta.")
    except Exception as e:
        print(f"!!! ERRO ao carregar modelo: {e}")
else:
    print(f"!!! AVISO: Arquivo '{MODEL_FILE}' não encontrado.")
    print("!!! Execute 'python train_model.py' para gerar o modelo.")


@app.route("/prever-risco", methods=["POST"])
def prever_risco():
    if not modelo_rf:
        return jsonify(
            {"sucesso": False, "mensagem": "Modelo de IA não disponível."}
        ), 503

    try:
        dados = request.json
        print(f"\n>>> Recebido pedido de análise para ID: {dados.get('usuarioId')}")

        idade = dados.get("idade", 0)
        rendimento = dados.get("rendimentoMensalDeclarado", 0)
        historico = dados.get("historicoEmprestimos", 0)

        novo_cliente = pd.DataFrame(
            {
                "idade": [idade],
                "rendimento": [rendimento],
                "historico_emprestimos": [historico],
            }
        )

        previsao_risco = modelo_rf.predict(novo_cliente)[0]

        probabilidades = modelo_rf.predict_proba(novo_cliente)
        confianca = np.max(probabilidades) * 100

        print(f"   Dados: Idade={idade}, Rendimento={rendimento}")
        print(f"   Resultado IA: {previsao_risco} (Confiança: {confianca:.1f}%)")

        limite_sugerido = 0
        if previsao_risco == "BAIXO":
            limite_sugerido = 2500000
        elif previsao_risco == "MEDIO":
            limite_sugerido = 600000
        elif previsao_risco == "ALTO":
            limite_sugerido = 60000
        else:
            limite_sugerido = 0

        teto_rendimento = (rendimento * 12) * 0.40
        if limite_sugerido > teto_rendimento and limite_sugerido > 0:
            limite_sugerido = teto_rendimento

        response = {
            "sucesso": True,
            "mensagem": "Análise realizada com sucesso.",
            "nivelRisco": previsao_risco,
            "scoreCredito": int(confianca * 10),
            "limiteSugerido": float(round(limite_sugerido, 2)),
        }

        return jsonify(response), 200

    except Exception as e:
        print(f"ERRO NA IA: {e}")
        return jsonify({"sucesso": False, "mensagem": str(e)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
