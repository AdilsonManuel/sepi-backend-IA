import pandas as pd
import numpy as np
import joblib
import os
from flask import Flask, request, jsonify

app = Flask(__name__)

# ==============================================================================
# 1. CARREGAMENTO DO MODELO (MODO PRODUÇÃO)
# ==============================================================================
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

# ==============================================================================
# 2. DEFINIÇÃO DA API
# ==============================================================================


@app.route("/prever-risco", methods=["POST"])
def prever_risco():
    if not modelo_rf:
        return jsonify(
            {"sucesso": False, "mensagem": "Modelo de IA não disponível."}
        ), 503

    try:
        dados = request.json
        print(f"\n>>> Recebido pedido de análise para ID: {dados.get('usuarioId')}")

        # 1. Extrair dados do JSON
        idade = dados.get("idade", 0)
        rendimento = dados.get("rendimentoMensalDeclarado", 0)
        historico = dados.get("historicoEmprestimos", 0)

        # 2. Criar DataFrame para previsão (mesma estrutura do treino)
        # Importante: Os nomes das colunas DEVEM ser iguais aos usados no treino
        novo_cliente = pd.DataFrame(
            {
                "idade": [idade],
                "rendimento": [rendimento],
                "historico_emprestimos": [historico],
            }
        )

        # 3. Fazer a Previsão com a IA
        previsao_risco = modelo_rf.predict(novo_cliente)[0]

        # Obter probabilidades (confiança da IA)
        probabilidades = modelo_rf.predict_proba(novo_cliente)
        confianca = np.max(probabilidades) * 100  # Ex: 85.5%

        print(f"   Dados: Idade={idade}, Rendimento={rendimento}")
        print(f"   Resultado IA: {previsao_risco} (Confiança: {confianca:.1f}%)")

        # 4. Calcular Limite Sugerido (Regra de Negócio baseada na previsão)
        # Nota: O Java recalcula isto, mas a IA pode sugerir um "teto inteligente"
        limite_sugerido = 0
        if previsao_risco == "BAIXO":
            limite_sugerido = 2500000
        elif previsao_risco == "MEDIO":
            limite_sugerido = 600000
        elif previsao_risco == "ALTO":
            limite_sugerido = 60000
        else:  # MUITO_ALTO
            limite_sugerido = 0

        # Ajuste fino: O limite não pode ser maior que 40% do rendimento anual (exemplo de regra extra da IA)
        teto_rendimento = (rendimento * 12) * 0.40
        if limite_sugerido > teto_rendimento and limite_sugerido > 0:
            limite_sugerido = teto_rendimento

        # 5. Retornar Resposta para o Spring Boot
        response = {
            "sucesso": True,
            "mensagem": "Análise realizada com sucesso.",
            "nivelRisco": previsao_risco,
            "scoreCredito": int(confianca * 10),  # Score de 0 a 1000
            "limiteSugerido": float(round(limite_sugerido, 2)),
        }

        return jsonify(response), 200

    except Exception as e:
        print(f"ERRO NA IA: {e}")
        return jsonify({"sucesso": False, "mensagem": str(e)}), 500


# ==============================================================================
# 3. INICIAR SERVIDOR
# ==============================================================================
if __name__ == "__main__":
    # Roda na porta 5000 (Padrão Flask)
    app.run(host="0.0.0.0", port=5000, debug=True)
