# SEPI Backend IA 🧠

Este repositório contém o microserviço de Inteligência Artificial para o Sistema de Empréstimos (SEPI). Utiliza **Machine Learning (Random Forest)** para analisar o perfil do cliente e sugerir a aprovação de crédito, score e limites.

## 🚀 Tecnologias

- **Python 3.x**
- **Flask** (API REST)
- **Scikit-Learn** (Modelo de Machine Learning)
- **Pandas & NumPy** (Manipulação de Dados)
- **Joblib** (Persistência do Modelo)

## 📂 Estrutura do Projeto

- `app.py`: Servidor API Flask que serve as previsões.
- `train_model.py`: Script para gerar dados sintéticos, treinar e avaliar o modelo.
- `requirements.txt`: Dependências do projeto.
- `modelo_rf.pkl`: Arquivo do modelo treinado (gerado pelo script).

## 🛠️ Instalação e Configuração

1.  **Clone o repositório:**
    ```bash
    git clone https://github.com/AdilsonManuel/sepi-backend-IA.git
    cd sepi-backend-IA
    ```

2.  **Crie e ative um ambiente virtual (Opcional, mas recomendado):**
    ```bash
    python -m venv venv
    # Windows
    .\venv\Scripts\activate
    # Linux/Mac
    source venv/bin/activate
    ```

3.  **Instale as dependências:**
    ```bash
    pip install -r requirements.txt
    ```

## 🧠 Treinamento do Modelo

Antes de rodar a API, é necessário treinar o modelo. O script ira gerar dados sintéticos, treinar o Random Forest e salvar o arquivo `modelo_rf.pkl`.

```bash
python train_model.py
```

Você verá a acurácia do modelo e o relatório de classificação no terminal.

## ▶️ Executando a API

Após gerar o modelo, inicie o servidor Flask:

```bash
python app.py
```

A aplicação estará rodando em `http://localhost:5000`.

## 📡 Endpoints da API

### `POST /prever-risco`

Recebe os dados do cliente e retorna a análise de risco.

**Request Body (JSON):**
```json
{
  "usuarioId": 1,
  "idade": 30,
  "rendimentoMensalDeclarado": 50000,
  "historicoEmprestimos": 0
}
```

**Response (JSON):**
```json
{
  "sucesso": true,
  "mensagem": "Análise realizada com sucesso.",
  "nivelRisco": "ALTO",
  "scoreCredito": 530,
  "limiteSugerido": 60000.0
}
```

## 📝 Regras de Negócio (Simplificadas)

O modelo classifica o risco em: `BAIXO`, `MEDIO`, `ALTO`, `MUITO_ALTO`.
Com base nisso, o sistema sugere um limite de crédito, respeitando um teto de 40% do rendimento anual do cliente.

---
Desenvolvido por **Ngolax Techstruct Solutions**.
