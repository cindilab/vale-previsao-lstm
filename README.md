# Previsão do fechamento da VALE3 com LSTM

Projeto da Fase 5 da pós em Machine Learning Engineering (FIAP).

Modelo LSTM que prevê o fechamento do próximo pregão da Vale (VALE3.SA), servido por uma API FastAPI em Docker, com monitoramento de uso, de data drift e do erro real em produção.

- **API:** `https://vale-previsao-lstm.onrender.com` (documentação interativa em `/docs`)
- **Vídeo:** LINK DO VÍDEO

## Estrutura

```
app/
  main.py            rotas da API
  dados.py           download do Yahoo, features e drift (mesmo código do notebook)
  monitoramento.py   métricas da API e registro das previsões
model/
  vale_lstm.keras    modelo treinado
  scalers.joblib     scalers das features e do alvo
  model_info.json    features, janela, métricas e referência de drift
notebooks/
  previsao-fechamento-vale.ipynb   coleta, treino, validação e serialização
Dockerfile
requirements.txt
render.yaml
```

## Dados

- Fonte: Yahoo Finance (endpoint de chart, sem a biblioteca yfinance).
- Período: 2008 até hoje, cerca de 4.600 pregões.
- O treino usa o **fechamento ajustado**. Sem o ajuste, cada pagamento de dividendo apareceria como uma queda falsa.
- A previsão devolvida pela API é aplicada sobre o **fechamento real**, o mesmo valor que aparece no Google.
- Mantive os dias extremos (crise de 2008, Brumadinho, Covid): o modelo em produção também vai enfrentar eventos assim.

## Modelo

- Alvo: variação do fechamento no dia seguinte. Preço previsto = fechamento de hoje × (1 + variação prevista).
- Entrada: janela dos últimos 60 pregões.
- Arquitetura: LSTM(64) → Dropout(0,2) → LSTM(32) → Dropout(0,2) → Dense(1), com EarlyStopping.
- Scalers ajustados só com os dados de treino, para não vazar informação do teste.
- Baseline de comparação: "amanhã = hoje".

## Experimentos

| Versão | Ideia | Resultado |
|---|---|---|
| v1 | Prever o preço direto | MAE 1,29 contra 0,73 do baseline: o modelo só repetia o preço anterior com atraso |
| v2 | Prever a variação diária | Empate com o baseline |
| v3 | 5 features técnicas (amplitude, volume relativo, volatilidade, distância da média) | Empate com o baseline |
| v4 | Dados desde 2008 + BHP, Rio Tinto, Baosteel (grupo China Baowu), Ibovespa e dólar | Pior que v2 e v3 |

### Validação walk-forward

K-Fold comum mistura passado e futuro, então usei `TimeSeriesSplit`: 5 períodos de teste de cerca de 1 ano cada (2021 a 2026), sempre treinando só com o passado. Todas as opções são testadas nos mesmos dias.

| | Só variação | 5 features | Técnicas + mercado | Baseline |
|---|---|---|---|---|
| MAE médio (R$) | 0,7709 | **0,7678** | 0,7769 | 0,7698 |
| RMSE médio (R$) | 1,0304 | 1,0296 | 1,0394 | 1,0279 |
| Acerto de direção | 48,5% | 51,2% | 49,4% | – |
| Períodos em que venceu o baseline | 2 de 5 | 3 de 5 | 1 de 5 | – |

O notebook escolhe automaticamente a opção de menor MAE médio. O modelo final usa as **5 features técnicas**.

Rodei o notebook duas vezes, com um dia de diferença nos dados. Na primeira, a opção "só variação" ficou à frente (0,7686 contra 0,7717); na segunda, as 5 features. As diferenças são de cerca de R$ 0,003, menores que a variação entre execuções, então na prática **todas empatam com o baseline**. Os dados de mercado ficaram atrás nas duas.

### Teste final (80% treino / 20% teste, 10/02/2023 a 28/10/2026)

| | MAE (R$) | RMSE (R$) | MAPE |
|---|---|---|---|
| LSTM | 0,7020 | 0,9728 | 1,19% |
| Baseline | 0,6959 | 0,9657 | 1,19% |

Acerto de direção: 52,3%.

### Por que o modelo não supera o baseline

| Correlação da VALE3 com | No mesmo dia | Com o dia seguinte |
|---|---|---|
| BHP | ≈ 0,68 | ≈ 0,03 |
| Rio Tinto | ≈ 0,67 | ≈ 0,04 |
| Ibovespa | ≈ 0,66 | ≈ -0,03 |

A Vale anda junto com o setor no mesmo dia, mas essa relação desaparece para o dia seguinte: o mercado incorpora a informação no próprio dia. É o que se espera pela hipótese do mercado eficiente.

Na prática, o modelo prevê variações pequenas, perto de zero, com erro médio em torno de R$ 0,70 a 0,77 dependendo do período (cerca de 1,2% do preço). Todas as métricas estão em `model/model_info.json` e na rota `/modelo`.

## API

| Rota | O que faz |
|---|---|
| `GET /` | Status da API |
| `GET /previsao` | Baixa os dados recentes da VALE3 e prevê o próximo fechamento |
| `POST /prever` | Recebe os últimos pregões (mínimo 81, com fechamento, máxima, mínima e volume) e prevê o próximo fechamento |
| `GET /modelo` | Informações do modelo: features, métricas e referência de drift |
| `GET /monitoramento` | Uso da API, último drift e erro real das previsões já feitas |

Exemplo de resposta de `GET /previsao`:

```json
{
  "ticker": "VALE3.SA",
  "ultimo_pregao": "2026-10-29",
  "ultimo_fechamento": 63.10,
  "variacao_prevista_pct": 0.031,
  "fechamento_previsto": 63.12,
  "alerta_drift": false,
  "drift": {"retorno": {"distancia": -0.119, "alerta": false}, "...": "uma entrada por feature"}
}
```

Exemplo de `POST /prever` (a página `/docs` já traz um corpo de exemplo pronto para testar):

```bash
curl -X POST https://vale-previsao-lstm.onrender.com/prever \
  -H "Content-Type: application/json" \
  -d '{"pregoes": [{"fechamento": 60.1, "maxima": 60.8, "minima": 59.7, "volume": 25000000}, ...]}'
```

## Monitoramento

1. **Uso da API:** total de requisições, erros, tempo médio e p95 de resposta, e contagem por rota. Cada requisição também gera uma linha de log.
2. **Data drift:** a cada previsão, a média das features nos últimos 60 dias é comparada com a média do treino. Se a distância passar de 1 desvio padrão, a API devolve `alerta_drift: true` e registra um aviso no log. É o sinal para retreinar.
3. **Erro real em produção:** cada previsão é salva em `logs/previsoes.csv`. Quando o pregão seguinte acontece, `/monitoramento` compara o previsto com o fechamento real e calcula o MAE do modelo e do baseline em produção.

Limitação: no plano gratuito do Render o disco não é permanente, então o histórico de previsões é perdido quando o serviço reinicia. Em produção de verdade, isso iria para um banco de dados.

## Como rodar

Localmente:

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Com Docker:

```bash
docker build -t vale-api .
docker run -p 8000:8000 vale-api
```

Depois é só abrir `http://localhost:8000/docs`.

## Deploy

No Render, em New → Web Service, conectei este repositório. O Render detecta o `Dockerfile` e usa o `render.yaml`. A porta vem da variável `PORT`.

No plano gratuito, o serviço "dorme" depois de 15 minutos sem uso, e a primeira requisição depois disso pode levar cerca de 1 minuto.

## Retreino

1. Rodar o notebook no Kaggle (Internet ligada).
2. Substituir os 3 arquivos da pasta `model/`.
3. Fazer commit: o Render faz o deploy automaticamente.
