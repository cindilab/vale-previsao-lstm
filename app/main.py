import json
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path
from threading import Lock

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel, Field

from app.dados import baixar_direto, baixar_mercado, limpar, preparar_features, verificar_drift
from app.monitoramento import Metricas, desempenho_em_producao, logger, salvar_previsao

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

PASTA_MODELO = Path(__file__).resolve().parent.parent / "model"

modelo = tf.keras.models.load_model(PASTA_MODELO / "vale_lstm.keras")
scalers = joblib.load(PASTA_MODELO / "scalers.joblib")
with open(PASTA_MODELO / "model_info.json", encoding="utf-8") as f:
    INFO = json.load(f)

TICKER = INFO["ticker"]
JANELA = INFO["janela_dias"]
FEATURES = INFO["features"]
MERCADO = INFO.get("mercado", {})
REF_DRIFT = INFO["drift"]["referencia_treino"]
LIMITE_DRIFT = INFO["drift"]["limite_desvios"]

CACHE_MINUTOS = 15
DIAS_HISTORICO = 400  # dias corridos: sobra para a janela e para avaliar previsões antigas

metricas = Metricas()
_cache = {"quando": 0.0, "dados": None, "mercado": None}
_cache_lock = Lock()

app = FastAPI(
    title="Previsão VALE3",
    description="LSTM que prevê o fechamento do próximo pregão da Vale (VALE3.SA).",
    version="1.0.0",
)


@app.middleware("http")
async def medir_tempo(request: Request, call_next):
    inicio = time.perf_counter()
    status = 500
    try:
        resposta = await call_next(request)
        status = resposta.status_code
        return resposta
    finally:
        ms = (time.perf_counter() - inicio) * 1000
        metricas.registrar(request.url.path, status, ms)
        logger.info("%s %s -> %s (%.0f ms)", request.method, request.url.path, status, ms)


def dados_recentes():
    with _cache_lock:
        if _cache["dados"] is not None and time.time() - _cache["quando"] < CACHE_MINUTOS * 60:
            return _cache["dados"], _cache["mercado"]
        inicio = (datetime.now() - timedelta(days=DIAS_HISTORICO)).strftime("%Y-%m-%d")
        try:
            df = limpar(baixar_direto(TICKER, inicio))
            mercado = baixar_mercado(MERCADO, inicio, df.index) if MERCADO else None
        except Exception as erro:
            logger.error("falha ao baixar dados do Yahoo: %s", erro)
            raise HTTPException(status_code=503, detail="Não foi possível baixar os dados do Yahoo Finance agora.")
        _cache.update(quando=time.time(), dados=df, mercado=mercado)
        return df, mercado


def prever(df, mercado=None):
    feats = preparar_features(df, mercado)[FEATURES].dropna().tail(JANELA)
    if len(feats) < JANELA:
        raise HTTPException(status_code=422, detail=f"Dados insuficientes: preciso de {JANELA} dias com features completas.")
    entrada = scalers["X"].transform(feats.values).reshape(1, JANELA, len(FEATURES))
    saida = modelo(entrada, training=False).numpy()
    variacao = float(scalers["y"].inverse_transform(saida)[0, 0])
    drift = verificar_drift(feats, REF_DRIFT, LIMITE_DRIFT)
    alerta = any(d["alerta"] for d in drift.values())
    metricas.ultimo_drift = {"verificado_em": datetime.now().isoformat(timespec="seconds"), "alerta": alerta, "features": drift}
    if alerta:
        logger.warning("DRIFT detectado: %s", drift)
    return variacao, drift, alerta


class Fechamentos(BaseModel):
    fechamentos: list[float] = Field(
        ...,
        description=f"Fechamentos diários em ordem cronológica (mínimo {JANELA + 1}).",
        examples=[[60 + i * 0.05 for i in range(JANELA + 1)]],
    )


@app.get("/")
def raiz():
    return {"status": "ok", "ticker": TICKER, "docs": "/docs",
            "rotas": ["/previsao", "/prever", "/modelo", "/monitoramento"]}


@app.get("/modelo")
def sobre_o_modelo():
    return INFO


@app.get("/previsao")
def previsao():
    """Baixa os dados mais recentes da VALE3 e prevê o fechamento do próximo pregão."""
    df, mercado = dados_recentes()
    variacao, drift, alerta = prever(df, mercado)

    ultimo = float(df["CloseBruto"].iloc[-1])  # preço real (igual ao do Google)
    previsto = ultimo * (1 + variacao)
    data_base = df.index[-1].date().isoformat()

    salvar_previsao({
        "registrado_em": datetime.now().isoformat(timespec="seconds"),
        "data_base": data_base,
        "fechamento_base": round(ultimo, 4),
        "variacao_prevista": round(variacao, 6),
        "fechamento_previsto": round(previsto, 4),
        "alerta_drift": alerta,
    })

    return {
        "ticker": TICKER,
        "ultimo_pregao": data_base,
        "ultimo_fechamento": round(ultimo, 2),
        "variacao_prevista_pct": round(variacao * 100, 3),
        "fechamento_previsto": round(previsto, 2),
        "alerta_drift": alerta,
        "drift": drift,
    }


@app.post("/prever")
def prever_com_dados(corpo: Fechamentos):
    """Recebe uma lista de fechamentos e prevê o próximo."""
    if FEATURES != ["retorno"]:
        raise HTTPException(status_code=400, detail="Este modelo usa outras features além do fechamento. Use GET /previsao.")
    precos = np.array(corpo.fechamentos, dtype=float)
    if len(precos) < JANELA + 1:
        raise HTTPException(status_code=422, detail=f"Envie pelo menos {JANELA + 1} fechamentos.")
    if np.any(precos <= 0):
        raise HTTPException(status_code=422, detail="Os fechamentos precisam ser positivos.")

    df = pd.DataFrame({"Close": precos, "CloseBruto": precos})
    variacao, drift, alerta = prever_somente_retorno(df)
    previsto = precos[-1] * (1 + variacao)
    return {
        "ultimo_fechamento": round(float(precos[-1]), 2),
        "variacao_prevista_pct": round(variacao * 100, 3),
        "fechamento_previsto": round(float(previsto), 2),
        "alerta_drift": alerta,
        "drift": drift,
    }


def prever_somente_retorno(df):
    # só precisa da coluna Close; as outras features não são usadas por este modelo
    df = df.assign(High=df["Close"], Low=df["Close"], Volume=1.0)
    return prever(df)


@app.get("/monitoramento")
def monitoramento():
    """Uso da API, último drift e erro real das previsões já feitas."""
    resposta = {"api": metricas.resumo(), "metricas_do_treino": INFO.get("metricas_teste")}
    try:
        df, _ = dados_recentes()
        resposta["desempenho_em_producao"] = desempenho_em_producao(df)
    except HTTPException:
        resposta["desempenho_em_producao"] = {"mensagem": "Yahoo indisponível no momento"}
    return resposta
