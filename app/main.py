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

from app.dados import (baixar_direto, baixar_mercado, limpar, preparar_features,
                       remover_pregao_em_andamento, verificar_drift)
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
            df = remover_pregao_em_andamento(limpar(baixar_direto(TICKER, inicio)))
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


MIN_PREGOES = INFO.get("dias_brutos_necessarios", JANELA + 21)


class Pregao(BaseModel):
    fechamento: float = Field(..., gt=0)
    maxima: float = Field(..., gt=0)
    minima: float = Field(..., gt=0)
    volume: float = Field(..., gt=0)


class Pregoes(BaseModel):
    pregoes: list[Pregao] = Field(
        ...,
        description=f"Pregões diários em ordem cronológica, do mais antigo ao mais recente (mínimo {MIN_PREGOES}).",
        examples=[[
            {"fechamento": round(60 + i * 0.05, 2), "maxima": round(60.6 + i * 0.05, 2),
             "minima": round(59.4 + i * 0.05, 2), "volume": 25_000_000 + (i % 7) * 1_000_000}
            for i in range(MIN_PREGOES)
        ]],
    )


@app.api_route("/", methods=["GET", "HEAD"])
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
def prever_com_dados(corpo: Pregoes):
    """Recebe os últimos pregões (fechamento, máxima, mínima e volume) e prevê o próximo fechamento."""
    if MERCADO:
        raise HTTPException(status_code=400, detail="Este modelo usa dados de mercado. Use GET /previsao.")
    if len(corpo.pregoes) < MIN_PREGOES:
        raise HTTPException(status_code=422, detail=f"Envie pelo menos {MIN_PREGOES} pregões.")
    if any(p.minima > p.maxima for p in corpo.pregoes):
        raise HTTPException(status_code=422, detail="A mínima não pode ser maior que a máxima.")

    df = pd.DataFrame({
        "Close": [p.fechamento for p in corpo.pregoes],
        "CloseBruto": [p.fechamento for p in corpo.pregoes],
        "High": [p.maxima for p in corpo.pregoes],
        "Low": [p.minima for p in corpo.pregoes],
        "Volume": [p.volume for p in corpo.pregoes],
    })
    variacao, drift, alerta = prever(df)
    ultimo = float(df["CloseBruto"].iloc[-1])
    return {
        "ultimo_fechamento": round(ultimo, 2),
        "variacao_prevista_pct": round(variacao * 100, 3),
        "fechamento_previsto": round(ultimo * (1 + variacao), 2),
        "alerta_drift": alerta,
        "drift": drift,
    }


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
