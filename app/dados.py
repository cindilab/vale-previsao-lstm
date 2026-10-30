import time
from datetime import datetime
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests

URL_YAHOO = "https://query1.finance.yahoo.com/v8/finance/chart/{ticker}"


def baixar_direto(ticker, inicio):
    params = {
        "period1": int(pd.Timestamp(inicio).timestamp()),
        "period2": int(time.time()),
        "interval": "1d",
        "events": "history",
    }
    r = requests.get(URL_YAHOO.format(ticker=ticker), params=params,
                     headers={"User-Agent": "Mozilla/5.0"}, timeout=30)
    r.raise_for_status()
    res = r.json()["chart"]["result"][0]
    ind = res["indicators"]
    cotacao = ind["quote"][0]
    ajustado = ind["adjclose"][0]["adjclose"] if "adjclose" in ind else cotacao["close"]
    datas = pd.to_datetime(res["timestamp"], unit="s").normalize()
    return pd.DataFrame(
        {
            "Close": ajustado,
            "High": cotacao["high"],
            "Low": cotacao["low"],
            "CloseBruto": cotacao["close"],
            "Volume": cotacao["volume"],
        },
        index=datas,
    )


def limpar(df):
    df = df.dropna()
    df = df[~df.index.duplicated(keep="last")]
    return df[df["Volume"] > 0]


def remover_pregao_em_andamento(df, hora_fechamento=18, minuto_fechamento=30):
    # durante o pregão o Yahoo devolve o preço do momento como se fosse o fechamento do dia
    agora = datetime.now(ZoneInfo("America/Sao_Paulo"))
    fechou = (agora.hour, agora.minute) >= (hora_fechamento, minuto_fechamento)
    if len(df) and df.index[-1].date() == agora.date() and not fechou:
        return df.iloc[:-1]
    return df


def baixar_mercado(tickers, inicio, datas_referencia):
    series = {}
    for nome, ticker in tickers.items():
        serie = baixar_direto(ticker, inicio)["Close"].dropna()
        serie = serie[~serie.index.duplicated(keep="last")]
        serie = serie.reindex(serie.index.union(datas_referencia)).ffill().reindex(datas_referencia)
        series[nome] = serie
    return pd.DataFrame(series, index=datas_referencia)


# mesma função do notebook
def preparar_features(df, mercado=None):
    f = pd.DataFrame(index=df.index)
    f["retorno"] = df["Close"].pct_change()
    f["amplitude"] = (df["High"] - df["Low"]) / df["CloseBruto"]
    f["volume_rel"] = np.log(df["Volume"] / df["Volume"].rolling(20).mean())
    f["volatilidade_20"] = f["retorno"].rolling(20).std()
    f["dist_media_20"] = df["Close"] / df["Close"].rolling(20).mean() - 1
    if mercado is not None:
        for nome in mercado.columns:
            f[f"retorno_{nome}"] = mercado[nome].pct_change()
    return f.replace([np.inf, -np.inf], np.nan)


def verificar_drift(features_recentes, referencia, limite):
    resultado = {}
    for f, ref in referencia.items():
        distancia = (features_recentes[f].mean() - ref["media"]) / ref["desvio"]
        resultado[f] = {"distancia": round(float(distancia), 3), "alerta": bool(abs(distancia) > limite)}
    return resultado
