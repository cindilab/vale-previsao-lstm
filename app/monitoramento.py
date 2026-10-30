import csv
import logging
import time
from pathlib import Path
from threading import Lock

import pandas as pd

ARQUIVO_PREVISOES = Path(__file__).resolve().parent.parent / "logs" / "previsoes.csv"
CAMPOS = ["registrado_em", "data_base", "fechamento_base", "variacao_prevista", "fechamento_previsto", "alerta_drift"]

logger = logging.getLogger("vale-api")


class Metricas:
    """Contadores simples de uso da API (ficam em memória)."""

    def __init__(self):
        self.inicio = time.time()
        self.lock = Lock()
        self.requisicoes = 0
        self.erros = 0
        self.tempos_ms = []
        self.por_rota = {}
        self.ultimo_drift = None

    def registrar(self, rota, status, tempo_ms):
        with self.lock:
            self.requisicoes += 1
            if status >= 400:
                self.erros += 1
            self.tempos_ms = (self.tempos_ms + [tempo_ms])[-1000:]
            self.por_rota[rota] = self.por_rota.get(rota, 0) + 1

    def resumo(self):
        with self.lock:
            tempos = sorted(self.tempos_ms)
            return {
                "no_ar_ha_minutos": round((time.time() - self.inicio) / 60, 1),
                "requisicoes": self.requisicoes,
                "erros": self.erros,
                "tempo_medio_ms": round(sum(tempos) / len(tempos), 1) if tempos else None,
                "tempo_p95_ms": round(tempos[int(len(tempos) * 0.95) - 1], 1) if tempos else None,
                "requisicoes_por_rota": dict(self.por_rota),
                "ultimo_drift": self.ultimo_drift,
            }


def salvar_previsao(linha):
    ARQUIVO_PREVISOES.parent.mkdir(parents=True, exist_ok=True)
    novo = not ARQUIVO_PREVISOES.exists()
    with open(ARQUIVO_PREVISOES, "a", newline="", encoding="utf-8") as f:
        escritor = csv.DictWriter(f, fieldnames=CAMPOS)
        if novo:
            escritor.writeheader()
        escritor.writerow(linha)


def desempenho_em_producao(dados_recentes):
    """Compara as previsões já feitas com o fechamento que de fato aconteceu."""
    if not ARQUIVO_PREVISOES.exists():
        return {"avaliadas": 0, "mensagem": "nenhuma previsão registrada ainda"}

    prev = pd.read_csv(ARQUIVO_PREVISOES, parse_dates=["data_base"])
    prev = prev.drop_duplicates("data_base", keep="last")

    datas = dados_recentes.index
    linhas = []
    for _, p in prev.iterrows():
        if p["data_base"] < datas[0]:
            continue  # previsão antiga, fora do período baixado
        seguintes = datas[datas > p["data_base"]]
        if len(seguintes) == 0:
            continue  # ainda não aconteceu o próximo pregão
        real = float(dados_recentes.loc[seguintes[0], "CloseBruto"])
        linhas.append({
            "data_base": p["data_base"].date().isoformat(),
            "data_alvo": seguintes[0].date().isoformat(),
            "previsto": round(float(p["fechamento_previsto"]), 2),
            "real": round(real, 2),
            "erro_modelo": abs(real - p["fechamento_previsto"]),
            "erro_baseline": abs(real - p["fechamento_base"]),
        })

    if not linhas:
        return {"avaliadas": 0, "mensagem": "aguardando o próximo pregão para comparar"}

    tab = pd.DataFrame(linhas)
    return {
        "avaliadas": len(tab),
        "mae_modelo": round(tab["erro_modelo"].mean(), 4),
        "mae_baseline": round(tab["erro_baseline"].mean(), 4),
        "ultimas": tab.tail(10).round(4).to_dict(orient="records"),
    }
