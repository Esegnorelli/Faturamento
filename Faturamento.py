from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Tuple
import io
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

"""
Hora do Pastel — Dashboard ‘Perfeito’ (v3)
-----------------------------------------
Como executar:
    streamlit run hora_do_pastel_dashboard_perfeito.py

O app lê, por padrão, um arquivo "Faturamento.csv" na mesma pasta.
Você também pode enviar um CSV pela sidebar (tem prioridade sobre o arquivo local).

Formato esperado (nomes podem variar; o script normaliza):
    Data (mm/aaaa ou dd/mm/aaaa)
    Loja
    Faturamento (R$ opcional)
    Pedidos (inteiro)
    Ticket Médio (opcional)

Principais melhorias da v3:
- Upload direto no app + cache inteligente
- Parser de datas robusto (mm/aaaa ou dd/mm/aaaa)
- KPIs com variações MoM, rankings, alertas, sazonalidade e heatmap
- Exportadores (CSV dos dados filtrados e agregado mensal)
- Formatação BR (R$ e milhares)
- Controles de filtro práticos (Selecionar tudo / Limpar)
"""

# ============================
# CONSTANTES E UTILITÁRIOS
# ============================
APP_TITLE = "🥟 Hora do Pastel — Insights de Faturamento"
APP_CAPTION = (
    "KPIs enxutos, rankings, alertas e sazonalidade para decisões rápidas "
    "nas lojas da rede."
)

MESES = {
    1: "Janeiro", 2: "Fevereiro", 3: "Março", 4: "Abril",
    5: "Maio", 6: "Junho", 7: "Julho", 8: "Agosto",
    9: "Setembro", 10: "Outubro", 11: "Novembro", 12: "Dezembro"
}

COLUNAS_REQUERIDAS = {"Data", "Loja"}
QUEDA_ALERTA_THRESHOLD = -30  # Percentual para alertas de queda
TOP_N = 10  # Número de itens em rankings

# ============================
# CARREGAMENTO
# ============================
@st.cache_data(show_spinner=False)
def _ler_csv_bytes(arquivo_subido: bytes) -> pd.DataFrame:
    head = arquivo_subido[:8192].decode("utf-8", errors="ignore")
    sep = ";" if ";" in head else ","
    return pd.read_csv(io.BytesIO(arquivo_subido), sep=sep, encoding="utf-8")

@st.cache_data(show_spinner=False)
def _ler_csv_caminho(caminho: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(caminho, encoding="utf-8-sig", sep=";")
    except Exception:
        return pd.read_csv(caminho, encoding="utf-8-sig", sep=",")

# ============================
# LIMPEZA E TRANSFORMAÇÕES
# ============================
def _normalizar_colunas(df: pd.DataFrame) -> pd.DataFrame:
    mapa = {}
    for c in df.columns:
        k = str(c).strip().lower()
        if k == "data": mapa[c] = "Data"
        elif k == "loja": mapa[c] = "Loja"
        elif k in ("faturamento", "receita", "valor"): mapa[c] = "Faturamento"
        elif k in ("pedidos", "qtd_pedidos", "qtd"): mapa[c] = "Pedidos"
        elif k in ("ticket médio", "ticket medio", "ticket_medio", "ticket-medio"): mapa[c] = "Ticket Médio"
    return df.rename(columns=mapa)


def _limpar_valor(serie: pd.Series) -> pd.Series:
    s = serie.astype(str)
    s = (s.str.replace("R$", "", regex=False)
           .str.replace(" ", "", regex=False)
           .str.replace(".", "", regex=False)
           .str.replace(",", "."))
    return s


def _parse_data(col: pd.Series) -> pd.Series:
    s = col.astype(str).str.strip()
    m_mmyyyy = s.str.match(r"^\d{1,2}/\d{4}$")
    m_ddmmyyyy = s.str.match(r"^\d{1,2}/\d{1,2}/\d{4}$")

    dt1 = pd.to_datetime(s.where(m_ddmmyyyy), dayfirst=True, errors="coerce")
    dt2 = pd.to_datetime("01/" + s.where(m_mmyyyy), format="%d/%m/%Y", errors="coerce")
    dt3 = pd.to_datetime(s, dayfirst=True, errors="coerce")
    return dt1.fillna(dt2).fillna(dt3)


def processar_dados(df: pd.DataFrame) -> pd.DataFrame:
    if not COLUNAS_REQUERIDAS.issubset(df.columns):
        raise ValueError(f"CSV precisa ter ao menos: {', '.join(COLUNAS_REQUERIDAS)}")

    df = df.copy()
    df["Data"] = _parse_data(df["Data"])  # aceita mm/aaaa e dd/mm/aaaa
    df = df.dropna(subset=["Data"])  # remove datas inválidas

    df["Loja"] = (df["Loja"].astype(str).str.strip()
                                .str.replace(r"\s+", " ", regex=True)
                                .str.title())

    if "Faturamento" in df.columns:
        df["Faturamento"] = pd.to_numeric(_limpar_valor(df["Faturamento"]), errors="coerce").fillna(0.0)
    else:
        df["Faturamento"] = 0.0

    if "Pedidos" in df.columns:
        df["Pedidos"] = pd.to_numeric(_limpar_valor(df["Pedidos"]), errors="coerce").fillna(0).astype(int)
    else:
        df["Pedidos"] = 0

    if "Ticket Médio" in df.columns:
        df["Ticket Médio"] = pd.to_numeric(_limpar_valor(df["Ticket Médio"]), errors="coerce")
    else:
        df["Ticket Médio"] = np.nan

    df["Ano"] = df["Data"].dt.year
    df["Mes"] = df["Data"].dt.month
    df["MesNome"] = df["Mes"].map(MESES)
    df["Mes/Ano"] = df["Data"].dt.to_period("M").dt.to_timestamp()

    return df.sort_values(["Data", "Loja"]).reset_index(drop=True)

# ============================
# CÁLCULOS
# ============================
def _variacao_mom(df: pd.DataFrame, meses: List[pd.Timestamp], coluna: str) -> Optional[float]:
    if len(meses) < 2:
        return None
    a, b = meses[-1], meses[-2]
    va = df[df["Mes/Ano"] == a][coluna].sum()
    vb = df[df["Mes/Ano"] == b][coluna].sum()
    if vb == 0:
        return None
    return (va - vb) / vb * 100


def _variacao_ticket_mom(df: pd.DataFrame, meses: List[pd.Timestamp]) -> Optional[float]:
    if len(meses) < 2:
        return None
    a, b = meses[-1], meses[-2]
    atual = df[df["Mes/Ano"] == a]
    ant = df[df["Mes/Ano"] == b]
    ta = atual["Faturamento"].sum() / atual["Pedidos"].sum() if atual["Pedidos"].sum() > 0 else np.nan
    tb = ant["Faturamento"].sum() / ant["Pedidos"].sum() if ant["Pedidos"].sum() > 0 else np.nan
    if pd.notna(tb) and tb > 0:
        return (ta - tb) / tb * 100
    return None


def _fmt_moeda(x: float) -> str:
    return f"R$ {x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def _fmt_int(x: float) -> str:
    return f"{int(x):,}".replace(",", ".")


def _fmt_pct(x: Optional[float]) -> str:
    return (f"{x:.1f}%".replace(".", ",") if x is not None else "—")


def gerar_alertas(mensal: pd.DataFrame, filtrado: pd.DataFrame, meses: List[pd.Timestamp]) -> list[tuple[str, str]]:
    alertas: list[tuple[str, str]] = []
    if len(meses) >= 2:
        a, b = meses[-1], meses[-2]
        atual = mensal[mensal["Mes/Ano"] == a].set_index("Loja")
        ant = mensal[mensal["Mes/Ano"] == b].set_index("Loja")
        comp = atual[["Faturamento"]].join(ant[["Faturamento"]], lsuffix="_atual", rsuffix="_anterior")
        comp["var_%"] = (comp["Faturamento_atual"] - comp["Faturamento_anterior"]) / comp["Faturamento_anterior"].replace({0: np.nan}) * 100
        comp = comp.dropna().sort_values("var_%")
        for loja, row in comp.iterrows():
            if row["var_%"] <= QUEDA_ALERTA_THRESHOLD:
                alertas.append(("warning", f"⚠️ {loja}: queda de {row['var_%']:.1f}% vs mês anterior."))

    zeradas = filtrado.groupby("Loja")["Faturamento"].sum()
    for loja in zeradas[zeradas == 0].index:
        alertas.append(("error", f"🛑 {loja}: sem faturamento neste recorte."))

    for loja, g in mensal.groupby("Loja"):
        g = g.sort_values("Mes/Ano")
        if len(g) >= 2:
            u = g.iloc[-1]["Faturamento"]
            if u == g["Faturamento"].max() and u > 0:
                alertas.append(("success", f"✅ {loja}: maior faturamento mensal do histórico no recorte."))

    return alertas

# ============================
# PÁGINA
# ============================
st.set_page_config(page_title=APP_TITLE, page_icon="🥟", layout="wide")
st.title(APP_TITLE)
st.caption(APP_CAPTION)

with st.sidebar:
    st.header("📥 Fonte de dados")
    up = st.file_uploader("Envie um CSV (opcional)", type=["csv"])
    st.markdown("— ou —")
    st.code("Faturamento.csv", language="bash")

# Carrega dados (upload tem prioridade)
try:
    if up is not None:
        _raw = _ler_csv_bytes(up.read())
    else:
        caminho = (Path(__file__).parent / "Faturamento.csv") if "__file__" in globals() else Path("Faturamento.csv")
        _raw = _ler_csv_caminho(caminho)

    _raw = _normalizar_colunas(_raw)
    dados = processar_dados(_raw)
except FileNotFoundError as e:
    st.error(f"❌ {e}")
    st.stop()
except ValueError as e:
    st.error(f"❌ {e}")
    st.stop()
except Exception as e:
    st.error(f"❌ Erro ao processar dados: {e}")
    st.stop()

# ============================
# FILTROS
# ============================
st.sidebar.header("🎚️ Filtros")
anos = sorted(dados["Ano"].unique().tolist())
meses = sorted(dados["Mes"].unique().tolist())
lojas = sorted(dados["Loja"].dropna().unique().tolist())

c1, c2 = st.sidebar.columns(2)
with c1:
    all_years = st.checkbox("Selecionar todos os anos", value=True)
with c2:
    all_stores = st.checkbox("Selecionar todas as lojas", value=True)

anos_sel = st.sidebar.multiselect("Ano", options=anos, default=anos if all_years else [])
mes_label = ["Todos"] + [f"{m:02d} - {MESES[m]}" for m in meses]
mes_sel = st.sidebar.selectbox("Mês", mes_label, index=0)
lojas_sel = st.sidebar.multiselect("Lojas", options=lojas, default=lojas if all_stores else [])

f = dados[dados["Ano"].isin(anos_sel)] if anos_sel else dados.copy()
if mes_sel != "Todos":
    mes_num = int(mes_sel.split(" - ")[0])
    f = f[f["Mes"] == mes_num]
if lojas_sel:
    f = f[f["Loja"].isin(lojas_sel)]

if f.empty:
    st.warning("⚠️ Sem dados para este recorte. Ajuste os filtros.")
    st.stop()

# ============================
# AGREGAÇÕES
# ============================
mensal = (
    f.groupby(["Loja", "Mes/Ano"], as_index=False)
     .agg({"Faturamento": "sum", "Pedidos": "sum"})
     .sort_values(["Loja", "Mes/Ano"]) 
)
mensal["Ticket Médio"] = mensal["Faturamento"] / mensal["Pedidos"].replace(0, np.nan)

fat_total = float(f["Faturamento"].sum())
ped_total = int(f["Pedidos"].sum())
ticket_geral = fat_total / ped_total if ped_total > 0 else 0.0

meses_unicos = sorted(mensal["Mes/Ano"].unique())
var_fat = _variacao_mom(mensal, meses_unicos, "Faturamento")
var_ped = _variacao_mom(mensal, meses_unicos, "Pedidos")
var_ticket = _variacao_ticket_mom(mensal, meses_unicos)

# ============================
# KPIs
# ============================
col1, col2, col3, col4 = st.columns(4)
with col1:
    lojas_ativas = mensal[mensal["Mes/Ano"] == meses_unicos[-1]]["Loja"].nunique() if meses_unicos else 0
    st.metric("🏪 Lojas ativas no recorte", lojas_ativas)
with col2:
    st.metric("💸 Faturamento (R$)", _fmt_moeda(fat_total), _fmt_pct(var_fat) if var_fat is not None else None)
with col3:
    st.metric("🧾 Pedidos", _fmt_int(ped_total), _fmt_pct(var_ped) if var_ped is not None else None)
with col4:
    st.metric("🎟️ Ticket médio (R$)", _fmt_moeda(ticket_geral), _fmt_pct(var_ticket) if var_ticket is not None else None)

st.caption("ℹ️ Deltas (%) comparam o mês atual com o imediatamente anterior do mesmo recorte.")

# ============================
# ABAS
# ============================
t1, t2, t3, t4, t5, t6 = st.tabs([
    "📊 Gráficos",
    "🏆 Rankings",
    "🚨 Alertas",
    "🗓️ Sazonalidade",
    "📋 Tabela",
    "⬇️ Exportar"
])

# --- GRÁFICOS ---
with t1:
    a, b = st.columns(2)
    with a:
        fig_fat = px.line(mensal, x="Mes/Ano", y="Faturamento", color="Loja", markers=True,
                          title="Faturamento mensal por loja")
        fig_fat.update_layout(hovermode="x unified", yaxis_tickprefix="R$ ")
        st.plotly_chart(fig_fat, use_container_width=True)
    with b:
        fig_ped = px.line(mensal, x="Mes/Ano", y="Pedidos", color="Loja", markers=True,
                          title="Pedidos mensais por loja")
        fig_ped.update_layout(hovermode="x unified")
        st.plotly_chart(fig_ped, use_container_width=True)

    fig_tk = px.line(mensal, x="Mes/Ano", y="Ticket Médio", color="Loja", markers=True,
                     title="Ticket médio mensal (R$)")
    fig_tk.update_layout(yaxis_tickprefix="R$ ")
    st.plotly_chart(fig_tk, use_container_width=True)

# --- RANKINGS ---
with t2:
    st.subheader("Top 10 — Faturamento no recorte")
    top_fat = (
        f.groupby("Loja")["Faturamento"].sum().sort_values(ascending=False).head(TOP_N).reset_index()
    )
    if not top_fat.empty:
        fig_top = px.bar(top_fat, x="Loja", y="Faturamento", title=f"Top {TOP_N} por Faturamento (R$)")
        fig_top.update_layout(yaxis_tickprefix="R$ ")
        st.plotly_chart(fig_top, use_container_width=True)

    st.subheader("Maiores crescimentos (MoM)")
    if len(meses_unicos) >= 2:
        a, b = meses_unicos[-1], meses_unicos[-2]
        cur = mensal[mensal["Mes/Ano"] == a].set_index("Loja")
        prv = mensal[mensal["Mes/Ano"] == b].set_index("Loja")
        comp = cur[["Faturamento"]].join(prv[["Faturamento"]], lsuffix="_atual", rsuffix="_anterior")
        comp["MoM (%)"] = (comp["Faturamento_atual"] - comp["Faturamento_anterior"]) / comp["Faturamento_anterior"].replace({0: np.nan}) * 100
        comp = comp.dropna(subset=["MoM (%)"]).sort_values("MoM (%)", ascending=False).head(TOP_N).reset_index()
        fig_c = px.bar(comp, x="Loja", y="MoM (%)", title=f"Top {TOP_N} Crescimentos MoM (%)")
        st.plotly_chart(fig_c, use_container_width=True)
    else:
        st.info("Precisa de pelo menos 2 meses para comparar.")

# --- ALERTAS ---
with t3:
    st.subheader("Alertas do período")
    al = gerar_alertas(mensal, f, meses_unicos)

    if len(meses_unicos) >= 2:
        a, b = meses_unicos[-1], meses_unicos[-2]
        cur = mensal[mensal["Mes/Ano"] == a].set_index("Loja")
        prv = mensal[mensal["Mes/Ano"] == b].set_index("Loja")
        comp = cur[["Faturamento"]].join(prv[["Faturamento"]], lsuffix="_atual", rsuffix="_anterior")
        comp["var_%"] = (comp["Faturamento_atual"] - comp["Faturamento_anterior"]) / comp["Faturamento_anterior"].replace({0: np.nan}) * 100
        quedas = comp.dropna().sort_values("var_%").head(TOP_N).reset_index()
        fig_q = px.bar(quedas, x="Loja", y="var_%", title="Maiores quedas (MoM %)")
        st.plotly_chart(fig_q, use_container_width=True)

    if al:
        for tipo, msg in al:
            if tipo == "success":
                st.success(msg)
            elif tipo == "error":
                st.error(msg)
            else:
                st.warning(msg)
    else:
        st.success("✅ Nenhum alerta relevante encontrado.")

# --- SAZONALIDADE ---
with t4:
    st.subheader("Faturamento por mês do ano (histórico)")
    saz = dados.groupby(dados["Data"].dt.month)["Faturamento"].sum().reset_index()
    saz.columns = ["Mes", "Faturamento"]
    saz["Mês"] = saz["Mes"].map(lambda m: f"{m:02d} - {MESES[m]}")
    fig_s = px.bar(saz, x="Mês", y="Faturamento", title="Sazonalidade — Histórico")
    fig_s.update_layout(yaxis_tickprefix="R$ ")
    st.plotly_chart(fig_s, use_container_width=True)

    st.subheader("Heatmap — Loja x Mês/Ano (recorte)")
    pv = mensal.pivot_table(index="Loja", columns="Mes/Ano", values="Faturamento", aggfunc="sum", fill_value=0.0)
    if not pv.empty:
        fig_h = px.imshow(pv, aspect="auto", labels=dict(x="Mês/Ano", y="Loja", color="R$"), title="Heatmap de Faturamento")
        st.plotly_chart(fig_h, use_container_width=True)

# --- TABELAS ---
with t5:
    st.subheader("Transacional do recorte")
    cols = [c for c in ["Data", "Loja", "Faturamento", "Pedidos", "Ticket Médio", "Ano", "Mes", "MesNome", "Mes/Ano"] if c in f.columns]
    st.dataframe(f[cols], use_container_width=True, hide_index=True)

    st.subheader("Agregado mensal por loja (recorte)")
    st.dataframe(mensal, use_container_width=True, hide_index=True)

# --- EXPORTAR ---
with t6:
    st.subheader("Exportar CSVs do recorte atual")

    buf_f = io.StringIO()
    f.to_csv(buf_f, index=False)
    st.download_button("⬇️ Baixar transacional filtrado (CSV)", buf_f.getvalue(), file_name="hora_do_pastel_transacional.csv", mime="text/csv")

    buf_m = io.StringIO()
    mensal.to_csv(buf_m, index=False)
    st.download_button("⬇️ Baixar agregado mensal por loja (CSV)", buf_m.getvalue(), file_name="hora_do_pastel_agregado_mensal.csv", mime="text/csv")

# ============================
# RODAPÉ
# ============================
st.markdown(
    """
---
### ℹ️ Notas
- **Objetivo**: visão clara e mínima do desempenho por loja, destacando *o que mudou* (MoM) e *onde agir* (quedas, recordes e sazonalidade).
- **MoM (%)**: `(mês atual − mês anterior) / mês anterior × 100`.
- **Ticket médio (geral)**: `Faturamento total / Pedidos totais` no recorte.
- **Dica**: use a aba *Exportar* para compartilhar rapidamente os dados filtrados.
"""
)
