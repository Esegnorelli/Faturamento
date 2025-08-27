# app.py — Dashboard Inteligente (Streamlit)
#
# Requisitos: streamlit, pandas, plotly, python-dateutil
# Execução: streamlit run app.py
#
# • 1 página, sem usar st.markdown/HTML
# • Período via intervalo contínuo (AAAA‑MM) com select_slider + Lojas por checkboxes
# • KPIs nativos (st.metric) com deltas MoM
# • Gráficos coloridos e limpos: Faturamento, Pedidos, Ticket médio
# • "Insights automáticos": MoM/YoY, Top Movimentos, YTD, Participação e Eficiência (com explicações em texto simples)

import os
import re
import unicodedata
from dateutil.relativedelta import relativedelta

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# -----------------------------------------------------------------------------
# CONFIG BÁSICA
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Dashboard Hora do Pastel — KPIs Inteligentes",
    page_icon="🥟",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Plotly mais colorido e consistente
# Utilize uma paleta pastel e um template claro para um visual mais moderno e leve.
px.defaults.template = "simple_white"
px.defaults.color_discrete_sequence = px.colors.qualitative.Pastel

# -----------------------------------------------------------------------------
# HELPERS DE TRATAMENTO E FORMATAÇÃO
# -----------------------------------------------------------------------------

def normalize_col(name: str) -> str:
    name = name.strip().lower()
    name = "".join(c for c in unicodedata.normalize("NFKD", name) if not unicodedata.combining(c))
    return re.sub(r"\s+", "_", name)


def br_to_float(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    s = s.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    s = s.str.replace(r"[^0-9,.\-]", "", regex=True)
    has_comma = s.str.contains(",", na=False)
    s = s.mask(has_comma, s.str.replace(".", "", regex=False))
    s = s.mask(has_comma, s.str.replace(",", ".", regex=False))
    return pd.to_numeric(s, errors="coerce")


def month_to_int(series: pd.Series) -> pd.Series:
    mapa = {
        "jan":1,"janeiro":1,"fev":2,"fevereiro":2,"mar":3,"marco":3,"março":3,
        "abr":4,"abril":4,"mai":5,"maio":5,"jun":6,"junho":6,"jul":7,"julho":7,
        "ago":8,"agosto":8,"set":9,"setembro":9,"sep":9,"out":10,"outubro":10,
        "nov":11,"novembro":11,"dez":12,"dezembro":12
    }
    s = series.astype(str).str.strip().str.lower().map(lambda x: mapa.get(x, x))
    return pd.to_numeric(s, errors="coerce").astype("Int64")

ALIASES = {
    "mes":["mes","mês","month"],
    "ano":["ano","year"],
    "loja":["loja","filial","store"],
    "faturamento":["faturamento","receita","vendas","valor","total","valor_total"],
    "pedidos":["pedidos","qtde_pedidos","qtd_pedidos","qtd","quantidade_pedidos"],
    "ticket":["ticket","ticket_medio","ticket_médio","ticket medio","ticket médio"],
}

def rename_by_alias(cols: list[str]) -> dict:
    ren = {}
    for c in cols:
        for target, opts in ALIASES.items():
            if c in opts:
                ren[c] = target
                break
    return ren


def safe_div(a, b):
    try:
        return (a / b) if b not in (0, None) and not pd.isna(b) else 0.0
    except Exception:
        return 0.0


def fmt_brl(v) -> str:
    if pd.isna(v):
        return "R$ 0,00"
    s = f"{float(v):,.2f}"
    return "R$ " + s.replace(",","X").replace(".",",").replace("X", ".")


def fmt_int(v) -> str:
    try:
        return f"{int(v):,}".replace(",", ".")
    except Exception:
        return "0"

# -----------------------------------------------------------------------------
# CARGA E TRATAMENTO DE DADOS
# -----------------------------------------------------------------------------

@st.cache_data(ttl=3600, max_entries=8, show_spinner=False)
def load_data() -> pd.DataFrame:
    if os.path.exists("Faturamento_tratado.csv"):
        df = pd.read_csv("Faturamento_tratado.csv")
        for c in ["mes","ano","pedidos"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
        for c in ["faturamento","ticket"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        if "loja" in df.columns:
            df["loja"] = df["loja"].astype(str).str.strip()
    else:
        df = pd.read_csv("Faturamento.csv", sep=None, engine="python")
        df.columns = [normalize_col(c) for c in df.columns]
        df = df.loc[:, ~df.columns.duplicated()].dropna(axis=1, how="all")
        df = df.rename(columns=rename_by_alias(list(df.columns)))
        for col in ["mes","ano","loja","faturamento","pedidos","ticket"]:
            if col not in df.columns:
                df[col] = pd.NA
        df["mes"] = month_to_int(df["mes"])
        df["ano"] = pd.to_numeric(df["ano"], errors="coerce").astype("Int64")
        df["loja"] = df["loja"].astype(str).str.strip()
        df["faturamento"] = br_to_float(df["faturamento"])
        df["ticket"] = br_to_float(df["ticket"])
        df["pedidos"] = pd.to_numeric(df["pedidos"], errors="coerce").round().astype("Int64")

    mask = df["ano"].notna() & df["mes"].notna()
    df["data"] = pd.NaT
    df.loc[mask, "data"] = pd.to_datetime(
        {"year": df.loc[mask, "ano"].astype(int), "month": df.loc[mask, "mes"].astype(int), "day": 1},
        errors="coerce",
    )
    df["periodo"] = df["data"].dt.to_period("M").astype(str)
    return df


df = load_data()

# -----------------------------------------------------------------------------
# UI: FILTROS (Período por intervalo + Lojas por checkboxes)
# -----------------------------------------------------------------------------

st.sidebar.header("Filtros")

# Períodos (AAAA‑MM) como intervalo contínuo
periodos = sorted(p for p in df["periodo"].dropna().unique().tolist())
if len(periodos) == 0:
    st.stop()

rng_default = (periodos[0], periodos[-1])
periodo_ini, periodo_fim = st.sidebar.select_slider(
    "Período (AAAA‑MM)",
    options=periodos,
    value=rng_default,
)

# Lojas (checkboxes com "Selecionar todas")
lojas = sorted(df["loja"].dropna().unique().tolist())
st.sidebar.write("Lojas (marque as desejadas):")
all_l = st.sidebar.checkbox("Selecionar todas as lojas", value=True, key="all_l")
sel_lojas = lojas if all_l else [l for l in lojas if st.sidebar.checkbox(l, value=False, key=f"l_{l}")]
if not sel_lojas:
    sel_lojas = lojas  # fallback

# Aplica filtros
mask = (df["periodo"] >= periodo_ini) & (df["periodo"] <= periodo_fim) & (df["loja"].isin(sel_lojas))
df_f = df.loc[mask].copy()
# Para comparações globais (MoM/YoY, YTD, etc.), considera todas as datas das lojas selecionadas
df_lojas = df[df["loja"].isin(sel_lojas)].copy()

# -----------------------------------------------------------------------------
# KPI ENGINE
# -----------------------------------------------------------------------------

@st.cache_data(show_spinner=False)
def month_series(df_: pd.DataFrame) -> pd.DataFrame:
    if df_.empty or df_["data"].isna().all():
        return pd.DataFrame(columns=["data","faturamento","pedidos","ticket_medio"])
    s = (df_.dropna(subset=["data"]).groupby("data", as_index=False)
          .agg(faturamento=("faturamento","sum"), pedidos=("pedidos","sum")))
    s["ticket_medio"] = s.apply(lambda r: safe_div(r["faturamento"], r["pedidos"]), axis=1)
    return s.sort_values("data")


def get_current_prev_yoy(series: pd.DataFrame):
    if series.empty:
        return None, None, None
    cur = series["data"].max()
    prev_candidate = cur - relativedelta(months=1)
    prev = series.loc[series["data"] == prev_candidate, "data"]
    if prev.empty and len(series) >= 2:
        prev = pd.Series(series.sort_values("data")["data"].iloc[-2])
    prev = prev.iloc[0] if not prev.empty else None
    yoy_candidate = cur - relativedelta(years=1)
    yoy = series.loc[series["data"] == yoy_candidate, "data"]
    yoy = yoy.iloc[0] if not yoy.empty else None
    return cur, prev, yoy


def compute_kpis(df_range: pd.DataFrame, df_comp: pd.DataFrame):
    serie_all = month_series(df_comp)
    cur, prev, yoy = get_current_prev_yoy(serie_all)

    tot_fat = float(df_range["faturamento"].sum())
    tot_ped = int(df_range["pedidos"].sum()) if df_range["pedidos"].notna().any() else 0
    tik_med = safe_div(tot_fat, tot_ped)

    fat_cur = ped_cur = tik_cur = None
    fat_prev = ped_prev = tik_prev = None
    fat_yoy = ped_yoy = tik_yoy = None

    if cur is not None:
        r = serie_all.loc[serie_all["data"] == cur].iloc[0]
        fat_cur, ped_cur, tik_cur = r["faturamento"], r["pedidos"], r["ticket_medio"]
    if prev is not None:
        r = serie_all.loc[serie_all["data"] == prev].iloc[0]
        fat_prev, ped_prev, tik_prev = r["faturamento"], r["pedidos"], r["ticket_medio"]
    if yoy is not None:
        r = serie_all.loc[serie_all["data"] == yoy].iloc[0]
        fat_yoy, ped_yoy, tik_yoy = r["faturamento"], r["pedidos"], r["ticket_medio"]

    def delta(cur_v, base_v):
        if cur_v is None or base_v in (None, 0) or pd.isna(base_v):
            return None
        return safe_div((cur_v - base_v), base_v)

    return {
        "period_sum": {"fat": tot_fat, "ped": tot_ped, "tik": tik_med},
        "current": cur, "prev": prev, "yoy_m": yoy,
        "fat_cur": fat_cur, "ped_cur": ped_cur, "tik_cur": tik_cur,
        "mom_fat": delta(fat_cur, fat_prev),
        "mom_ped": delta(ped_cur, ped_prev),
        "mom_tik": delta(tik_cur, tik_prev),
        "yoy_fat": delta(fat_cur, fat_yoy),
        "yoy_ped": delta(ped_cur, ped_yoy),
        "yoy_tik": delta(tik_cur, tik_yoy),
    }


def per_store_current_vs_prev(df_comp: pd.DataFrame, cur, prev):
    if cur is None:
        return pd.DataFrame(columns=["loja","fat_cur","fat_prev","participacao","mom"])
    cur_t = (df_comp.loc[df_comp["data"]==cur].groupby("loja", as_index=False)["faturamento"].sum()
             .rename(columns={"faturamento":"fat_cur"}))
    total_cur = cur_t["fat_cur"].sum() if not cur_t.empty else 0
    prev_t = (df_comp.loc[df_comp["data"]==prev].groupby("loja", as_index=False)["faturamento"].sum()
              .rename(columns={"faturamento":"fat_prev"})) if prev is not None else pd.DataFrame(columns=["loja","fat_prev"])
    out = cur_t.merge(prev_t, on="loja", how="left")
    out["fat_prev"] = out["fat_prev"].fillna(0.0)
    out["participacao"] = out["fat_cur"].apply(lambda v: safe_div(v, total_cur))
    out["mom"] = out.apply(lambda r: safe_div((r["fat_cur"]-r["fat_prev"]), r["fat_prev"]) if r["fat_prev"] else None, axis=1)
    return out.sort_values("fat_cur", ascending=False)

k = compute_kpis(df_f, df_lojas)

# -----------------------------------------------------------------------------
# CABEÇALHO E KPIs (APENAS COMPONENTES NATIVOS)
# -----------------------------------------------------------------------------

st.title("Dashboard Inteligente — Hora do Pastel")
st.write(
    f"Período: {periodo_ini} → {periodo_fim}  | "
    f"Lojas selecionadas: {len(sel_lojas)}"
)

m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric(
        label="Faturamento no período",
        value=fmt_brl(k["period_sum"]["fat"]),
        delta=(f"{k['mom_fat']*100:.1f}%".replace(".",",") if k["mom_fat"] is not None else None),
        delta_color="normal",
    )
with m2:
    st.metric(
        label="Pedidos no período",
        value=fmt_int(k["period_sum"]["ped"]),
        delta=(f"{k['mom_ped']*100:.1f}%".replace(".",",") if k["mom_ped"] is not None else None),
        delta_color="normal",
    )
with m3:
    st.metric(
        label=f"Ticket médio {'(' + k['current'].strftime('%b/%Y') + ')' if k['current'] else ''}",
        value=fmt_brl(k["tik_cur"] if k["tik_cur"] is not None else k["period_sum"]["tik"]),
        delta=(f"{k['mom_tik']*100:.1f}%".replace(".",",") if k["mom_tik"] is not None else None),
        delta_color="normal",
    )
with m4:
    st.metric(
        label=f"Faturamento {'(' + k['current'].strftime('%b/%Y') + ')' if k['current'] else ''}",
        value=fmt_brl(k["fat_cur"] if k["fat_cur"] is not None else 0),
        delta=(f"{k['mom_fat']*100:.1f}%".replace(".",",") if k["mom_fat"] is not None else None),
        delta_color="normal",
    )

# -----------------------------------------------------------------------------
# GRÁFICOS PRINCIPAIS (Faturamento, Pedidos, Ticket)
# -----------------------------------------------------------------------------

st.subheader("Evolução do faturamento")
serie_f = (df_f.dropna(subset=["data"]).groupby("data", as_index=False)["faturamento"].sum().sort_values("data"))
if not serie_f.empty:
    fig_f = px.line(serie_f, x="data", y="faturamento", markers=True)
    fig_f.update_traces(line=dict(width=3))
    fig_f.update_layout(
        height=320,
        xaxis_title="Data",
        yaxis_title="Faturamento (R$)"
    )
    st.plotly_chart(fig_f, use_container_width=True)
else:
    st.info("Sem dados no filtro atual.")

st.subheader("Evolução de pedidos")
serie_p = (df_f.dropna(subset=["data"]).groupby("data", as_index=False)["pedidos"].sum().sort_values("data"))
if not serie_p.empty:
    fig_p = px.line(serie_p, x="data", y="pedidos", markers=True)
    fig_p.update_traces(line=dict(width=3))
    fig_p.update_layout(
        height=320,
        xaxis_title="Data",
        yaxis_title="Pedidos"
    )
    st.plotly_chart(fig_p, use_container_width=True)
else:
    st.info("Sem dados de pedidos no filtro atual.")

st.subheader("Ticket médio (mensal)")
serie_t = (df_f.dropna(subset=["data"]).groupby("data", as_index=False)
           .agg(faturamento=("faturamento","sum"), pedidos=("pedidos","sum")))
if not serie_t.empty:
    serie_t["ticket_medio"] = serie_t.apply(lambda r: safe_div(r["faturamento"], r["pedidos"]), axis=1)
    fig_t = px.line(serie_t, x="data", y="ticket_medio", markers=True)
    fig_t.update_traces(line=dict(width=3))
    fig_t.update_layout(
        height=320,
        xaxis_title="Data",
        yaxis_title="Ticket médio (R$)"
    )
    st.plotly_chart(fig_t, use_container_width=True)
else:
    st.info("Sem dados para calcular ticket médio.")

# -----------------------------------------------------------------------------
# INSIGHTS AUTOMÁTICOS (Gráficos e explicações simples)
# -----------------------------------------------------------------------------

st.subheader("Insights automáticos")

# 1) Variações MoM e YoY de Faturamento
serie_all = month_series(df_lojas)
col_a, col_b = st.columns([2, 1])
with col_a:
    if not serie_all.empty:
        serie_all = serie_all.sort_values("data")
        serie_all["mom"] = serie_all["faturamento"].pct_change()
        serie_all["yoy"] = serie_all["faturamento"].pct_change(12)
        fig_var = go.Figure()
        fig_var.add_trace(go.Scatter(
            x=serie_all["data"],
            y=100*serie_all["mom"],
            name="MoM %",
            mode="lines+markers",
            line=dict(width=2)
        ))
        fig_var.add_trace(go.Scatter(
            x=serie_all["data"],
            y=100*serie_all["yoy"],
            name="YoY %",
            mode="lines+markers",
            line=dict(width=2, dash='dot')
        ))
        fig_var.update_layout(
            height=330,
            xaxis_title="Data",
            yaxis_title="Variação (%)"
        )
        st.plotly_chart(fig_var, use_container_width=True)
    else:
        st.info("Sem dados para variações MoM/YoY.")
with col_b:
    st.write("Como calculamos:")
    st.write("MoM = (Faturamento do mês atual − Faturamento do mês anterior) ÷ Faturamento do mês anterior")
    st.write("YoY = (Faturamento do mês atual − Faturamento do mesmo mês do ano anterior) ÷ Faturamento do mesmo mês do ano anterior")
    st.write("MoM destaca tendência recente. YoY reduz o efeito de sazonalidade mensal.")

# 2) Top Movimentos (maiores crescimentos MoM por loja no mês atual)
col_c, col_d = st.columns([2, 1])
with col_c:
    if k["current"] is not None:
        tbl = per_store_current_vs_prev(df_lojas, k["current"], k["prev"])
        if not tbl.empty:
            tbl = tbl.assign(mom_pct=100*tbl["mom"]).sort_values("mom_pct", ascending=False)
            top_up = tbl.head(10)
            fig_up = px.bar(top_up, x="mom_pct", y="loja", orientation="h")
            fig_up.update_layout(
                height=330,
                xaxis_title="MoM (%)",
                yaxis_title="Loja"
            )
            st.plotly_chart(fig_up, use_container_width=True)
        else:
            st.info("Sem base para comparar mês atual vs anterior.")
    else:
        st.info("Mês atual indisponível para comparação.")
with col_d:
    st.write("O que mostra:")
    st.write("Lojas com maior crescimento percentual de faturamento no mês atual em relação ao mês anterior.")
    st.write("Ajuda a priorizar ações e replicar boas práticas.")

# 3) Progresso YTD (acumulado no ano)
col_e, col_f = st.columns([2, 1])
with col_e:
    if not df_lojas.empty and df_lojas["data"].notna().any():
        df_aux = df_lojas.copy()
        df_aux["ano"] = df_aux["data"].dt.year
        df_aux["mes"] = df_aux["data"].dt.month
        cur_year = df_aux["ano"].max()
        series_ytd = (df_aux[df_aux["ano"].isin([cur_year, cur_year-1])]
                      .groupby(["ano","mes"], as_index=False)["faturamento"].sum()
                      .sort_values(["ano","mes"]))
        series_ytd["ytd"] = series_ytd.groupby("ano")["faturamento"].cumsum()
        fig_ytd = px.line(series_ytd, x="mes", y="ytd", color="ano", markers=True)
        fig_ytd.update_layout(height=330, xaxis_title="Mês", yaxis_title="YTD Faturamento (R$)")
        st.plotly_chart(fig_ytd, use_container_width=True)
    else:
        st.info("Sem dados suficientes para YTD.")
with col_f:
    st.write("Como calculamos:")
    st.write("YTD é a soma acumulada do faturamento dentro do ano. Comparamos o ano atual com o anterior para avaliar ritmo.")

# 4) Participação por loja no mês atual
st.subheader("Participação por loja (mês atual)")
if k["current"] is not None:
    part = (df_lojas.loc[df_lojas["data"]==k["current"]].groupby("loja", as_index=False)["faturamento"].sum())
    if not part.empty:
        total = part["faturamento"].sum()
        part["share"] = 100*part["faturamento"].apply(lambda v: safe_div(v, total))
        part = part.sort_values("share", ascending=False).head(20)
        fig_part = px.bar(part, x="share", y="loja", orientation="h")
        fig_part.update_layout(
            height=380,
            xaxis_title="Participação (%)",
            yaxis_title="Loja"
        )
        st.plotly_chart(fig_part, use_container_width=True)
    else:
        st.info("Sem dados para participação no mês atual.")
else:
    st.info("Mês atual indisponível para participação.")

# 5) Eficiência por loja (Faturamento vs Pedidos)
st.subheader("Eficiência por loja (Faturamento vs Pedidos)")
if k["current"] is not None:
    eff = df_lojas.loc[df_lojas["data"]==k["current"],["loja","faturamento","pedidos"]]
    if not eff.empty:
        eff = eff.groupby("loja", as_index=False).agg(faturamento=("faturamento","sum"), pedidos=("pedidos","sum"))
        eff["ticket"] = eff.apply(lambda r: safe_div(r["faturamento"], r["pedidos"]), axis=1)
        fig_eff = px.scatter(eff, x="pedidos", y="faturamento", size="ticket", hover_name="loja")
        fig_eff.update_layout(height=380, xaxis_title="Pedidos", yaxis_title="Faturamento (R$)")
        st.plotly_chart(fig_eff, use_container_width=True)
    else:
        st.info("Sem dados para eficiência no mês atual.")
else:
    st.info("Mês atual indisponível para eficiência.")

# -----------------------------------------------------------------------------
# RESUMO + DOWNLOAD
# -----------------------------------------------------------------------------

st.subheader("Resumo por loja e período")
if df_f.empty:
    st.info("Sem dados no filtro atual.")
else:
    resumo = (df_f.assign(ano_mes=df_f["periodo"]).groupby(["ano_mes","loja"], as_index=False)
              .agg(faturamento=("faturamento","sum"), pedidos=("pedidos","sum")))
    resumo["ticket_medio"] = resumo.apply(lambda r: safe_div(r["faturamento"], r["pedidos"]), axis=1)
    st.dataframe(resumo, use_container_width=True, height=360)
    st.download_button("Baixar resumo (CSV)", data=resumo.to_csv(index=False).encode("utf-8"), file_name="resumo_faturamento.csv", mime="text/csv")

st.caption("Os gráficos e KPIs se ajustam dinamicamente ao intervalo de períodos e lojas selecionadas.")
st.caption("Dashboard desenvolvido por Evandro Segnorelli.")
