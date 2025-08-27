# app_insights_firasans.py — Dashboard Inteligente (Streamlit)
# 
# • 1 página (sem abas), focado em Faturamento, Pedidos e Ticket
# • KPIs compactos + variações MoM/YoY
# • Gráficos limpos e coloridos (faturamento, pedidos, ticket)
# • Seção “🔎 Insights automáticos” com gráficos e explicações
# • Fonte única: Fira Sans (texto e números)
# • Lê Faturamento_tratado.csv; se não existir, trata Faturamento.csv
# 
# Execução:
#   streamlit run app_insights_firasans.py

import os
import re
import unicodedata
from datetime import datetime
from dateutil.relativedelta import relativedelta

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ──────────────────────────────────────────────────────────────────────────────
# CONFIG & ESTILO (somente Fira Sans)
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Dashboard Hora do Pastel — KPIs Inteligentes",
    page_icon="🥟",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Paleta de cores consistente para os gráficos
COLOR_SEQ = ["#B40000", "#0E7490", "#7C3AED", "#EA580C", "#047857", "#2563EB", "#9333EA"]

# ──────────────────────────────────────────────────────────────────────────────
# HELPERS
# ──────────────────────────────────────────────────────────────────────────────

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


def fmt_pct(v) -> str:
    if v is None or pd.isna(v):
        return "—"
    return f"{100*float(v):.1f}%".replace(".", ",")


def safe_div(a, b):
    try:
        return (a / b) if b not in (0, None) and not pd.isna(b) else 0.0
    except Exception:
        return 0.0

# ──────────────────────────────────────────────────────────────────────────────
# CARGA/TRATAMENTO
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner=False)
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

# ──────────────────────────────────────────────────────────────────────────────
# FILTROS
# ──────────────────────────────────────────────────────────────────────────────

st.sidebar.title("⚙️ Filtros")

periodos = sorted(p for p in df["periodo"].dropna().unique())
if periodos:
    start_p, end_p = st.sidebar.select_slider(
        "Período",
        options=periodos,
        value=(periodos[0], periodos[-1]),
        help="Intervalo de meses (AAAA-MM)",
    )
else:
    start_p, end_p = None, None

lojas = sorted(df["loja"].dropna().unique().tolist())
lojas_sel = st.sidebar.multiselect("Lojas", options=lojas, default=lojas)

mask = pd.Series([True]*len(df))
if start_p and end_p:
    mask &= (df["periodo"] >= start_p) & (df["periodo"] <= end_p)
if lojas_sel:
    mask &= df["loja"].isin(lojas_sel)

df_f = df.loc[mask].copy()
df_lojas = df[df["loja"].isin(lojas_sel)].copy()

# ──────────────────────────────────────────────────────────────────────────────
# KPI ENGINE
# ──────────────────────────────────────────────────────────────────────────────

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

# ──────────────────────────────────────────────────────────────────────────────
# HEADER
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("### 🥟 Dashboard Inteligente — Hora do Pastel")
st.markdown(
    f"<span class='badge'>Período:</span> <b>{start_p or '—'}</b> a <b>{end_p or '—'}</b> &nbsp; "
    f"<span class='badge'>Lojas:</span> <b>{len(lojas_sel)}</b>",
    unsafe_allow_html=True,
)

# ──────────────────────────────────────────────────────────────────────────────
# KPIs COMPACTOS
# ──────────────────────────────────────────────────────────────────────────────

def kpi_card(title, main_value, delta_value, help_text):
    arrow = "▲" if (delta_value is not None and delta_value >= 0) else ("▼" if delta_value is not None else "•")
    cls = "up" if (delta_value is not None and delta_value >= 0) else ("down" if delta_value is not None else "")
    delta_txt = "" if delta_value is None else f"<span class='delta {cls}'>{arrow} {fmt_pct(delta_value)}</span>"
    st.markdown(f"<div class='kpi'><h3>{title}</h3><p>{main_value}{delta_txt}</p><small>{help_text}</small></div>", unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns([1,1,1,1])
with c1:
    kpi_card("Faturamento (período)", fmt_brl(k["period_sum"]["fat"]), None, "Total considerando filtros")
with c2:
    kpi_card("Pedidos (período)", fmt_int(k["period_sum"]["ped"]), None, "Total considerando filtros")
with c3:
    kpi_card(
        f"Ticket médio {'('+k['current'].strftime('%b/%Y')+')' if k['current'] else ''}",
        fmt_brl(k["tik_cur"] if k["tik_cur"] is not None else k["period_sum"]["tik"]),
        k["mom_tik"],
        "Comparado ao mês anterior (MoM)",
    )
with c4:
    kpi_card(
        f"Faturamento {'('+k['current'].strftime('%b/%Y')+')' if k['current'] else ''}",
        fmt_brl(k["fat_cur"]),
        k["mom_fat"],
        "Variações abaixo",
    )

# ──────────────────────────────────────────────────────────────────────────────
# GRÁFICOS PRINCIPAIS (faturamento, pedidos, ticket)
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("<div class='section-title'>📈 Evolução do faturamento</div>", unsafe_allow_html=True)
serie_f = (df_f.dropna(subset=["data"]).groupby("data", as_index=False)["faturamento"].sum().sort_values("data"))
if not serie_f.empty:
    fig = px.line(serie_f, x="data", y="faturamento", markers=True, color_discrete_sequence=[COLOR_SEQ[0]])
    fig.update_layout(margin=dict(l=0,r=0,t=0,b=0), height=320, font=dict(family="Fira Sans"))
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Sem dados no filtro atual.")

st.markdown("<div class='section-title'>📦 Evolução de pedidos</div>", unsafe_allow_html=True)
serie_p = (df_f.dropna(subset=["data"]).groupby("data", as_index=False)["pedidos"].sum().sort_values("data"))
if not serie_p.empty:
    fig = px.line(serie_p, x="data", y="pedidos", markers=True, color_discrete_sequence=[COLOR_SEQ[1]])
    fig.update_layout(margin=dict(l=0,r=0,t=0,b=0), height=320, font=dict(family="Fira Sans"))
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Sem dados de pedidos no filtro atual.")

st.markdown("<div class='section-title'>🎟️ Ticket médio (mensal)</div>", unsafe_allow_html=True)
serie_t = (df_f.dropna(subset=["data"]).groupby("data", as_index=False)
           .agg(faturamento=("faturamento","sum"), pedidos=("pedidos","sum")))
if not serie_t.empty:
    serie_t["ticket_medio"] = serie_t.apply(lambda r: safe_div(r["faturamento"], r["pedidos"]), axis=1)
    fig = px.line(serie_t, x="data", y="ticket_medio", markers=True, color_discrete_sequence=[COLOR_SEQ[2]])
    fig.update_layout(margin=dict(l=0,r=0,t=0,b=0), height=320, font=dict(family="Fira Sans"))
    st.plotly_chart(fig, use_container_width=True)
else:
    st.info("Sem dados para calcular ticket médio.")

# ──────────────────────────────────────────────────────────────────────────────
# 🔎 INSIGHTS AUTOMÁTICOS (gráficos + explicações)
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("<div class='section-title'>🔎 Insights automáticos</div>", unsafe_allow_html=True)

# 1) MoM e YoY de Faturamento (séries)
serie_all = month_series(df_lojas)
col_a, col_b = st.columns([2,1])
with col_a:
    if not serie_all.empty:
        serie_all = serie_all.sort_values("data")
        serie_all["mom"] = serie_all["faturamento"].pct_change()
        serie_all["yoy"] = serie_all["faturamento"].pct_change(12)
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=serie_all["data"], y=100*serie_all["mom"], name="MoM %", mode="lines+markers"))
        fig.add_trace(go.Scatter(x=serie_all["data"], y=100*serie_all["yoy"], name="YoY %", mode="lines+markers"))
        fig.update_layout(margin=dict(l=0,r=0,t=0,b=0), height=340, font=dict(family="Fira Sans"))
        fig.update_yaxes(title_text="Variação (%)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Sem dados para variações MoM/YoY.")
with col_b:
    st.markdown(
        """
        <div class='explain'>
        <b>Como calculamos:</b><br/>
        <span class='mono'>MoM = (Fat_mês_atual − Fat_mês_anterior) / Fat_mês_anterior</span><br/>
        <span class='mono'>YoY = (Fat_mês_atual − Fat_mesmo_mês_ano_passado) / Fat_mesmo_mês_ano_passado</span>
        <br/><br/>
        • MoM foca em <i>tendência recente</i>.<br/>
        • YoY elimina sazonalidade mensal ao comparar com o mesmo mês do ano anterior.
        </div>
        """,
        unsafe_allow_html=True,
    )

# 2) Top Movimentos (MoM por loja no mês atual)
col_c, col_d = st.columns([2,1])
with col_c:
    if k["current"] is not None:
        tbl = per_store_current_vs_prev(df_lojas, k["current"], k["prev"])
        if not tbl.empty:
            tbl = tbl.assign(mom_pct=100*tbl["mom"]).sort_values("mom_pct", ascending=False)
            top_up = tbl.head(10)
            fig = px.bar(top_up, x="loja", y="mom_pct", color_discrete_sequence=[COLOR_SEQ[3]])
            fig.update_layout(margin=dict(l=0,r=0,t=0,b=0), height=340, font=dict(family="Fira Sans"))
            fig.update_yaxes(title_text="MoM (%)")
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("Sem base para comparar mês atual vs anterior.")
    else:
        st.info("Mês atual indisponível para comparação.")
with col_d:
    st.markdown(
        """
        <div class='explain'>
        <b>O que mostra:</b><br/>
        Lojas com maior <b>crescimento percentual</b> no faturamento no mês atual vs o mês anterior.
        <br/><br/>
        Útil para priorizar o que está funcionando e replicar boas práticas.
        </div>
        """,
        unsafe_allow_html=True,
    )

# 3) Progresso YTD (acumulado no ano)
col_e, col_f = st.columns([2,1])
with col_e:
    if not df_lojas.empty and df_lojas["data"].notna().any():
        df_lojas["ano_num"] = df_lojas["data"].dt.year
        df_lojas["mes_num"] = df_lojas["data"].dt.month
        cur_year = df_lojas["ano_num"].max()
        series_ytd = (df_lojas[df_lojas["ano_num"].isin([cur_year, cur_year-1])]
                      .groupby(["ano_num","mes_num"], as_index=False)["faturamento"].sum()
                      .sort_values(["ano_num","mes_num"]))
        series_ytd["ytd"] = series_ytd.groupby("ano_num")["faturamento"].cumsum()
        fig = px.line(series_ytd, x="mes_num", y="ytd", color="ano_num", markers=True, color_discrete_sequence=[COLOR_SEQ[0], COLOR_SEQ[4]])
        fig.update_layout(margin=dict(l=0,r=0,t=0,b=0), height=340, font=dict(family="Fira Sans"))
        fig.update_xaxes(title_text="Mês")
        fig.update_yaxes(title_text="YTD Faturamento (R$)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Sem dados suficientes para YTD.")
with col_f:
    st.markdown(
        """
        <div class='explain'>
        <b>Como calculamos:</b><br/>
        <span class='mono'>YTD = soma mensal acumulada dentro do mesmo ano</span>.<br/>
        Comparamos o ano atual com o anterior para avaliar ritmo ao longo do ano.
        </div>
        """,
        unsafe_allow_html=True,
    )

# 4) Participação por loja no mês atual
st.markdown("<div class='section-title'>🥧 Participação por loja (mês atual)</div>", unsafe_allow_html=True)
if k["current"] is not None:
    part = (df_lojas.loc[df_lojas["data"]==k["current"]].groupby("loja", as_index=False)["faturamento"].sum())
    if not part.empty:
        total = part["faturamento"].sum()
        part["share"] = 100*part["faturamento"].apply(lambda v: safe_div(v, total))
        part = part.sort_values("share", ascending=False).head(20)
        fig = px.bar(part, x="share", y="loja", orientation="h", color_discrete_sequence=[COLOR_SEQ[5]])
        fig.update_layout(margin=dict(l=0,r=0,t=0,b=0), height=420, font=dict(family="Fira Sans"))
        fig.update_xaxes(title_text="Participação (%)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Sem dados para participação no mês atual.")
else:
    st.info("Mês atual indisponível para participação.")

# 5) Eficiência por loja (Faturamento vs Pedidos) — bolha = Ticket
st.markdown("<div class='section-title'>🧪 Eficiência por loja (Faturamento vs Pedidos)</div>", unsafe_allow_html=True)
if k["current"] is not None:
    eff = df_lojas.loc[df_lojas["data"]==k["current"],["loja","faturamento","pedidos"]]
    if not eff.empty:
        eff = eff.groupby("loja", as_index=False).agg(faturamento=("faturamento","sum"), pedidos=("pedidos","sum"))
        eff["ticket"] = eff.apply(lambda r: safe_div(r["faturamento"], r["pedidos"]), axis=1)
        fig = px.scatter(eff, x="pedidos", y="faturamento", size="ticket", hover_name="loja",
                         color_discrete_sequence=[COLOR_SEQ[6]])
        fig.update_layout(margin=dict(l=0,r=0,t=0,b=0), height=420, font=dict(family="Fira Sans"))
        fig.update_xaxes(title_text="Pedidos")
        fig.update_yaxes(title_text="Faturamento (R$)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Sem dados para eficiência no mês atual.")
else:
    st.info("Mês atual indisponível para eficiência.")

# ──────────────────────────────────────────────────────────────────────────────
# RESUMO + DOWNLOAD
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("<div class='section-title'>📄 Resumo por loja e período</div>", unsafe_allow_html=True)
if df_f.empty:
    st.info("Sem dados no filtro atual.")
else:
    resumo = (df_f.assign(ano_mes=df_f["periodo"]).groupby(["ano_mes","loja"], as_index=False)
              .agg(faturamento=("faturamento","sum"), pedidos=("pedidos","sum")))
    resumo["ticket_medio"] = resumo.apply(lambda r: safe_div(r["faturamento"], r["pedidos"]), axis=1)
    st.dataframe(resumo, use_container_width=True, height=360)
    st.download_button("⬇️ Baixar resumo (CSV)", data=resumo.to_csv(index=False).encode("utf-8"), file_name="resumo_faturamento.csv", mime="text/csv")

st.caption("Fonte: base local. Gráficos e KPIs adaptam-se dinamicamente ao período e às lojas selecionadas.")
