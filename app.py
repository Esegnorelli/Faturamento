# app.py — Dashboard Inteligente (Streamlit)
#
# Requisitos: streamlit, pandas, plotly, python-dateutil, statsmodels
# Execução: streamlit run app.py
#
# • 1 página, sem usar st.markdown/HTML
# • Período via intervalo contínuo (AAAA‑MM) com select_slider + Lojas por checkboxes (com grupos BGPF/Ismael opcionais)
# • KPIs nativos (st.metric) com deltas MoM
# • Gráficos coloridos e limpos: Faturamento, Pedidos, Ticket médio
# • "Insights automáticos": MoM/YoY, Top Movimentos, YTD, Participação e Eficiência (com explicações em texto simples)
# • NOVOS: Média Móvel, Treemap, Heatmap de Desempenho, Decomposição de Série Temporal, Top 3 por Pedidos (ignora filtro de lojas)

import os
import re
import unicodedata
from dateutil.relativedelta import relativedelta  # <— corrigido (era relativelota)

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
import statsmodels.api as sm

# -----------------------------------------------------------------------------
# CONFIG BÁSICA
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Dashboard Hora do Pastel — KPIs Inteligentes",
    page_icon="🥟",
    layout="wide",
    initial_sidebar_state="expanded",
)

px.defaults.template = "simple_white"
px.defaults.color_discrete_sequence = px.colors.qualitative.Pastel

# -----------------------------------------------------------------------------
# HELPERS DE TRATAMENTO E FORMATAÇÃO
# -----------------------------------------------------------------------------

def normalize_col(name: str) -> str:
    name = name.strip().lower()
    name = "".join(c for c in unicodedata.normalize("NFKD", name) if not unicodedata.combining(c))
    return re.sub(r"\s+", "_", name)


def _norm_text(s: str) -> str:
    s = ''.join(c for c in unicodedata.normalize('NFKD', str(s).strip().lower()) if not unicodedata.combining(c))
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return ' '.join(s.split())


def br_to_float(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()
    s = s.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    s = s.str.replace(r"[^0-9,\.\-]", "", regex=True)
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


def fmt_pct(v, decimals=1):
    if pd.isna(v) or v is None:
        return None
    return f"{v * 100:,.{decimals}f}%".replace(".", ",")

# -----------------------------------------------------------------------------
# CARGA E TRATAMENTO DE DADOS
# -----------------------------------------------------------------------------

@st.cache_data(ttl=3600, max_entries=8, show_spinner=False)
def load_data() -> pd.DataFrame:
    # Tenta carregar o arquivo tratado, se não existir, processa o original
    if os.path.exists("Faturamento_tratado.csv"):
        df = pd.read_csv("Faturamento_tratado.csv")
    elif os.path.exists("Faturamento.csv"):
        df = pd.read_csv("Faturamento.csv", sep=None, engine="python")
        df.columns = [normalize_col(c) for c in df.columns]
        df = df.loc[:, ~df.columns.duplicated()].dropna(axis=1, how="all")
        df = df.rename(columns=rename_by_alias(list(df.columns)))
        for col in ["mes","ano","loja","faturamento","pedidos","ticket"]:
            if col not in df.columns:
                df[col] = pd.NA
        df["mes"] = month_to_int(df["mes"])
        df["ano"] = pd.to_numeric(df["ano"], errors="coerce").astype("Int64")
        df["faturamento"] = br_to_float(df["faturamento"])
        df["ticket"] = br_to_float(df["ticket"])
        df["pedidos"] = pd.to_numeric(df["pedidos"], errors="coerce").round().astype("Int64")
    else:
        st.error("Arquivo 'Faturamento.csv' não encontrado. Por favor, faça o upload do arquivo.")
        st.stop()

    # Padronização final e criação de colunas de data
    for c in ["mes","ano","pedidos"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    for c in ["faturamento","ticket"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "loja" in df.columns:
        df["loja"] = df["loja"].astype(str).str.strip()

    mask = df["ano"].notna() & df["mes"].notna()
    df["data"] = pd.NaT
    df.loc[mask, "data"] = pd.to_datetime(
        {"year": df.loc[mask, "ano"].astype(int), "month": df.loc[mask, "mes"].astype(int), "day": 1},
        errors="coerce",
    )
    df["periodo"] = df["data"].dt.to_period("M").astype(str)
    return df.dropna(subset=['data'])


df = load_data()

# -----------------------------------------------------------------------------
# UI: FILTROS (Período por intervalo + Lojas por checkboxes com grupos opcionais)
# -----------------------------------------------------------------------------

# Logo (opcional, evita erro quando arquivo não existe)
if os.path.exists("logo.png"):
    st.sidebar.image("logo.png", use_column_width=True)

st.sidebar.header("Filtros")

# Períodos (AAAA‑MM) como intervalo contínuo
periodos = sorted(p for p in df["periodo"].dropna().unique().tolist())
if len(periodos) < 2:
    st.warning("Dados insuficientes para análise. São necessários pelo menos 2 meses de dados.")
    st.stop()

rng_default = (periodos[0], periodos[-1])
periodo_ini, periodo_fim = st.sidebar.select_slider(
    "Período (AAAA‑MM)",
    options=periodos,
    value=rng_default,
)

# Grupos opcionais
GROUPS = {
    "BGPF": [
        "Caxias do Sul",
        "Bento Goncalves",
        "Novo Hamburgo",
        "Sao leopoldo",
        "Canoas",
        "Protasio",
        "Floresta",
        "Barra Shopping",
    ],
    "Ismael": ["Montenegro", "Lajeado"],
}
mode = st.sidebar.radio("Selecionar lojas por", ["Manual", "BGPF", "Ismael"], index=0)

# Lojas (checkboxes com "Selecionar todas")
lojas = sorted(df["loja"].dropna().unique().tolist())
map_norm_to_loja = {_norm_text(l): l for l in lojas}

if mode == "Manual":
    st.sidebar.write("Lojas (marque as desejadas):")
    all_l = st.sidebar.checkbox("Selecionar todas as lojas", value=True, key="all_l")
    sel_lojas = lojas if all_l else [l for l in lojas if st.sidebar.checkbox(l, value=False, key=f"l_{l}")]
    if not sel_lojas:
        sel_lojas = lojas  # fallback
else:
    candidatos = [_norm_text(x) for x in GROUPS.get(mode, [])]
    sel_lojas = [map_norm_to_loja[c] for c in candidatos if c in map_norm_to_loja]
    st.sidebar.info(f"Grupo {mode}: {', '.join(sel_lojas) if sel_lojas else 'nenhuma loja do grupo encontrada nos dados.'}")

# Aplica filtros
mask = (df["periodo"] >= periodo_ini) & (df["periodo"] <= periodo_fim) & (df["loja"].isin(sel_lojas))
df_f = df.loc[mask].copy()

# Para comparações globais (MoM/YoY, YTD, etc.), considera toda a série temporal das lojas escolhidas
df_lojas = df[df["loja"].isin(sel_lojas)].copy()

# -----------------------------------------------------------------------------
# KPI ENGINE
# -----------------------------------------------------------------------------

def _delta(cur_v, base_v):
    if cur_v is None or base_v in (None, 0) or pd.isna(base_v):
        return None
    return safe_div((cur_v - base_v), base_v)

@st.cache_data(show_spinner=False)
def compute_kpis(df_range: pd.DataFrame, df_comp: pd.DataFrame, p_ini: str, p_fim: str):
    # KPIs do período selecionado
    tot_fat = float(df_range["faturamento"].sum())
    tot_ped = int(df_range["pedidos"].sum()) if df_range["pedidos"].notna().any() else 0
    tik_med = safe_div(tot_fat, tot_ped)

    # Comparação com período anterior (mesmo tamanho)
    start_date = pd.to_datetime(p_ini)
    end_date = pd.to_datetime(p_fim)
    num_months = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month) + 1

    prev_end_date = start_date - relativedelta(months=1)
    prev_start_date = prev_end_date - relativedelta(months=num_months - 1)

    mask_prev = (df_comp["data"] >= prev_start_date) & (df_comp["data"] <= prev_end_date)
    df_prev_period = df_comp[mask_prev]

    prev_fat = float(df_prev_period["faturamento"].sum())
    delta_period_fat = _delta(tot_fat, prev_fat)

    # Comparação com mesmo período do ano anterior
    yoy_start_date = start_date - relativedelta(years=1)
    yoy_end_date = end_date - relativedelta(years=1)

    mask_yoy = (df_comp["data"] >= yoy_start_date) & (df_comp["data"] <= yoy_end_date)
    df_yoy_period = df_comp[mask_yoy]

    yoy_fat = float(df_yoy_period["faturamento"].sum())
    delta_yoy_fat = _delta(tot_fat, yoy_fat)

    # KPIs do último mês (MoM)
    serie_all = (df_comp.dropna(subset=["data"]).groupby("data", as_index=False)
                 .agg(faturamento=("faturamento","sum"), pedidos=("pedidos","sum")))
    serie_all["ticket_medio"] = serie_all.apply(lambda r: safe_div(r["faturamento"], r["pedidos"]), axis=1)
    serie_all = serie_all.sort_values("data")

    mom_fat = mom_ped = mom_tik = None
    if len(serie_all) >= 2:
        last = serie_all.iloc[-1]
        prev = serie_all.iloc[-2]
        mom_fat = _delta(last["faturamento"], prev["faturamento"])
        mom_ped = _delta(last["pedidos"], prev["pedidos"])
        mom_tik = _delta(last["ticket_medio"], prev["ticket_medio"])

    return {
        "period_sum": {"fat": tot_fat, "ped": tot_ped, "tik": tik_med},
        "prev_period_fat": prev_fat,
        "delta_period_fat": delta_period_fat,
        "delta_yoy_fat": delta_yoy_fat,
        "yoy_fat_abs": yoy_fat,
        "mom_fat": mom_fat,
        "mom_ped": mom_ped,
        "mom_tik": mom_tik,
    }

k = compute_kpis(df_f, df_lojas, periodo_ini, periodo_fim)

# -----------------------------------------------------------------------------
# CABEÇALHO E KPIs
# -----------------------------------------------------------------------------

st.title("Dashboard Inteligente — Hora do Pastel")
st.write(
    f"Período: **{periodo_ini}** a **{periodo_fim}** | "
    f"Lojas selecionadas: **{len(sel_lojas)}** de {len(lojas)}"
)
st.divider()

m1, m2, m3, m4 = st.columns(4)
m1.metric(
    label="Faturamento no Período",
    value=fmt_brl(k["period_sum"]["fat"]),
    delta=fmt_pct(k["delta_period_fat"]),
    help=(f"Período anterior: {fmt_brl(k['prev_period_fat'])}" if k.get('prev_period_fat') is not None else None),
)
m2.metric(
    label="Pedidos no Período",
    value=fmt_int(k["period_sum"]["ped"]),
    delta=fmt_pct(k["mom_ped"]),
    help="Variação MoM (Mês vs Mês anterior) do total de pedidos.",
)
m3.metric(
    label="Ticket Médio no Período",
    value=fmt_brl(k["period_sum"]["tik"]),
    delta=fmt_pct(k["mom_tik"]),
    help="Variação MoM (Mês vs Mês anterior) do ticket médio.",
)
m4.metric(
    label="Fat. vs Ano Anterior",
    value=fmt_brl(k["period_sum"]["fat"]),
    delta=fmt_pct(k["delta_yoy_fat"]),
    help=(f"Mesmo período AA: {fmt_brl(k['yoy_fat_abs'])}" if k.get('yoy_fat_abs') is not None else None),
)

st.divider()

# -----------------------------------------------------------------------------
# TOP 3 LOJAS POR PEDIDOS (IGNORA FILTRO DE LOJAS)
# -----------------------------------------------------------------------------

st.subheader("Top 3 lojas por pedidos (período selecionado — ignora filtro de lojas)")
rank_mask = (df["periodo"] >= periodo_ini) & (df["periodo"] <= periodo_fim) & df["pedidos"].notna()
rank_df = (df.loc[rank_mask]
             .groupby("loja", as_index=False)["pedidos"].sum()
             .sort_values("pedidos", ascending=False))
if not rank_df.empty:
    total_ped = int(rank_df["pedidos"].sum()) if pd.notna(rank_df["pedidos"].sum()) else 0
    for _, r in rank_df.head(3).iterrows():
        pct = (r["pedidos"] / total_ped * 100) if total_ped else 0
        st.write(f"• {r['loja']}: {fmt_int(r['pedidos'])} pedidos ({pct:.1f}%)")
else:
    st.info("Sem pedidos no período.")

# -----------------------------------------------------------------------------
# GRÁFICOS PRINCIPAIS (tabs)
# -----------------------------------------------------------------------------

tabs = st.tabs(["📈 Evolução", "🏢 Desempenho por Loja", "🔬 Análise Avançada"])

with tabs[0]:
    st.subheader("Evolução dos Indicadores no Período")

    serie_f = (df_f.dropna(subset=["data"]).groupby("data", as_index=False)
               .agg(faturamento=("faturamento", "sum"), pedidos=("pedidos", "sum"))
               .sort_values("data"))

    if not serie_f.empty:
        serie_f["ticket_medio"] = serie_f.apply(lambda r: safe_div(r["faturamento"], r["pedidos"]), axis=1)
        serie_f['faturamento_mm3'] = serie_f['faturamento'].rolling(window=3).mean()

        fig = make_subplots(specs=[[{"secondary_y": True}]])

        fig.add_trace(go.Scatter(x=serie_f['data'], y=serie_f['faturamento'], name='Faturamento', mode='lines+markers', line=dict(width=3)), secondary_y=False)
        fig.add_trace(go.Scatter(x=serie_f['data'], y=serie_f['faturamento_mm3'], name='Média Móvel (3M)', mode='lines', line=dict(width=2, dash='dot')), secondary_y=False)
        fig.add_trace(go.Bar(x=serie_f['data'], y=serie_f['pedidos'], name='Pedidos', opacity=0.5), secondary_y=True)

        fig.update_layout(height=400, title="Faturamento, Média Móvel e Pedidos", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
        fig.update_yaxes(title_text="Faturamento (R$)", secondary_y=False)
        fig.update_yaxes(title_text="Pedidos", secondary_y=True)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("Sem dados no filtro atual.")

with tabs[1]:
    st.subheader("Análise Comparativa entre Lojas")

    if not df_f.empty:
        col1, col2 = st.columns([1, 1])

        with col1:
            st.write("**Contribuição no Faturamento (Treemap)**")
            part = df_f.groupby("loja", as_index=False)["faturamento"].sum()
            if not part.empty:
                fig_tree = px.treemap(part, path=['loja'], values='faturamento',
                                      title='Participação de cada loja no faturamento do período',
                                      color='faturamento',
                                      color_continuous_scale='Greens')
                fig_tree.update_layout(height=400)
                st.plotly_chart(fig_tree, use_container_width=True)
            else:
                st.info("Sem dados para o treemap.")

        with col2:
            st.write("**Eficiência (Faturamento vs. Pedidos)**")
            eff = df_f.groupby("loja", as_index=False).agg(faturamento=("faturamento","sum"), pedidos=("pedidos","sum"))
            if not eff.empty and eff['pedidos'].sum() > 0:
                eff["ticket"] = eff.apply(lambda r: safe_div(r["faturamento"], r["pedidos"]), axis=1)
                fig_eff = px.scatter(eff, x="pedidos", y="faturamento", size="ticket", color="loja",
                                     hover_name="loja", size_max=60, title="Eficiência da Loja no Período")
                fig_eff.update_layout(height=400)
                st.plotly_chart(fig_eff, use_container_width=True)
            else:
                st.info("Sem dados para o gráfico de eficiência.")

        st.write("**Desempenho Mensal por Loja (Heatmap)**")
        heatmap_data = df_f.pivot_table(index='loja', columns='periodo', values='faturamento', aggfunc='sum').fillna(0)
        if not heatmap_data.empty:
            fig_heatmap = go.Figure(data=go.Heatmap(
                z=heatmap_data.values,
                x=heatmap_data.columns,
                y=heatmap_data.index,
                colorscale='Greens'))
            fig_heatmap.update_layout(title='Faturamento Mensal por Loja', xaxis_nticks=36, height=500)
            st.plotly_chart(fig_heatmap, use_container_width=True)
        else:
            st.info("Sem dados para o heatmap.")

with tabs[2]:
    st.subheader("Análise de Série Temporal")
    st.write("Esta análise decompõe a série de faturamento para revelar padrões mais profundos.")

    serie_all = (df_lojas.groupby('data')['faturamento'].sum().sort_index())
    if len(serie_all) >= 24:  # Requer pelo menos 2 anos para sazonalidade robusta
        try:
            serie_all.index = pd.to_datetime(serie_all.index)
            res = sm.tsa.seasonal_decompose(serie_all.asfreq('MS'), model='additive')

            fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                                subplot_titles=("Original", "Tendência", "Sazonalidade", "Resíduos"))
            fig.add_trace(go.Scatter(x=res.observed.index, y=res.observed, mode='lines', name='Original'), row=1, col=1)
            fig.add_trace(go.Scatter(x=res.trend.index, y=res.trend, mode='lines', name='Tendência'), row=2, col=1)
            fig.add_trace(go.Scatter(x=res.seasonal.index, y=res.seasonal, mode='lines', name='Sazonalidade'), row=3, col=1)
            fig.add_trace(go.Scatter(x=res.resid.index, y=res.resid, mode='markers', name='Resíduos'), row=4, col=1)

            fig.update_layout(height=700, showlegend=False)
            st.plotly_chart(fig, use_container_width=True)

            st.write("""
            - **Tendência**: A direção geral do faturamento ao longo do tempo (crescendo, diminuindo ou estável).
            - **Sazonalidade**: Padrões que se repetem em intervalos fixos (ex.: picos de vendas no final do ano).
            - **Resíduos**: Flutuações aleatórias que não são explicadas pela tendência ou sazonalidade.
            """)
        except Exception as e:
            st.info(f"Não foi possível decompor a série (verifique lacunas mensais): {e}")
    else:
        st.info("A análise de decomposição requer pelo menos 24 meses de dados para as lojas selecionadas.")

st.divider()

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

    # Formatação para exibição
    resumo_fmt = resumo.copy()
    resumo_fmt['faturamento'] = resumo_fmt['faturamento'].apply(fmt_brl)
    resumo_fmt['ticket_medio'] = resumo_fmt['ticket_medio'].apply(fmt_brl)
    resumo_fmt['pedidos'] = resumo_fmt['pedidos'].apply(fmt_int)

    st.dataframe(resumo_fmt, use_container_width=True, height=360)
    st.download_button(
        "Baixar resumo (CSV)",
        data=resumo.to_csv(index=False).encode("utf-8"),
        file_name="resumo_faturamento.csv",
        mime="text/csv",
    )
try:
    import statsmodels.api as sm
except ModuleNotFoundError:
    sm = None
# ...
# onde usa:
if sm is None:
    st.info("Instale 'statsmodels' (requirements.txt) para ver a decomposição de série.")
else:
    # segue a análise com sm.tsa.seasonal_decompose(...)

st.caption("Dashboard aprimorado com novos indicadores inteligentes e correções de estabilidade.")
