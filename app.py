"""
Hora do Pastel — Dashboard Inteligente v3.1 (adaptado aos anexos)
================================================================

✔ Usa automaticamente os anexos enviados nesta conversa:
   - /mnt/data/Faturamento.csv (dados)
   - /mnt/data/logo.png (logo)

✔ Compatível também com execução local: se os arquivos estiverem no mesmo
  diretório do app ("Faturamento.csv" e "logo.png"), ele detecta e usa.

Rodar:
    streamlit run app.py

Dependências mínimas: streamlit, pandas, plotly, numpy, python-dateutil, scipy
Opcionais: statsmodels (previsões/ACF), scikit-learn (clustering)
"""
from __future__ import annotations

import os
import re
import unicodedata
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from scipy.stats import pearsonr, zscore
from dateutil.relativedelta import relativedelta

# Fallbacks opcionais
try:
    import statsmodels.api as sm  # type: ignore
    from statsmodels.tsa.holtwinters import ExponentialSmoothing  # type: ignore
    HAS_STATSMODELS = True
except Exception:  # pragma: no cover
    sm = None  # type: ignore
    ExponentialSmoothing = None  # type: ignore
    HAS_STATSMODELS = False

try:
    from sklearn.cluster import KMeans  # type: ignore
    from sklearn.preprocessing import StandardScaler  # type: ignore
    HAS_SKLEARN = True
except Exception:
    KMeans = None  # type: ignore
    StandardScaler = None  # type: ignore
    HAS_SKLEARN = False

# =============================================================================
# ARQUIVOS/ANEXOS — paths amigáveis
# =============================================================================

def _first_existing(paths: Sequence[str]) -> Optional[str]:
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None

DATA_CANDIDATES = [
    "Faturamento_tratado.csv",
    "Faturamento.csv",
    os.path.join("data", "Faturamento_tratado.csv"),
    os.path.join("data", "Faturamento.csv"),
    "/mnt/data/Faturamento_tratado.csv",
    "/mnt/data/Faturamento.csv",
]

LOGO_CANDIDATES = [
    "logo.png",
    os.path.join("assets", "logo.png"),
    "/mnt/data/logo.png",
]

# =============================================================================
# TEMA/ESTILO
# =============================================================================

def configure_page() -> None:
    st.set_page_config(
        page_title="Hora do Pastel — Dashboard v3.1",
        page_icon="🥟",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    px.defaults.template = "plotly_white"
    px.defaults.color_discrete_sequence = [
        "#FF6B35", "#004E89", "#28A745", "#FFC107", "#DC3545", "#17A2B8",
        "#6C757D", "#343A40",
    ]

    st.markdown(
        """
        <style>
        .hero{background:linear-gradient(120deg,#FF6B35,#004E89);color:#fff;
              padding:1rem 1.25rem;border-radius:14px;margin-bottom:1rem}
        .metric-card{background:#fff;padding:1rem;border-radius:10px;
              box-shadow:0 4px 14px rgba(0,0,0,.06)}
        .badge{display:inline-block;padding:.1rem .5rem;border-radius:12px;font-size:.8rem}
        .b-ok{background:#28a745;color:#fff}.b-warn{background:#ffc107;color:#212529}
        .b-bad{background:#dc3545;color:#fff}
        </style>
        """,
        unsafe_allow_html=True,
    )

# =============================================================================
# HELPERS
# =============================================================================

def normalize_col(name: str) -> str:
    name = str(name).strip().lower()
    name = "".join(c for c in unicodedata.normalize("NFKD", name) if not unicodedata.combining(c))
    name = re.sub(r"[^\w\s]", "", name)
    return re.sub(r"\s+", "_", name)

ALIASES: Dict[str, List[str]] = {
    "mes": ["mes", "mês", "month", "mm"],
    "ano": ["ano", "year", "yyyy", "aa"],
    "loja": ["loja", "filial", "store", "unidade"],
    "faturamento": ["faturamento", "receita", "vendas", "valor_total", "revenue"],
    "pedidos": ["pedidos", "qtde_pedidos", "qtd_pedidos", "orders", "quantidade"],
    "ticket": ["ticket", "ticket_medio", "ticket_médio", "average_order"],
}

def rename_by_alias(cols: Sequence[str]) -> Dict[str, str]:
    ren: Dict[str, str] = {}
    for c in cols:
        for target, opts in ALIASES.items():
            if c in opts and target not in cols:
                ren[c] = target
                break
    return ren

# formatos BR

def fmt_brl(v: Any) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "R$ 0,00"
    s = f"{float(v):,.2f}"
    return "R$ " + s.replace(",", "X").replace(".", ",").replace("X", ".")

def fmt_int(v: Any) -> str:
    try:
        return f"{int(v):,}".replace(",", ".")
    except Exception:
        return "0"

def fmt_pct(v: Optional[float], dec: int = 1) -> str:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return f"0,{dec*'0'}%"
    return f"{v*100:,.{dec}f}%".replace(".", ",")

# =============================================================================
# CARGA DE DADOS
# =============================================================================

@st.cache_data(ttl=3600, show_spinner=False)
def load_data() -> pd.DataFrame:
    """Tenta carregar o CSV do diretório local e, se não existir,
    usa os anexos em /mnt/data. Normaliza e cria colunas derivadas.
    """
    path = _first_existing(DATA_CANDIDATES)

    uploaded = st.sidebar.file_uploader("Substituir dados (CSV)", type=["csv"], help="Opcional: envie um CSV no mesmo formato")
    if uploaded is not None:
        df = pd.read_csv(uploaded, sep=None, engine="python")
        src = f"upload: {uploaded.name}"
    elif path:
        df = pd.read_csv(path, sep=None, engine="python")
        src = os.path.basename(path)
    else:
        st.warning("Arquivo 'Faturamento.csv' não encontrado. Gerando dados de exemplo.")
        df = _generate_sample()
        src = "amostra sintética"

    # Normalização
    df.columns = [normalize_col(c) for c in df.columns]
    df = df.rename(columns=rename_by_alias(list(df.columns)))

    # Conversões básicas
    for c in ["mes", "ano", "pedidos"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    for c in ["faturamento", "ticket"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    if "loja" in df.columns:
        df["loja"] = df["loja"].astype(str).str.strip()

    # Construir data/periodo
    if "mes" in df.columns and "ano" in df.columns:
        mask = df["ano"].notna() & df["mes"].notna()
        df.loc[mask, "data"] = pd.to_datetime({
            "year": df.loc[mask, "ano"].astype(int),
            "month": df.loc[mask, "mes"].astype(int),
            "day": 1,
        }, errors="coerce")
    if "data" not in df.columns:
        st.error("O CSV precisa ter ao menos colunas de mês/ano ou 'data'.")
        st.stop()
    df["periodo"] = pd.to_datetime(df["data"]).dt.to_period("M").astype(str)

    # Ticket se faltar
    if "ticket" not in df.columns and {"faturamento", "pedidos"}.issubset(df.columns):
        df["ticket"] = df["faturamento"] / df["pedidos"].replace(0, np.nan)

    st.sidebar.caption(f"Fonte de dados: **{src}** — {len(df):,} linhas".replace(",", "."))
    return df.dropna(subset=["data"]).copy()


def _generate_sample() -> pd.DataFrame:
    np.random.seed(42)
    lojas = ["Centro", "Shopping A", "Shopping B", "Bairro Norte", "Bairro Sul"]
    dates = pd.date_range("2023-01-01", periods=24, freq="MS")
    data: List[Dict[str, Any]] = []
    for d in dates:
        for loja in lojas:
            base = 60000 + np.random.normal(0, 4000)
            saz = 1 + 0.25 * np.sin(2*np.pi*d.month/12)
            trend = 1 + 0.015 * ((d.year-2023)*12 + d.month-1)
            fat = base * saz * trend
            pedidos = int(max(1, fat / max(20, 30 + np.random.normal(0, 4))))
            ticket = fat / pedidos
            data.append({
                "mes": d.month, "ano": d.year, "loja": loja,
                "faturamento": fat, "pedidos": pedidos, "ticket": ticket,
                "data": d, "periodo": d.strftime("%Y-%m")
            })
    return pd.DataFrame(data)

# =============================================================================
# KPIs / MÉTRICAS
# =============================================================================

def safe_div(a: Optional[float], b: Optional[float]) -> float:
    try:
        if b in (0, None) or pd.isna(b):
            return 0.0
        return float(a) / float(b)
    except Exception:
        return 0.0

@st.cache_data(show_spinner=False)
def compute_kpis(df_range: pd.DataFrame, df_hist: pd.DataFrame,
                  p_ini: str, p_fim: str) -> Dict[str, Any]:
    tot_fat = float(df_range["faturamento"].sum())
    tot_ped = int(df_range["pedidos"].sum()) if "pedidos" in df_range.columns else 0
    tik_med = safe_div(tot_fat, tot_ped)

    serie_hist = (df_hist.dropna(subset=["data"]).groupby("data", as_index=False)
                   .agg(faturamento=("faturamento", "sum"), pedidos=("pedidos", "sum"))
                   .sort_values("data"))

    # períodos para comparação
    start = pd.to_datetime(p_ini)
    end = pd.to_datetime(p_fim)
    n_months = (end.year - start.year) * 12 + (end.month - start.month) + 1

    prev_end = start - relativedelta(months=1)
    prev_start = prev_end - relativedelta(months=n_months - 1)
    mask_prev = (df_hist["data"] >= prev_start) & (df_hist["data"] <= prev_end)
    prev_fat = float(df_hist.loc[mask_prev, "faturamento"].sum())

    def _delta(cur: float, base: float) -> Optional[float]:
        if base in (0, None) or pd.isna(base):
            return None
        return (cur - base) / base

    # YoY
    yoy_start = start - relativedelta(years=1)
    yoy_end = end - relativedelta(years=1)
    yoy_mask = (df_hist["data"] >= yoy_start) & (df_hist["data"] <= yoy_end)
    yoy_fat = float(df_hist.loc[yoy_mask, "faturamento"].sum())

    # MoM da última observação agregada
    mom_fat = mom_ped = mom_tik = None
    if len(serie_hist) >= 2:
        last, prev = serie_hist.iloc[-1], serie_hist.iloc[-2]
        mom_fat = _delta(float(last["faturamento"]), float(prev["faturamento"]))
        mom_ped = _delta(float(last["pedidos"]), float(prev["pedidos"]))
        last_tk = safe_div(last["faturamento"], last["pedidos"]) ; prev_tk = safe_div(prev["faturamento"], prev["pedidos"]) 
        mom_tik = _delta(last_tk, prev_tk)

    # Métricas avançadas
    adv: Dict[str, Any] = {}
    if not serie_hist.empty:
        s = serie_hist.set_index("data")["faturamento"].sort_index()
        if len(s) > 2:
            adv["growth_rate"] = (s.iloc[-1] / max(1e-9, s.iloc[0])) ** (1/(len(s)-1)) - 1
            adv["volatility"] = float(s.pct_change().dropna().std()) if len(s) > 3 else 0.0
        if len(serie_hist) > 3 and np.nanstd(serie_hist["pedidos"]) > 0 and np.nanstd(serie_hist["faturamento"]) > 0:
            corr, p = pearsonr(serie_hist["pedidos"], serie_hist["faturamento"])  # type: ignore
            adv["correlation"] = float(corr)
            adv["correlation_pvalue"] = float(p)
        # Eficiência: ticket atual vs histórico
        hist_ticket = safe_div(serie_hist["faturamento"].sum(), serie_hist["pedidos"].sum())
        adv["efficiency"] = safe_div(tik_med, hist_ticket) - 1
        adv["estimated_roi"] = tot_fat * 0.20

    return {
        "period_sum": {"fat": tot_fat, "ped": tot_ped, "tik": tik_med},
        "prev_period_fat": prev_fat,
        "delta_period_fat": _delta(tot_fat, prev_fat),
        "delta_yoy_fat": _delta(tot_fat, yoy_fat),
        "yoy_fat_abs": yoy_fat,
        "mom_fat": mom_fat, "mom_ped": mom_ped, "mom_tik": mom_tik,
        "advanced": adv, "serie_temporal": serie_hist,
    }

# =============================================================================
# UI COMPONENTES
# =============================================================================

def display_header(periodo_ini: str, periodo_fim: str, sel_lojas: Sequence[str], mode: str) -> None:
    st.markdown('<div class="hero"><h2>🥟 Hora do Pastel — Dashboard Inteligente</h2>'
                f'<p>Período: <b>{periodo_ini}</b> a <b>{periodo_fim}</b> · Lojas: <b>{len(sel_lojas)}</b> · Modo: <b>{mode}</b></p>'
                '</div>', unsafe_allow_html=True)


def display_kpis(k: Dict[str, Any]) -> None:
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("💰 Faturamento", fmt_brl(k["period_sum"]["fat"]), fmt_pct(k.get("delta_period_fat") or 0))
    with col2:
        st.metric("🛒 Pedidos", fmt_int(k["period_sum"]["ped"]), fmt_pct(k.get("mom_ped") or 0))
    with col3:
        st.metric("🎯 Ticket Médio", fmt_brl(k["period_sum"]["tik"]), fmt_pct(k.get("mom_tik") or 0))
    with col4:
        st.metric("📈 vs Ano Anterior", fmt_brl(k["yoy_fat_abs"]), fmt_pct(k.get("delta_yoy_fat") or 0))

    adv = k.get("advanced", {})
    if adv:
        col5, col6, col7, col8 = st.columns(4)
        with col5: st.metric("📊 Crescimento", fmt_pct(adv.get("growth_rate", 0)))
        with col6: st.metric("📉 Volatilidade", fmt_pct(adv.get("volatility", 0)))
        with col7: st.metric("⚡ Eficiência", fmt_pct(adv.get("efficiency", 0)))
        with col8: st.metric("💎 ROI Estimado", fmt_brl(adv.get("estimated_roi", 0)))


def display_alerts(k: Dict[str, Any]) -> None:
    adv = k.get("advanced", {})
    growth = adv.get("growth_rate", 0)
    vol = adv.get("volatility", 0)
    eff = adv.get("efficiency", 0)

    left, right = st.columns(2)
    with left:
        if growth > 0.05:
            st.success(f"🚀 Crescimento expressivo: {fmt_pct(growth)}")
        elif growth < -0.02:
            st.error(f"📉 Queda média mensal: {fmt_pct(abs(growth))}")
        else:
            st.warning(f"📊 Crescimento estável: {fmt_pct(growth)}")
    with right:
        if eff > 0.10:
            st.success(f"⚡ Eficiência acima do histórico: {fmt_pct(eff)}")
        elif vol > 0.30:
            st.warning(f"📈 Volatilidade elevada: {fmt_pct(vol)}")
        else:
            st.info("🎯 Performance consistente no período")


def display_detailed(df_f: pd.DataFrame, df_lojas: pd.DataFrame, k: Dict[str, Any]) -> None:
    tabs = st.tabs(["📊 Evolução", "🏪 Lojas", "🔬 Avançado"]) 

    # Evolução
    with tabs[0]:
        serie = (df_f.dropna(subset=["data"]).groupby("data", as_index=False)
                   .agg(faturamento=("faturamento", "sum"), pedidos=("pedidos", "sum"))
                   .sort_values("data"))
        if not serie.empty:
            serie["ticket"] = serie.apply(lambda r: safe_div(r["faturamento"], r["pedidos"]), axis=1)
            serie["mm3"] = serie["faturamento"].rolling(3, min_periods=1).mean()
            serie["mm6"] = serie["faturamento"].rolling(6, min_periods=1).mean()

            fig = make_subplots(rows=2, cols=2, specs=[[{"secondary_y": True}, {"secondary_y": False}],
                                                     [{"secondary_y": False}, {"secondary_y": False}]],
                                subplot_titles=("Faturamento e MM", "Pedidos", "Ticket", "Crescimento MoM"))
            fig.add_trace(go.Scatter(x=serie["data"], y=serie["faturamento"],
                                     mode="lines+markers", name="Faturamento"), row=1, col=1)
            fig.add_trace(go.Scatter(x=serie["data"], y=serie["mm3"], name="MM3", line=dict(dash="dot")), row=1, col=1)
            fig.add_trace(go.Scatter(x=serie["data"], y=serie["mm6"], name="MM6", line=dict(dash="dash")), row=1, col=1)
            fig.add_trace(go.Bar(x=serie["data"], y=serie["pedidos"], name="Pedidos"), row=1, col=2)
            fig.add_trace(go.Scatter(x=serie["data"], y=serie["ticket"], name="Ticket", mode="lines+markers"), row=2, col=1)
            if len(serie) > 1:
                serie["mom"] = serie["faturamento"].pct_change()
                fig.add_trace(go.Bar(x=serie["data"], y=serie["mom"], name="MoM"), row=2, col=2)
            fig.update_layout(height=600, showlegend=True)
            st.plotly_chart(fig, use_container_width=True)

            c1, c2, c3, c4 = st.columns(4)
            with c1: st.metric("Média", fmt_brl(serie["faturamento"].mean()))
            with c2: st.metric("Mediana", fmt_brl(serie["faturamento"].median()))
            with c3: st.metric("Desvio Padrão", fmt_brl(serie["faturamento"].std()))
            with c4: st.metric("Coef. Var.", fmt_pct(serie["faturamento"].std() / max(1e-9, serie["faturamento"].mean())))

    # Lojas
    with tabs[1]:
        if not df_f.empty:
            part = df_f.groupby("loja", as_index=False)["faturamento"].sum()
            fig_tree = px.treemap(part, path=["loja"], values="faturamento", title="Participação por Loja")
            st.plotly_chart(fig_tree, use_container_width=True)

            eff = df_f.groupby("loja", as_index=False).agg(faturamento=("faturamento", "sum"), pedidos=("pedidos", "sum"))
            eff["ticket"] = eff["faturamento"] / eff["pedidos"].replace(0, np.nan)
            fig_eff = px.scatter(eff, x="pedidos", y="faturamento", size="ticket", color="loja",
                                 title="Eficiência por Loja", size_max=60)
            st.plotly_chart(fig_eff, use_container_width=True)

            # Ranking
            rank = eff.sort_values("faturamento", ascending=False).copy()
            rank["faturamento"] = rank["faturamento"].apply(fmt_brl)
            rank["pedidos"] = rank["pedidos"].apply(fmt_int)
            rank["ticket"] = rank["ticket"].apply(fmt_brl)
            st.dataframe(rank.rename(columns={"loja": "Loja", "faturamento": "Faturamento", "pedidos": "Pedidos", "ticket": "Ticket Médio"}),
                         use_container_width=True, height=300)

            # Heatmap
            hm = df_f.pivot_table(index="loja", columns="periodo", values="faturamento", aggfunc="sum").fillna(0)
            fig_hm = go.Figure(data=go.Heatmap(z=hm.values, x=hm.columns, y=hm.index, colorscale="RdYlGn"))
            fig_hm.update_layout(title="Faturamento Mensal por Loja", height=480)
            st.plotly_chart(fig_hm, use_container_width=True)

    # Avançado
    with tabs[2]:
        serie_all = df_lojas.groupby("data")["faturamento"].sum().sort_index()
        if HAS_STATSMODELS and len(serie_all) >= 24:
            try:
                res = sm.tsa.seasonal_decompose(serie_all.asfreq("MS"), model="additive")
                fig = make_subplots(rows=4, cols=1, shared_xaxes=True,
                                    subplot_titles=("Original", "Tendência", "Sazonalidade", "Resíduos"))
                fig.add_trace(go.Scatter(x=res.observed.index, y=res.observed, name="Original"), 1, 1)
                fig.add_trace(go.Scatter(x=res.trend.index, y=res.trend, name="Tendência"), 2, 1)
                fig.add_trace(go.Scatter(x=res.seasonal.index, y=res.seasonal, name="Sazonalidade"), 3, 1)
                fig.add_trace(go.Scatter(x=res.resid.index, y=res.resid, name="Resíduos", mode="markers"), 4, 1)
                fig.update_layout(height=700, showlegend=False)
                st.plotly_chart(fig, use_container_width=True)
            except Exception as e:  # pragma: no cover
                st.info(f"Decomposição indisponível: {e}")
        else:
            st.caption("(Instale 'statsmodels' e tenha ≥24 meses para ver decomposição)")


def display_top_performers(df: pd.DataFrame, p_ini: str, p_fim: str) -> None:
    st.markdown("### 🏆 Top Performers do Período")
    mask = (df["periodo"] >= p_ini) & (df["periodo"] <= p_fim)
    base = df.loc[mask]
    if base.empty:
        st.info("Sem dados para o período.")
        return
    agg = base.groupby("loja", as_index=False).agg(faturamento=("faturamento", "sum"), pedidos=("pedidos", "sum"))
    agg["ticket"] = agg["faturamento"] / agg["pedidos"].replace(0, np.nan)
    agg = agg.sort_values("faturamento", ascending=False)

    top = agg.head(min(3, len(agg)))
    cols = st.columns(len(top))
    medals = ["🥇", "🥈", "🥉"]
    for i, (_, r) in enumerate(top.iterrows()):
        with cols[i]:
            st.markdown(
                f"""
                <div class='metric-card'>
                    <h3>{medals[i]} {r['loja']}</h3>
                    <p><b>Faturamento:</b> {fmt_brl(r['faturamento'])}</p>
                    <p><b>Pedidos:</b> {fmt_int(r['pedidos'])} · <b>Ticket:</b> {fmt_brl(r['ticket'])}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


def display_predictive(k: Dict[str, Any], enable: bool) -> None:
    st.markdown("### 🔮 Análise Preditiva")
    if not enable:
        st.info("Ative a opção na barra lateral para ver a projeção.")
        return
    serie = k.get("serie_temporal")
    if not HAS_STATSMODELS or serie is None or len(serie) < 12:
        st.info("Requer ≥12 meses e 'statsmodels'.")
        return
    try:
        s = serie.set_index("data")["faturamento"].asfreq("MS")
        model = ExponentialSmoothing(s, seasonal="add", seasonal_periods=12).fit()
        periods = st.slider("Meses de previsão", 3, 12, 6)
        fc = model.forecast(periods)
        future = pd.date_range(s.index[-1] + pd.DateOffset(months=1), periods=periods, freq="MS")
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=s.index, y=s.values, name="Histórico", mode="lines+markers"))
        fig.add_trace(go.Scatter(x=future, y=fc.values, name="Previsão", mode="lines+markers", line=dict(dash="dash")))
        fig.update_layout(title=f"Previsão — próximos {periods} meses", height=420)
        st.plotly_chart(fig, use_container_width=True)
        c1, c2, c3 = st.columns(3)
        with c1: st.metric("Próximo mês", fmt_brl(fc.iloc[0]))
        with c2: st.metric("Média prevista", fmt_brl(fc.mean()))
        with c3: st.metric("Total previsto", fmt_brl(fc.sum()))
    except Exception as e:  # pragma: no cover
        st.warning(f"Não foi possível gerar previsão: {e}")

# =============================================================================
# LAYOUT / SIDEBAR
# =============================================================================

def prepare_filters(df: pd.DataFrame) -> Tuple[str, str, str, List[str], bool]:
    # Logo (anexo)
    logo_path = _first_existing(LOGO_CANDIDATES)
    if logo_path:
        st.sidebar.image(logo_path, use_container_width=True)

    st.sidebar.markdown("### 🎯 Filtros")
    analysis_mode = st.sidebar.selectbox("Modo", ["Geral", "Comparativo", "Preditivo"], index=0)

    periodos = sorted(df["periodo"].dropna().unique().tolist())
    if len(periodos) < 2:
        st.error("Dados insuficientes (precisa de ≥2 meses)")
        st.stop()

    period_type = st.sidebar.selectbox("Período", ["Personalizado", "Últimos 3M", "Últimos 6M", "Último Ano", "YTD"]) 
    if period_type == "Últimos 3M":
        p_fim = periodos[-1]; p_ini = periodos[max(0, len(periodos)-3)]
    elif period_type == "Últimos 6M":
        p_fim = periodos[-1]; p_ini = periodos[max(0, len(periodos)-6)]
    elif period_type == "Último Ano":
        p_fim = periodos[-1]; p_ini = periodos[max(0, len(periodos)-12)]
    elif period_type == "YTD":
        ano = datetime.now().year
        ano_periods = [p for p in periodos if p.startswith(str(ano))]
        p_ini, p_fim = (ano_periods[0], ano_periods[-1]) if ano_periods else (periodos[0], periodos[-1])
    else:
        p_ini, p_fim = st.sidebar.select_slider("Período (AAAA-MM)", options=periodos, value=(periodos[0], periodos[-1]))

    lojas = sorted(df["loja"].dropna().unique().tolist()) if "loja" in df.columns else []
    sel_lojas = st.sidebar.multiselect("Lojas", lojas, default=lojas[:min(5, len(lojas))]) if lojas else []
    if not sel_lojas and lojas:
        sel_lojas = lojas

    show_forecasts = st.sidebar.checkbox("Mostrar previsões (Holt‑Winters)", value=False)

    return analysis_mode, p_ini, p_fim, sel_lojas, show_forecasts

# =============================================================================
# MAIN
# =============================================================================

def main() -> None:
    configure_page()
    df = load_data()

    analysis_mode, p_ini, p_fim, sel_lojas, show_forecasts = prepare_filters(df)

    if sel_lojas:
        df_f = df[(df["periodo"] >= p_ini) & (df["periodo"] <= p_fim) & (df["loja"].isin(sel_lojas))].copy()
        df_lojas = df[df["loja"].isin(sel_lojas)].copy()
    else:
        df_f = df[(df["periodo"] >= p_ini) & (df["periodo"] <= p_fim)].copy()
        df_lojas = df.copy()

    k = compute_kpis(df_f, df_lojas, p_ini, p_fim)

    # Header e KPIs
    display_header(p_ini, p_fim, sel_lojas or ["Todas"], analysis_mode)
    display_kpis(k)
    display_alerts(k)

    # Módulos
    if analysis_mode == "Comparativo":
        st.markdown("### 🔄 Comparativo de Faturamento")
        cur_fat = k["period_sum"]["fat"]; prev_fat = k["prev_period_fat"]
        fig = go.Figure(go.Bar(x=["Atual", "Anterior"], y=[cur_fat, prev_fat], text=[fmt_brl(cur_fat), fmt_brl(prev_fat)], textposition="auto"))
        st.plotly_chart(fig, use_container_width=True)
    elif analysis_mode == "Preditivo":
        display_predictive(k, show_forecasts)

    display_detailed(df_f, df_lojas, k)
    display_top_performers(df, p_ini, p_fim)

    with st.sidebar:
        st.markdown("---")
        st.caption(f"Última atualização: {datetime.now().strftime('%d/%m/%Y %H:%M')}")
        st.caption(f"Registros: {len(df):,}".replace(",", "."))
        if st.button("🔄 Limpar cache"):
            st.cache_data.clear(); st.success("Cache limpo!"); st.experimental_rerun()

    st.success("✅ Dashboard pronto! Usando anexos se presentes (CSV/Logo).")


if __name__ == "__main__":
    main()
