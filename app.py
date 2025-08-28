"""
app.py — Dashboard Inteligente Modularizado (v2.1)
==================================================

Melhorias desta versão:
• Correções de UX na sidebar (todos os controles na própria sidebar)
• Funções mais robustas (sazonalidade e correlação com checagens de variância)
• Tipagem aprimorada e pequenos refinamentos de performance
• Ajustes visuais (cores de crescimento/queda consistentes)
• Previsão com seleção de períodos dentro do modo Preditivo
• Compatibilidade quando "statsmodels" não estiver instalado

Para executar: `streamlit run app.py`.

Dependências: streamlit, pandas, plotly, python-dateutil,
statsmodels (opcional), numpy, scipy.
"""

from __future__ import annotations

import os
import re
import unicodedata
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from scipy.stats import pearsonr

# Tipos
from typing import Optional, Any, Dict, Tuple, List

# Blindamos o import de statsmodels para evitar falhas em deploys sem a dependência
try:
    import statsmodels.api as sm  # type: ignore
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
except ModuleNotFoundError:
    sm = None  # type: ignore
    ExponentialSmoothing = None  # type: ignore


# =============================================================================
# CONFIGURAÇÕES E TEMA
# =============================================================================

def configure_page() -> None:
    """Configura a página e define temas e estilos."""
    st.set_page_config(
        page_title="Dashboard Avançado — Hora do Pastel",
        page_icon="🥟",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            "Get Help": "https://www.streamlit.io/community",
            "Report a bug": "mailto:admin@horadopastel.com",
            "About": "### Dashboard Inteligente v2.1\nDesenvolvido para análise avançada de vendas.",
        },
    )

    # Paleta de cores
    global theme_colors, custom_colors
    theme_colors = {
        "primary": "#FF6B35",
        "secondary": "#004E89",
        "success": "#28A745",
        "warning": "#FFC107",
        "danger": "#DC3545",
        "info": "#17A2B8",
    }
    custom_colors = [
        "#FF6B35",
        "#004E89",
        "#28A745",
        "#FFC107",
        "#DC3545",
        "#17A2B8",
        "#6C757D",
        "#343A40",
    ]

    px.defaults.template = "plotly_white"
    px.defaults.color_discrete_sequence = custom_colors

    # CSS
    st.markdown(
        """
        <style>
            .main-header {
                background: linear-gradient(90deg, #FF6B35, #004E89);
                padding: 1rem; border-radius: 10px; color: white;
                text-align: center; margin-bottom: 2rem;
            }
            .metric-card {
                background: white; padding: 1rem; border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1); text-align: center;
                color: #212529; /* garante contraste em temas escuros */
            }
            .alert-success { background:#d4edda; border:1px solid #c3e6cb; color:#155724; padding:1rem; border-radius:8px; margin:1rem 0; }
            .alert-warning { background:#fff3cd; border:1px solid #ffeaa7; color:#856404; padding:1rem; border-radius:8px; margin:1rem 0; }
            .alert-danger  { background:#f8d7da; border:1px solid #f5c6cb; color:#721c24; padding:1rem; border-radius:8px; margin:1rem 0; }

            /* Pódio */
            .podium-container { display:flex; justify-content:center; align-items:flex-end; gap:0.5rem; margin-top:1rem; }
            .podium-item { flex:1; padding:1rem; border-radius:8px; box-shadow:0 2px 4px rgba(0,0,0,0.1); text-align:center; background:white; color:#212529; }
            .podium-item.first { transform: translateY(-15px); }
        </style>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# FUNÇÕES AUXILIARES
# =============================================================================

def normalize_col(name: str) -> str:
    """Normaliza nomes de colunas (removendo acentos e espaços)."""
    name = name.strip().lower()
    name = "".join(
        c for c in unicodedata.normalize("NFKD", name) if not unicodedata.combining(c)
    )
    return re.sub(r"\s+", "_", name)


def _norm_text(s: str) -> str:
    """Normaliza texto para comparação (removendo acentos e caracteres especiais)."""
    s = "".join(
        c for c in unicodedata.normalize("NFKD", str(s).strip().lower()) if not unicodedata.combining(c)
    )
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return " ".join(s.split())


def br_to_float(series: pd.Series) -> pd.Series:
    """Converte strings monetárias brasileiras para float."""
    s = series.astype(str).str.strip()
    s = s.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    s = s.str.replace(r"[^0-9,\.\-]", "", regex=True)
    has_comma = s.str.contains(",", na=False)
    s = s.mask(has_comma, s.str.replace(".", "", regex=False))
    s = s.mask(has_comma, s.str.replace(",", ".", regex=False))
    return pd.to_numeric(s, errors="coerce")


def month_to_int(series: pd.Series) -> pd.Series:
    """Mapeia nomes de meses (português) para inteiros."""
    mapa = {
        "jan": 1, "janeiro": 1, "fev": 2, "fevereiro": 2, "mar": 3, "marco": 3, "março": 3,
        "abr": 4, "abril": 4, "mai": 5, "maio": 5, "jun": 6, "junho": 6, "jul": 7, "julho": 7,
        "ago": 8, "agosto": 8, "set": 9, "setembro": 9, "sep": 9, "out": 10, "outubro": 10,
        "nov": 11, "novembro": 11, "dez": 12, "dezembro": 12,
    }
    s = series.astype(str).str.strip().str.lower().map(lambda x: mapa.get(x, x))
    return pd.to_numeric(s, errors="coerce").astype("Int64")


# Aliases para renomear colunas com termos equivalentes
ALIASES: Dict[str, List[str]] = {
    "mes": ["mes", "mês", "month"],
    "ano": ["ano", "year"],
    "loja": ["loja", "filial", "store"],
    "faturamento": ["faturamento", "receita", "vendas", "valor", "total", "valor_total"],
    "pedidos": ["pedidos", "qtde_pedidos", "qtd_pedidos", "qtd", "quantidade_pedidos"],
    "ticket": ["ticket", "ticket_medio", "ticket_médio", "ticket medio", "ticket médio"],
}


def rename_by_alias(cols: List[str]) -> Dict[str, str]:
    """Gera um dicionário de renome baseado em aliases pré-definidos."""
    ren: Dict[str, str] = {}
    for c in cols:
        for target, opts in ALIASES.items():
            if c in opts:
                ren[c] = target
                break
    return ren


def safe_div(a: float | int | None, b: float | int | None) -> float:
    """Retorna a divisão segura, evitando divisões por zero ou nulos."""
    try:
        if b in (0, None) or pd.isna(b):
            return 0.0
        return float(a) / float(b) if a not in (None, pd.NA) else 0.0
    except Exception:
        return 0.0


def fmt_brl(v: Any) -> str:
    """Formata um número como moeda brasileira."""
    if pd.isna(v):
        return "R$ 0,00"
    s = f"{float(v):,.2f}"
    return "R$ " + s.replace(",", "X").replace(".", ",").replace("X", ".")


def fmt_int(v: Any) -> str:
    """Formata um inteiro com separador de milhares."""
    try:
        return f"{int(v):,}".replace(",", ".")
    except Exception:
        return "0"


def fmt_pct(v: Optional[float], decimals: int = 1) -> str:
    """Formata um número como percentual."""
    if v is None or pd.isna(v):
        return "0,0%"
    return f"{v * 100:,.{decimals}f}%".replace(".", ",")


def calculate_growth_rate(df_series: pd.Series) -> float:
    """Calcula taxa de crescimento mensal médio."""
    if len(df_series) < 2:
        return 0.0
    first_val = df_series.iloc[0]
    last_val = df_series.iloc[-1]
    months = len(df_series) - 1
    if first_val <= 0:
        return 0.0
    try:
        return (last_val / first_val) ** (1 / months) - 1
    except Exception:
        return 0.0


def calculate_volatility(df_series: pd.Series) -> float:
    """Calcula a volatilidade como o desvio padrão das variações percentuais."""
    if len(df_series) < 2:
        return 0.0
    pct_changes = df_series.pct_change().dropna()
    return float(pct_changes.std()) if not pct_changes.empty else 0.0


def detect_seasonality(series: pd.Series) -> str:
    """Classifica a sazonalidade via coeficiente de variação por mês (robusto)."""
    if len(series) < 12:
        return "Dados insuficientes"
    # series esperada com DatetimeIndex
    try:
        s = series.copy()
        s.index = pd.to_datetime(s.index)
        monthly_avg = s.groupby(s.index.month).mean()
        cv = monthly_avg.std() / monthly_avg.mean() if monthly_avg.mean() != 0 else 0.0
        if cv > 0.20:
            return "Alta sazonalidade"
        elif cv > 0.10:
            return "Sazonalidade moderada"
        return "Baixa sazonalidade"
    except Exception:
        return "Indeterminado"


# =============================================================================
# CARGA E TRATAMENTO DE DADOS
# =============================================================================

@st.cache_data(ttl=3600, max_entries=8, show_spinner=False)
def load_data() -> pd.DataFrame:
    """Carrega e pré-processa dados de faturamento. Gera dados de exemplo se necessário."""
    def _finalize(df: pd.DataFrame) -> pd.DataFrame:
        # Conversões finais e criação de coluna de data/periodo
        for c in ["mes", "ano", "pedidos"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
        for c in ["faturamento", "ticket"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        if "loja" in df.columns:
            df["loja"] = df["loja"].astype(str).str.strip()
        mask = df.get("ano").notna() & df.get("mes").notna()
        df["data"] = pd.NaT
        df.loc[mask, "data"] = pd.to_datetime(
            {"year": df.loc[mask, "ano"].astype(int), "month": df.loc[mask, "mes"].astype(int), "day": 1},
            errors="coerce",
        )
        if "periodo" not in df.columns:
            df["periodo"] = pd.to_datetime(df["data"]).dt.to_period("M").astype(str)
        return df.dropna(subset=["data"]).copy()

    # Preferimos arquivo tratado, mas garantimos colunas essenciais
    if os.path.exists("Faturamento_tratado.csv"):
        df = pd.read_csv("Faturamento_tratado.csv")
        base_cols = {"mes", "ano", "loja", "faturamento", "pedidos", "ticket"}
        if not base_cols.issubset(set(map(normalize_col, df.columns))):
            df.columns = [normalize_col(c) for c in df.columns]
            df = df.rename(columns=rename_by_alias(list(df.columns)))
            for col in base_cols:
                if col not in df.columns:
                    df[col] = pd.NA
        return _finalize(df)

    # Arquivo bruto
    if os.path.exists("Faturamento.csv"):
        df = pd.read_csv("Faturamento.csv", sep=None, engine="python")
        df.columns = [normalize_col(c) for c in df.columns]
        df = df.loc[:, ~df.columns.duplicated()].dropna(axis=1, how="all")
        df = df.rename(columns=rename_by_alias(list(df.columns)))
        for col in ["mes", "ano", "loja", "faturamento", "pedidos", "ticket"]:
            if col not in df.columns:
                df[col] = pd.NA
        df["mes"] = month_to_int(df["mes"])
        df["ano"] = pd.to_numeric(df["ano"], errors="coerce").astype("Int64")
        df["faturamento"] = br_to_float(df["faturamento"])
        df["ticket"] = br_to_float(df["ticket"])
        df["pedidos"] = pd.to_numeric(df["pedidos"], errors="coerce").round().astype("Int64")
        return _finalize(df)

    st.warning("Arquivo 'Faturamento.csv' não encontrado. Usando dados de exemplo.")
    return generate_sample_data()


@st.cache_data(ttl=3600, show_spinner=False)
def generate_sample_data() -> pd.DataFrame:
    """Gera dados sintéticos de exemplo para demonstração."""
    np.random.seed(42)
    lojas = ["Centro", "Shopping A", "Shopping B", "Bairro Norte", "Bairro Sul"]
    start_date = datetime(2022, 1, 1)
    dates = pd.date_range(start_date, periods=30, freq="M")
    data: List[Dict[str, Any]] = []
    for date in dates:
        for loja in lojas:
            base_faturamento = 50000 + np.random.normal(0, 5000)
            seasonal_factor = 1 + 0.2 * np.sin(2 * np.pi * date.month / 12)
            trend_factor = 1 + 0.02 * ((date.year - 2022) * 12 + date.month - 1)
            faturamento = base_faturamento * seasonal_factor * trend_factor
            pedidos = int(max(1, faturamento / max(1, (25 + np.random.normal(0, 5)))))
            ticket = faturamento / pedidos if pedidos > 0 else 0.0
            data.append(
                {
                    "mes": date.month,
                    "ano": date.year,
                    "loja": loja,
                    "faturamento": faturamento,
                    "pedidos": pedidos,
                    "ticket": ticket,
                    "data": date,
                    "periodo": date.strftime("%Y-%m"),
                }
            )
    return pd.DataFrame(data)


# =============================================================================
# FILTROS E SIDEBAR
# =============================================================================

def prepare_filters(df: pd.DataFrame) -> Tuple[str, str, str, List[str], bool, bool, bool]:
    """Cria os filtros na sidebar e retorna as seleções do usuário."""
    # Logo
    if os.path.exists("logo.png"):
        st.sidebar.image("logo.png", use_container_width=True)

    st.sidebar.markdown("### 🎯 Filtros Avançados")

    analysis_mode = st.sidebar.selectbox(
        "Modo de Análise",
        ["Padrão", "Comparativo", "Preditivo", "Detalhado"],
        help="Escolha o tipo de análise desejada",
    )

    periodos = sorted(p for p in df["periodo"].dropna().unique().tolist())
    if len(periodos) < 2:
        st.error("Dados insuficientes para análise. São necessários pelo menos 2 meses de dados.")
        st.stop()

    # Tipo de período (agora na sidebar corretamente)
    period_type = st.sidebar.selectbox(
        "Tipo de Período",
        ["Personalizado", "Últimos 3M", "Últimos 6M", "Último Ano", "YTD"],
    )

    # Determina intervalo
    if period_type == "Últimos 3M":
        periodo_fim = periodos[-1]
        periodo_ini = periodos[max(0, len(periodos) - 3)]
    elif period_type == "Últimos 6M":
        periodo_fim = periodos[-1]
        periodo_ini = periodos[max(0, len(periodos) - 6)]
    elif period_type == "Último Ano":
        periodo_fim = periodos[-1]
        periodo_ini = periodos[max(0, len(periodos) - 12)]
    elif period_type == "YTD":
        current_year = datetime.now().year
        ytd_periods = [p for p in periodos if p.startswith(str(current_year))]
        if ytd_periods:
            periodo_ini, periodo_fim = ytd_periods[0], ytd_periods[-1]
        else:
            periodo_ini, periodo_fim = periodos[0], periodos[-1]
    else:
        periodo_ini, periodo_fim = st.sidebar.select_slider(
            "Período (AAAA‑MM)", options=periodos, value=(periodos[0], periodos[-1])
        )

    # Seleção de lojas
    st.sidebar.markdown("#### 🏪 Seleção de Lojas")
    GROUPS = {
        "BGPF": [
            "Caxias do Sul",
            "Bento Goncalves",
            "Novo Hamburgo",
            "Sao leopoldo",
            "Canoas",
            "Protásio Alves",
            "Floresta",
            "Barra Shopping",
        ],
        "Ismael": ["Montenegro", "Lajeado"],
    }

    selection_modes = ["Todas", "Manual", "Por Grupo", "Top Performers", "Personalizadas"]
    selection_mode = st.sidebar.radio("Modo de Seleção", selection_modes)

    lojas = sorted(df["loja"].dropna().unique().tolist())
    map_norm_to_loja = {_norm_text(l): l for l in lojas}

    if selection_mode == "Todas":
        sel_lojas = lojas
    elif selection_mode == "Manual":
        sel_lojas = st.sidebar.multiselect("Escolha as lojas:", lojas, default=lojas[:3])
    elif selection_mode == "Por Grupo":
        group_name = st.sidebar.selectbox("Escolha o grupo:", list(GROUPS.keys()))
        candidatos = [_norm_text(x) for x in GROUPS.get(group_name, [])]
        sel_lojas = [map_norm_to_loja[c] for c in candidatos if c in map_norm_to_loja]
    elif selection_mode == "Top Performers":
        n_top = st.sidebar.slider("Quantas top lojas?", 3, len(lojas), min(5, len(lojas)))
        top_lojas = (
            df.groupby("loja")["faturamento"].sum().sort_values(ascending=False).head(n_top).index.tolist()
        )
        sel_lojas = top_lojas
    else:
        min_faturamento = st.sidebar.number_input("Faturamento mínimo (R$)", 0, 1_000_000, 0)
        min_pedidos = st.sidebar.number_input("Pedidos mínimos", 0, 10_000, 0)
        loja_perf = df.groupby("loja").agg({"faturamento": "sum", "pedidos": "sum"}).reset_index()
        filtered = loja_perf[
            (loja_perf["faturamento"] >= min_faturamento) & (loja_perf["pedidos"] >= min_pedidos)
        ]["loja"].tolist()
        sel_lojas = st.sidebar.multiselect("Lojas que atendem critérios:", filtered, default=filtered)

    if not sel_lojas:
        sel_lojas = lojas[:3]

    # Filtros adicionais (reservados para expansão)
    st.sidebar.markdown("#### ⚙️ Filtros Adicionais")
    include_weekends = st.sidebar.checkbox("Incluir finais de semana", value=True)
    show_trends = st.sidebar.checkbox("Mostrar linhas de tendência", value=True)
    show_forecasts = st.sidebar.checkbox("Mostrar previsões", value=False)

    return (
        analysis_mode,
        periodo_ini,
        periodo_fim,
        sel_lojas,
        include_weekends,
        show_trends,
        show_forecasts,
    )


# =============================================================================
# PROCESSAMENTO DE DADOS E KPIs
# =============================================================================

def filter_data(
    df: pd.DataFrame, periodo_ini: str, periodo_fim: str, sel_lojas: List[str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Aplica filtros de período e lojas aos dados."""
    mask = (df["periodo"] >= periodo_ini) & (df["periodo"] <= periodo_fim) & df["loja"].isin(sel_lojas)
    df_f = df.loc[mask].copy()
    df_lojas = df[df["loja"].isin(sel_lojas)].copy()
    return df_f, df_lojas


def delta(cur_v: float | int | None, base_v: float | int | None) -> Optional[float]:
    """Calcula a variação percentual (delta) entre dois valores."""
    if cur_v is None or base_v in (None, 0) or pd.isna(base_v):
        return None
    return safe_div((float(cur_v) - float(base_v)), float(base_v))


@st.cache_data(show_spinner=False)
def compute_kpis(df_range: pd.DataFrame, df_comp: pd.DataFrame, p_ini: str, p_fim: str) -> Dict[str, Any]:
    """Computa KPIs básicos e avançados para um intervalo e um conjunto de comparação."""
    # Totais e ticket médio
    tot_fat = float(df_range["faturamento"].sum())
    tot_ped = int(df_range["pedidos"].sum()) if df_range["pedidos"].notna().any() else 0
    tik_med = safe_div(tot_fat, tot_ped)

    # Série temporal agregada por data (todas as lojas selecionadas)
    serie_comp = (
        df_comp.dropna(subset=["data"]).groupby("data", as_index=False).agg(
            faturamento=("faturamento", "sum"), pedidos=("pedidos", "sum")
        ).sort_values("data")
    )

    advanced_metrics: Dict[str, Any] = {}
    if not serie_comp.empty and len(serie_comp) > 1:
        # Crescimento & Volatilidade
        advanced_metrics["growth_rate"] = calculate_growth_rate(serie_comp["faturamento"]) if len(serie_comp) > 2 else 0.0
        advanced_metrics["volatility"] = calculate_volatility(serie_comp["faturamento"]) if len(serie_comp) > 2 else 0.0
        # Sazonalidade
        advanced_metrics["seasonality"] = detect_seasonality(
            serie_comp.set_index("data")["faturamento"]
        )
        # Correlação pedidos x faturamento (somente se variância > 0)
        x, y = serie_comp["pedidos"], serie_comp["faturamento"]
        if len(serie_comp) > 3 and np.nanstd(x) > 0 and np.nanstd(y) > 0:
            corr_coef, p_value = pearsonr(x, y)
            advanced_metrics["correlation"] = float(corr_coef)
            advanced_metrics["correlation_pvalue"] = float(p_value)
        # ROI aproximado (20% de margem)
        advanced_metrics["estimated_roi"] = tot_fat * 0.20
        # Eficiência (ticket atual vs histórico)
        historical_ticket = safe_div(serie_comp["faturamento"].sum(), serie_comp["pedidos"].sum())
        current_efficiency = safe_div(tik_med, historical_ticket) - 1
        advanced_metrics["efficiency"] = current_efficiency

    # Comparações temporais (período anterior)
    start_date = pd.to_datetime(p_ini)
    end_date = pd.to_datetime(p_fim)
    num_months = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month) + 1

    prev_end_date = start_date - relativedelta(months=1)
    prev_start_date = prev_end_date - relativedelta(months=num_months - 1)
    mask_prev = (df_comp["data"] >= prev_start_date) & (df_comp["data"] <= prev_end_date)
    df_prev_period = df_comp[mask_prev]
    prev_fat = float(df_prev_period["faturamento"].sum())
    delta_period_fat = delta(tot_fat, prev_fat)

    # YoY
    yoy_start_date = start_date - relativedelta(years=1)
    yoy_end_date = end_date - relativedelta(years=1)
    mask_yoy = (df_comp["data"] >= yoy_start_date) & (df_comp["data"] <= yoy_end_date)
    df_yoy_period = df_comp[mask_yoy]
    yoy_fat = float(df_yoy_period["faturamento"].sum())
    delta_yoy_fat = delta(tot_fat, yoy_fat)

    # MoM
    mom_fat = mom_ped = mom_tik = None
    if len(serie_comp) >= 2:
        last, prev = serie_comp.iloc[-1], serie_comp.iloc[-2]
        mom_fat = delta(last["faturamento"], prev["faturamento"])
        mom_ped = delta(last["pedidos"], prev["pedidos"])
        last_ticket = safe_div(last["faturamento"], last["pedidos"])
        prev_ticket = safe_div(prev["faturamento"], prev["pedidos"])
        mom_tik = delta(last_ticket, prev_ticket)

    return {
        "period_sum": {"fat": tot_fat, "ped": tot_ped, "tik": tik_med},
        "prev_period_fat": prev_fat,
        "delta_period_fat": delta_period_fat,
        "delta_yoy_fat": delta_yoy_fat,
        "yoy_fat_abs": yoy_fat,
        "mom_fat": mom_fat,
        "mom_ped": mom_ped,
        "mom_tik": mom_tik,
        "advanced": advanced_metrics,
        "serie_temporal": serie_comp,
    }


# =============================================================================
# COMPONENTES DE INTERFACE
# =============================================================================

def display_header(periodo_ini: str, periodo_fim: str, sel_lojas: List[str], analysis_mode: str) -> None:
    st.markdown(
        f"""
        <div class="main-header">
            <h1>🥟 Dashboard Inteligente — Hora do Pastel</h1>
            <p>Análise Avançada de Performance e Insights Automáticos</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    info_col1, info_col2, info_col3 = st.columns(3)
    with info_col1:
        st.info(f"📅 **Período:** {periodo_ini} a {periodo_fim}")
    with info_col2:
        st.info(f"🏪 **Lojas:** {len(sel_lojas)} selecionadas")
    with info_col3:
        st.info(f"📊 **Modo:** {analysis_mode}")


def display_kpi_panel(k: Dict[str, Any]) -> None:
    st.markdown("### 📊 Painel de Indicadores")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            label="💰 Faturamento Total",
            value=fmt_brl(k["period_sum"]["fat"]),
            delta=fmt_pct(k.get("delta_period_fat", 0)),
            help=f"Período anterior: {fmt_brl(k['prev_period_fat'])}",
        )
    with col2:
        st.metric(
            label="🛒 Total de Pedidos",
            value=fmt_int(k["period_sum"]["ped"]),
            delta=fmt_pct(k.get("mom_ped", 0)),
            help="Variação MoM do total de pedidos",
        )
    with col3:
        st.metric(
            label="🎯 Ticket Médio",
            value=fmt_brl(k["period_sum"]["tik"]),
            delta=fmt_pct(k.get("mom_tik", 0)),
            help="Variação MoM do ticket médio",
        )
    with col4:
        st.metric(
            label="📈 vs Ano Anterior",
            value=fmt_brl(k["period_sum"]["fat"]),
            delta=fmt_pct(k.get("delta_yoy_fat", 0)),
            help=f"Mesmo período AA: {fmt_brl(k['yoy_fat_abs'])}",
        )

    if k.get("advanced"):
        col5, col6, col7, col8 = st.columns(4)
        growth_rate = k["advanced"].get("growth_rate", 0)
        volatility = k["advanced"].get("volatility", 0)
        efficiency = k["advanced"].get("efficiency", 0)
        estimated_roi = k["advanced"].get("estimated_roi", 0)
        with col5:
            st.metric("📊 Taxa de Crescimento", fmt_pct(growth_rate), help="Crescimento mensal médio")
        with col6:
            st.metric("📉 Volatilidade", fmt_pct(volatility), help="Instabilidade das vendas")
        with col7:
            st.metric("⚡ Eficiência", fmt_pct(efficiency), help="Eficiência vs média histórica")
        with col8:
            st.metric("💎 ROI Estimado", fmt_brl(estimated_roi), help="Lucro estimado (margem 20%)")


def display_alerts(k: Dict[str, Any]) -> None:
    st.markdown("### 🚨 Alertas e Insights Automáticos")
    alert_col1, alert_col2 = st.columns(2)
    growth_rate = k.get("advanced", {}).get("growth_rate", 0)
    efficiency = k.get("advanced", {}).get("efficiency", 0)
    volatility = k.get("advanced", {}).get("volatility", 0)
    seasonality = k.get("advanced", {}).get("seasonality", "")

    with alert_col1:
        if growth_rate > 0.05:
            st.markdown(
                f"""
                <div class="alert-success">
                    <h4>🚀 Excelente Crescimento!</h4>
                    <p>Crescimento médio mensal de <strong>{fmt_pct(growth_rate)}</strong> indica performance excepcional.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        elif growth_rate < -0.02:
            st.markdown(
                f"""
                <div class="alert-danger">
                    <h4>⚠️ Atenção: Declínio nas Vendas</h4>
                    <p>Queda média mensal de <strong>{fmt_pct(abs(growth_rate))}</strong> requer análise de estratégias.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="alert-warning">
                    <h4>📊 Crescimento Estável</h4>
                    <p>Crescimento de <strong>{fmt_pct(growth_rate)}</strong> indica estabilidade no período.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )

    with alert_col2:
        if efficiency > 0.10:
            st.markdown(
                f"""
                <div class="alert-success">
                    <h4>⚡ Alta Eficiência</h4>
                    <p>Performance <strong>{fmt_pct(efficiency)}</strong> acima da média histórica!</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        elif volatility > 0.30:
            st.markdown(
                f"""
                <div class="alert-warning">
                    <h4>📈 Alta Volatilidade</h4>
                    <p>Vendas com variação de <strong>{fmt_pct(volatility)}</strong> — considere estratégias de estabilização.</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.markdown(
                f"""
                <div class="alert-success">
                    <h4>🎯 Performance Consistente</h4>
                    <p>Padrão sazonal: <strong>{seasonality}</strong></p>
                </div>
                """,
                unsafe_allow_html=True,
            )


def display_comparative_analysis(k: Dict[str, Any], df_f: pd.DataFrame) -> None:
    st.markdown("### 🔄 Análise Comparativa Detalhada")
    comp_col1, comp_col2 = st.columns(2)

    current_fat = k["period_sum"]["fat"]
    prev_fat = k["prev_period_fat"]

    with comp_col1:
        fig_comp = go.Figure()
        fig_comp.add_trace(
            go.Bar(
                x=["Período Atual", "Período Anterior"],
                y=[current_fat, prev_fat],
                marker_color=[theme_colors["primary"], theme_colors["secondary"]],
                text=[fmt_brl(current_fat), fmt_brl(prev_fat)],
                textposition="auto",
            )
        )
        fig_comp.update_layout(title="Comparação de Faturamento", height=300)
        st.plotly_chart(fig_comp, use_container_width=True)

    with comp_col2:
        correlation = k.get("advanced", {}).get("correlation", 0) or 0
        fig_corr = go.Figure(
            go.Indicator(
                mode="gauge+number",
                value=abs(correlation) * 100,
                title={"text": "Correlação Pedidos × Faturamento"},
                gauge={
                    "axis": {"range": [0, 100]},
                    "bar": {"color": theme_colors["primary"]},
                    "steps": [
                        {"range": [0, 50], "color": "#E9ECEF"},
                        {"range": [50, 80], "color": "#CED4DA"},
                        {"range": [80, 100], "color": "#ADB5BD"},
                    ],
                },
            )
        )
        fig_corr.update_layout(height=300)
        st.plotly_chart(fig_corr, use_container_width=True)


def display_predictive_analysis(k: Dict[str, Any], show_forecasts: bool) -> None:
    st.markdown("### 🔮 Análise Preditiva")
    if not show_forecasts:
        st.info("Ative a opção 'Mostrar previsões' para ver a projeção de faturamento.")
        return

    serie_temporal = k.get("serie_temporal")
    if sm is None or ExponentialSmoothing is None or serie_temporal is None or len(serie_temporal) < 12:
        st.info("Previsões requerem pelo menos 12 meses de dados e a biblioteca statsmodels.")
        return

    # Controles locais para previsão
    forecast_periods = st.slider("Períodos de Previsão (meses)", 3, 12, 6, key="forecast_periods_slider")
    seasonal_mode = st.selectbox("Sazonalidade", ["aditiva", "multiplicativa"], index=0)

    try:
        serie_forecast = serie_temporal.set_index("data")["faturamento"]
        serie_forecast.index = pd.to_datetime(serie_forecast.index)
        serie_forecast = serie_forecast.asfreq("MS")

        model = ExponentialSmoothing(
            serie_forecast,
            seasonal="add" if seasonal_mode == "aditiva" else "mul",
            seasonal_periods=12,
        ).fit()
        forecast = model.forecast(int(forecast_periods))
        forecast_dates = pd.date_range(
            start=serie_forecast.index[-1] + pd.DateOffset(months=1), periods=int(forecast_periods), freq="MS"
        )

        fig_pred = go.Figure()
        fig_pred.add_trace(
            go.Scatter(
                x=serie_forecast.index,
                y=serie_forecast.values,
                mode="lines+markers",
                name="Histórico",
                line=dict(color=theme_colors["primary"]),
            )
        )
        fig_pred.add_trace(
            go.Scatter(
                x=forecast_dates,
                y=forecast.values,
                mode="lines+markers",
                name="Previsão",
                line=dict(color=theme_colors["warning"], dash="dash"),
            )
        )
        fig_pred.update_layout(
            title=f"Previsão de Faturamento - Próximos {int(forecast_periods)} Meses",
            xaxis_title="Data",
            yaxis_title="Faturamento (R$)",
            height=400,
        )
        st.plotly_chart(fig_pred, use_container_width=True)

        # Métricas de previsão
        pred_col1, pred_col2, pred_col3 = st.columns(3)
        with pred_col1:
            st.metric("Previsão Próximo Mês", fmt_brl(forecast.iloc[0]), help="Modelo Holt‑Winters")
        with pred_col2:
            st.metric("Média Prevista", fmt_brl(forecast.mean()))
        with pred_col3:
            st.metric("Total Previsto", fmt_brl(forecast.sum()))
    except Exception as e:
        st.warning(f"Não foi possível gerar previsões: {e}")


def display_detailed_analysis(df_f: pd.DataFrame, df_lojas: pd.DataFrame, k: Dict[str, Any]) -> None:
    st.markdown("### 📈 Análise Detalhada")
    tabs = st.tabs(["📊 Evolução Temporal", "🏪 Performance por Loja", "🔬 Análise Avançada", "🎯 Benchmarking", "📱 Mobile Dashboard"])

    # Evolução temporal
    with tabs[0]:
        st.markdown("#### Evolução dos Indicadores no Período")
        serie_f = (
            df_f.dropna(subset=["data"]).groupby("data", as_index=False).agg(
                faturamento=("faturamento", "sum"), pedidos=("pedidos", "sum")
            ).sort_values("data")
        )
        if not serie_f.empty:
            serie_f["ticket_medio"] = serie_f.apply(lambda r: safe_div(r["faturamento"], r["pedidos"]), axis=1)
            serie_f["faturamento_mm3"] = serie_f["faturamento"].rolling(window=3, min_periods=1).mean()
            serie_f["faturamento_mm6"] = serie_f["faturamento"].rolling(window=6, min_periods=1).mean()

            fig_evolution = make_subplots(
                rows=2,
                cols=2,
                subplot_titles=("Faturamento e Médias Móveis", "Pedidos", "Ticket Médio", "Crescimento MoM"),
                specs=[[{"secondary_y": True}, {"secondary_y": False}], [{"secondary_y": False}, {"secondary_y": False}]],
            )
            # Faturamento e médias móveis
            fig_evolution.add_trace(
                go.Scatter(
                    x=serie_f["data"],
                    y=serie_f["faturamento"],
                    name="Faturamento",
                    mode="lines+markers",
                    line=dict(width=3, color=theme_colors["primary"]),
                ),
                row=1,
                col=1,
            )
            fig_evolution.add_trace(
                go.Scatter(
                    x=serie_f["data"],
                    y=serie_f["faturamento_mm3"],
                    name="MM 3M",
                    mode="lines",
                    line=dict(width=2, dash="dot", color=theme_colors["info"]),
                ),
                row=1,
                col=1,
            )
            fig_evolution.add_trace(
                go.Scatter(
                    x=serie_f["data"],
                    y=serie_f["faturamento_mm6"],
                    name="MM 6M",
                    mode="lines",
                    line=dict(width=2, dash="dash", color=theme_colors["success"]),
                ),
                row=1,
                col=1,
            )
            # Pedidos
            fig_evolution.add_trace(
                go.Bar(x=serie_f["data"], y=serie_f["pedidos"], name="Pedidos", marker_color=theme_colors["secondary"]),
                row=1,
                col=2,
            )
            # Ticket médio
            fig_evolution.add_trace(
                go.Scatter(
                    x=serie_f["data"],
                    y=serie_f["ticket_medio"],
                    name="Ticket Médio",
                    mode="lines+markers",
                    line=dict(width=2, color=theme_colors["warning"]),
                ),
                row=2,
                col=1,
            )
            # Crescimento MoM
            if len(serie_f) > 1:
                serie_f["growth_mom"] = serie_f["faturamento"].pct_change()
                fig_evolution.add_trace(
                    go.Bar(
                        x=serie_f["data"],
                        y=serie_f["growth_mom"],
                        name="Crescimento MoM",
                        marker_color=[
                            theme_colors["danger"] if x < 0 else theme_colors["success"] for x in serie_f["growth_mom"]
                        ],
                    ),
                    row=2,
                    col=2,
                )
            fig_evolution.update_layout(height=600, showlegend=True)
            st.plotly_chart(fig_evolution, use_container_width=True)

            # Estatísticas resumo
            stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
            with stats_col1:
                st.metric("Média Mensal", fmt_brl(serie_f["faturamento"].mean()))
            with stats_col2:
                st.metric("Mediana", fmt_brl(serie_f["faturamento"].median()))
            with stats_col3:
                st.metric("Desvio Padrão", fmt_brl(serie_f["faturamento"].std()))
            with stats_col4:
                coef_var = serie_f["faturamento"].std() / max(1e-9, serie_f["faturamento"].mean())
                st.metric("Coef. Variação", fmt_pct(coef_var))

    # Performance por loja
    with tabs[1]:
        st.markdown("#### Análise Comparativa entre Lojas")
        if not df_f.empty:
            vis_col1, vis_col2 = st.columns([1, 1])
            with vis_col1:
                st.markdown("**Contribuição no Faturamento (Treemap)**")
                part = df_f.groupby("loja", as_index=False)["faturamento"].sum()
                if not part.empty:
                    fig_tree = px.treemap(
                        part,
                        path=["loja"],
                        values="faturamento",
                        title="Participação de cada loja no faturamento do período",
                        color="faturamento",
                        color_continuous_scale="Viridis",
                    )
                    fig_tree.update_layout(height=400)
                    st.plotly_chart(fig_tree, use_container_width=True)
            with vis_col2:
                st.markdown("**Eficiência (Faturamento vs. Pedidos)**")
                eff = df_f.groupby("loja", as_index=False).agg(
                    faturamento=("faturamento", "sum"), pedidos=("pedidos", "sum")
                )
                if not eff.empty and eff["pedidos"].sum() > 0:
                    eff["ticket"] = eff["faturamento"] / eff["pedidos"].replace(0, np.nan)
                    eff["ticket"].fillna(0.0, inplace=True)
                    fig_eff = px.scatter(
                        eff,
                        x="pedidos",
                        y="faturamento",
                        size="ticket",
                        color="loja",
                        hover_name="loja",
                        size_max=60,
                        title="Eficiência da Loja no Período",
                        color_discrete_sequence=custom_colors,
                    )
                    fig_eff.update_layout(height=400)
                    st.plotly_chart(fig_eff, use_container_width=True)

            # Ranking de lojas (tabela)
            st.markdown("**Ranking de Performance**")
            ranking_data = df_f.groupby("loja", as_index=False).agg({"faturamento": "sum", "pedidos": "sum"})
            ranking_data["ticket_medio"] = ranking_data["faturamento"] / ranking_data["pedidos"].replace(0, np.nan)
            ranking_data["ticket_medio"].fillna(0.0, inplace=True)
            ranking_data["participacao"] = ranking_data["faturamento"] / max(1e-9, ranking_data["faturamento"].sum())
            ranking_data = ranking_data.sort_values("faturamento", ascending=False)

            ranking_display = ranking_data.copy()
            ranking_display["faturamento"] = ranking_display["faturamento"].apply(fmt_brl)
            ranking_display["pedidos"] = ranking_display["pedidos"].apply(fmt_int)
            ranking_display["ticket_medio"] = ranking_display["ticket_medio"].apply(fmt_brl)
            ranking_display["participacao"] = ranking_display["participacao"].apply(lambda x: fmt_pct(x, 2))

            st.dataframe(
                ranking_display.rename(
                    columns={
                        "loja": "Loja",
                        "faturamento": "Faturamento",
                        "pedidos": "Pedidos",
                        "ticket_medio": "Ticket Médio",
                        "participacao": "Participação %",
                    }
                ),
                use_container_width=True,
                height=300,
            )

            # Heatmap
            st.markdown("**Desempenho Mensal por Loja (Heatmap)**")
            heatmap_data = (
                df_f.pivot_table(index="loja", columns="periodo", values="faturamento", aggfunc="sum").fillna(0)
            )
            if not heatmap_data.empty:
                fig_heatmap = go.Figure(
                    data=go.Heatmap(
                        z=heatmap_data.values,
                        x=heatmap_data.columns,
                        y=heatmap_data.index,
                        colorscale="RdYlGn",
                        hoverongaps=False,
                    )
                )
                fig_heatmap.update_layout(title="Faturamento Mensal por Loja", xaxis_nticks=36, height=500)
                st.plotly_chart(fig_heatmap, use_container_width=True)

    # Análise avançada
    with tabs[2]:
        st.markdown("#### Análise de Série Temporal e Correlações")
        serie_all = df_lojas.groupby("data")["faturamento"].sum().sort_index()
        analysis_tabs = st.tabs(["Decomposição", "Correlações", "Distribuições"])

        # Decomposição
        with analysis_tabs[0]:
            if len(serie_all) >= 24 and sm is not None:
                try:
                    s = serie_all.copy()
                    s.index = pd.to_datetime(s.index)
                    res = sm.tsa.seasonal_decompose(s.asfreq("MS"), model="additive")
                    fig_decomp = make_subplots(
                        rows=4,
                        cols=1,
                        shared_xaxes=True,
                        subplot_titles=("Original", "Tendência", "Sazonalidade", "Resíduos"),
                    )
                    fig_decomp.add_trace(
                        go.Scatter(x=res.observed.index, y=res.observed, mode="lines", name="Original", line=dict(color=theme_colors["primary"])),
                        row=1,
                        col=1,
                    )
                    fig_decomp.add_trace(
                        go.Scatter(x=res.trend.index, y=res.trend, mode="lines", name="Tendência", line=dict(color=theme_colors["success"])),
                        row=2,
                        col=1,
                    )
                    fig_decomp.add_trace(
                        go.Scatter(x=res.seasonal.index, y=res.seasonal, mode="lines", name="Sazonalidade", line=dict(color=theme_colors["warning"])),
                        row=3,
                        col=1,
                    )
                    fig_decomp.add_trace(
                        go.Scatter(x=res.resid.index, y=res.resid, mode="markers", name="Resíduos", marker=dict(color=theme_colors["info"])),
                        row=4,
                        col=1,
                    )
                    fig_decomp.update_layout(height=700, showlegend=False)
                    st.plotly_chart(fig_decomp, use_container_width=True)
                    st.markdown(
                        """
                        **Interpretação da Decomposição:**
                        - **Tendência**: direção geral do faturamento (crescimento/declínio)
                        - **Sazonalidade**: padrões recorrentes (ex.: mensais/anuais)
                        - **Resíduos**: variações não explicadas pela tendência/sazonalidade
                        """
                    )
                except Exception as e:
                    st.warning(f"Erro na decomposição: {e}")
            else:
                st.info("A decomposição requer pelo menos 24 meses de dados e statsmodels instalado.")

        # Correlações
        with analysis_tabs[1]:
            if len(df_lojas) > 10:
                corr_data = (
                    df_lojas.pivot_table(index="data", columns="loja", values="faturamento", aggfunc="sum").fillna(0)
                )
                if corr_data.shape[1] > 1:
                    correlation_matrix = corr_data.corr()
                    fig_corr_matrix = go.Figure(
                        data=go.Heatmap(
                            z=correlation_matrix.values,
                            x=correlation_matrix.columns,
                            y=correlation_matrix.index,
                            colorscale="RdBu",
                            zmid=0,
                        )
                    )
                    fig_corr_matrix.update_layout(title="Matriz de Correlação entre Lojas", height=500)
                    st.plotly_chart(fig_corr_matrix, use_container_width=True)

                    # Top correlações
                    st.markdown("**Maiores Correlações:**")
                    corr_pairs: List[Dict[str, Any]] = []
                    cols = list(correlation_matrix.columns)
                    for i in range(len(cols)):
                        for j in range(i + 1, len(cols)):
                            corr_pairs.append({"Loja 1": cols[i], "Loja 2": cols[j], "Correlação": correlation_matrix.iloc[i, j]})
                    top_corr = pd.DataFrame(corr_pairs).sort_values("Correlação", ascending=False).head(5)
                    top_corr["Correlação"] = top_corr["Correlação"].apply(lambda x: f"{x:.3f}")
                    st.dataframe(top_corr, use_container_width=True)

        # Distribuições
        with analysis_tabs[2]:
            fig_dist = make_subplots(rows=1, cols=2, subplot_titles=["Distribuição do Faturamento", "Box Plot por Loja"])
            fig_dist.add_trace(
                go.Histogram(x=df_f["faturamento"], nbinsx=30, name="Distribuição", marker_color=theme_colors["primary"]),
                row=1,
                col=1,
            )
            fig_dist.add_trace(
                go.Box(y=df_f["faturamento"], x=df_f["loja"], name="Box Plot", marker_color=theme_colors["secondary"]),
                row=1,
                col=2,
            )
            fig_dist.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_dist, use_container_width=True)

    # Benchmarking
    with tabs[3]:
        st.markdown("#### Benchmarking e Comparações")
        bench_metrics = df_f.groupby("loja").agg({"faturamento": ["sum", "mean", "std"], "pedidos": ["sum", "mean"], "ticket": "mean"}).round(2)
        bench_metrics.columns = ["Fat_Total", "Fat_Médio", "Fat_StdDev", "Ped_Total", "Ped_Médio", "Ticket_Médio"]

        for col in ["Fat_Total", "Fat_Médio", "Ped_Total", "Ticket_Médio"]:
            if col in bench_metrics.columns and bench_metrics[col].max() > 0:
                bench_metrics[f"{col}_Score"] = (bench_metrics[col] / bench_metrics[col].max()) * 100

        score_cols = [c for c in bench_metrics.columns if c.endswith("_Score")]
        if score_cols:
            bench_metrics["Score_Geral"] = bench_metrics[score_cols].mean(axis=1).round(1)
            top_5_lojas = bench_metrics.nlargest(5, "Score_Geral")

            fig_radar = go.Figure()
            theta = ["Faturamento Total", "Faturamento Médio", "Total Pedidos", "Ticket Médio"]
            for idx, (loja, row) in enumerate(top_5_lojas.iterrows()):
                fig_radar.add_trace(
                    go.Scatterpolar(
                        r=[row.get("Fat_Total_Score", 0), row.get("Fat_Médio_Score", 0), row.get("Ped_Total_Score", 0), row.get("Ticket_Médio_Score", 0)],
                        theta=theta,
                        fill="toself",
                        name=loja,
                        marker_color=custom_colors[idx % len(custom_colors)],
                    )
                )
            fig_radar.update_layout(polar=dict(radialaxis=dict(visible=True, range=[0, 100])), showlegend=True, title="Benchmarking - Top 5 Lojas", height=500)
            st.plotly_chart(fig_radar, use_container_width=True)

            st.markdown("**Ranking Geral de Performance:**")
            ranking_display = bench_metrics[["Score_Geral"]].sort_values("Score_Geral", ascending=False)
            ranking_display["Posição"] = range(1, len(ranking_display) + 1)
            ranking_display = ranking_display[["Posição", "Score_Geral"]].reset_index().rename(columns={"index": "Loja", "Score_Geral": "Score"})
            st.dataframe(ranking_display, use_container_width=True)

    # Mobile Dashboard
    with tabs[4]:
        st.markdown("#### Dashboard Mobile (Resumo)")
        mobile_col1, mobile_col2 = st.columns(2)
        with mobile_col1:
            st.metric("💰 Faturamento", fmt_brl(k["period_sum"]["fat"]))
            st.metric("🛒 Pedidos", fmt_int(k["period_sum"]["ped"]))
            st.metric("🎯 Ticket Médio", fmt_brl(k["period_sum"]["tik"]))
        with mobile_col2:
            if k.get("advanced"):
                st.metric("📊 Crescimento", fmt_pct(k["advanced"].get("growth_rate", 0)))
                st.metric("⚡ Eficiência", fmt_pct(k["advanced"].get("efficiency", 0)))
                st.metric("📈 vs AA", fmt_pct(k.get("delta_yoy_fat", 0)))
        if not df_f.empty:
            serie_f = (
                df_f.dropna(subset=["data"]).groupby("data", as_index=False).agg(faturamento=("faturamento", "sum")).sort_values("data")
            )
            fig_mobile = px.line(x=serie_f["data"], y=serie_f["faturamento"], title="Evolução do Faturamento")
            fig_mobile.update_layout(height=300)
            st.plotly_chart(fig_mobile, use_container_width=True)


def display_top_performers(
    df: pd.DataFrame,
    periodo_ini: str,
    periodo_fim: str,
    *,
    top_n: int = 3,
    show_podium: bool = True,
) -> None:
    """
    Exibe:
      1) Ranking Top‑N por PEDIDOS (🥇🥈🥉)
      2) Pódio por métrica (💰 Faturamento | 🛒 Pedidos | 🎫 Ticket)
    """

    st.markdown("### 🏆 Top Performers do Período")

    required_cols = {"periodo", "loja", "pedidos", "faturamento"}
    if not required_cols.issubset(df.columns):
        missing = required_cols - set(df.columns)
        st.warning(f"Faltam colunas para esta seção: {', '.join(sorted(missing))}")
        return

    mask = (df["periodo"] >= periodo_ini) & (df["periodo"] <= periodo_fim)
    base = df.loc[mask, ["loja", "periodo", "pedidos", "faturamento"]].copy()

    if base.empty or base["pedidos"].notna().sum() == 0:
        st.info("Nenhum dado válido para o período selecionado.")
        return

    agg = (
        base.dropna(subset=["loja"]).groupby("loja", as_index=False).agg(
            faturamento=("faturamento", "sum"), pedidos=("pedidos", "sum")
        )
    )
    agg["ticket"] = agg["faturamento"] / agg["pedidos"].replace(0, np.nan)
    agg["ticket"].fillna(0.0, inplace=True)

    # 1) Ranking top‑N por pedidos
    rank = agg.sort_values("pedidos", ascending=False).head(max(1, min(top_n, len(agg))))
    if not rank.empty:
        cols = st.columns(len(rank))
        total_ped = int(agg["pedidos"].sum()) if pd.notna(agg["pedidos"].sum()) else 0
        medals = ["🥇", "🥈", "🥉"]
        for i, (_, row) in enumerate(rank.reset_index(drop=True).iterrows()):
            pct = (row["pedidos"] / total_ped * 100) if total_ped else 0.0
            pos = medals[i] if i < len(medals) else f"{i+1}º"
            with cols[i]:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <h3>{pos} {row['loja']}</h3>
                        <p><strong>{fmt_int(row['pedidos'])}</strong> pedidos</p>
                        <p>{pct:.1f}% do total</p>
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    # 2) Pódio por métrica
    if show_podium and not agg.empty:
        def _top_row(col: str) -> Optional[pd.Series]:
            s = agg[col]
            return agg.loc[s.idxmax()] if s.notna().any() else None

        top_fat_row = _top_row("faturamento")
        top_ped_row = _top_row("pedidos")
        top_tkt_row = _top_row("ticket")

        metrics_data: List[Tuple[str, str, float, str]] = []
        if top_fat_row is not None:
            metrics_data.append(("Faturamento", str(top_fat_row["loja"]), float(top_fat_row["faturamento"]), "💰"))
        if top_ped_row is not None:
            metrics_data.append(("Pedidos", str(top_ped_row["loja"]), float(top_ped_row["pedidos"]), "🛒"))
        if top_tkt_row is not None:
            metrics_data.append(("Ticket", str(top_tkt_row["loja"]), float(top_tkt_row["ticket"]), "🎫"))

        if metrics_data:
            order = {"Faturamento": 1, "Pedidos": 2, "Ticket": 3}
            metrics_data.sort(key=lambda x: order.get(x[0], 99))
            html = ['<div class="podium-container">']
            for metric, loja, valor, icon in metrics_data:
                cls = "podium-item first" if metric == "Faturamento" else "podium-item"
                val_fmt = fmt_brl(valor) if metric in ("Faturamento", "Ticket") else fmt_int(valor)
                html.append(
                    f"""
                    <div class="{cls}">
                        <h4>{icon} {metric}</h4>
                        <p><strong>{loja}</strong></p>
                        <p>{val_fmt}</p>
                    </div>
                    """
                )
            html.append("</div>")
            st.markdown("#### Pódio por Categoria")
            st.markdown("".join(html), unsafe_allow_html=True)


def display_insights(k: Dict[str, Any]) -> List[str]:
    st.markdown("### 💡 Insights e Recomendações")
    insights: List[str] = []
    adv = k.get("advanced", {})
    growth_rate = adv.get("growth_rate", 0)
    volatility = adv.get("volatility", 0)
    efficiency = adv.get("efficiency", 0)
    seasonality = adv.get("seasonality", "")
    correlation = adv.get("correlation", 0) or 0

    if growth_rate > 0.03:
        insights.append("✅ **Crescimento Acelerado**: Considere expandir operações nas lojas top performers.")
    elif growth_rate < -0.02:
        insights.append("⚠️ **Declínio Preocupante**: Revise estratégias de marketing e operações.")

    if volatility > 0.25:
        insights.append("📊 **Alta Volatilidade**: Implemente estratégias de estabilização de demanda.")

    if efficiency > 0.10:
        insights.append("⚡ **Excelente Eficiência**: Modelo operacional pode ser replicado em outras lojas.")
    elif efficiency < -0.10:
        insights.append("🔧 **Baixa Eficiência**: Revisar processos operacionais e treinamento de equipe.")

    if seasonality == "Alta sazonalidade":
        insights.append("📅 **Forte Sazonalidade**: Planeje estoques e promoções baseadas em padrões mensais.")

    if abs(correlation) > 0.80:
        insights.append("🔗 **Alta Correlação**: Estratégias unificadas podem ser eficazes.")
    elif abs(correlation) < 0.50:
        insights.append("🎯 **Baixa Correlação**: Considere estratégias personalizadas por loja.")

    if insights:
        for item in insights:
            st.markdown(item)
    else:
        st.info("📊 Performance estável no período analisado.")
    return insights


# =============================================================================
# FUNÇÃO PRINCIPAL
# =============================================================================

def main() -> None:
    """Função principal que orquestra o fluxo do dashboard."""
    configure_page()
    df = load_data()

    (
        analysis_mode,
        periodo_ini,
        periodo_fim,
        sel_lojas,
        include_weekends,  # reservado
        show_trends,       # reservado
        show_forecasts,
    ) = prepare_filters(df)

    df_f, df_lojas = filter_data(df, periodo_ini, periodo_fim, sel_lojas)
    k = compute_kpis(df_f, df_lojas, periodo_ini, periodo_fim)

    # Cabeçalho
    display_header(periodo_ini, periodo_fim, sel_lojas, analysis_mode)

    # Painel de indicadores
    display_kpi_panel(k)

    # Alertas
    display_alerts(k)

    # Modos específicos
    if analysis_mode == "Comparativo":
        display_comparative_analysis(k, df_f)
    elif analysis_mode == "Preditivo":
        display_predictive_analysis(k, show_forecasts)

    # Análise detalhada (sempre disponível)
    display_detailed_analysis(df_f, df_lojas, k)

    # Top performers
    display_top_performers(df, periodo_ini, periodo_fim)

    # Insights finais
    _ = display_insights(k)

    # Sidebar extra (utilidades)
    with st.sidebar:
        st.markdown("---")
        st.markdown("### ℹ️ Informações")
        st.caption(f"Última atualização: {datetime.now().strftime('%d/%m/%Y %H:%M')}")
        st.caption(f"Total de registros: {len(df):,}")
        periodos = sorted(p for p in df["periodo"].dropna().unique().tolist())
        if periodos:
            st.caption(f"Período dos dados: {periodos[0]} a {periodos[-1]}")

        if st.button("🔄 Limpar Cache"):
            st.cache_data.clear()
            st.success("Cache limpo com sucesso!")
            st.rerun()

        st.markdown("---")
        st.caption(
            "💡 **Dica**: Use os filtros para focar em períodos específicos e lojas de interesse. O modo 'Preditivo' oferece projeções baseadas em série temporal (Holt‑Winters)."
        )

    st.success("✅ Dashboard carregado com sucesso! Explore as abas e modos para insights detalhados.")


if __name__ == "__main__":
    main()
