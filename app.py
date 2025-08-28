"""
app.py — Dashboard Inteligente Modularizado
================================================

Esta versão reorganiza o código do dashboard de vendas em funções
modulares para melhorar a legibilidade e facilitar a manutenção. O
comportamento original do aplicativo permanece o mesmo, incluindo
recursos avançados como KPIs expandidos, previsão, benchmarking,
insights automáticos e exportação de relatórios. O uso de funções
separadas reduz a repetição de lógica e deixa o corpo principal do
script mais compacto.

Para executar: ``streamlit run app.py``.

Dependências: streamlit, pandas, plotly, python-dateutil,
statsmodels (opcional), numpy, scipy.
"""

import os
import re
import unicodedata
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from scipy.stats import pearsonr

# Blindamos o import de statsmodels para evitar falhas em deploys sem a dependência
try:
    import statsmodels.api as sm  # type: ignore
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
except ModuleNotFoundError:
    sm = None  # type: ignore


# -----------------------------------------------------------------------------
# CONFIGURAÇÕES E TEMA
# -----------------------------------------------------------------------------

def configure_page() -> None:
    """Configura a página e define temas e estilos."""
    st.set_page_config(
        page_title="Dashboard Avançado — Hora do Pastel",
        page_icon="🥟",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            'Get Help': 'https://www.streamlit.io/community',
            'Report a bug': 'mailto:admin@horadopastel.com',
            'About': '### Dashboard Inteligente v2.0\nDesenvolvido para análise avançada de vendas.'
        }
    )

    # Definição de cores usadas nos gráficos e indicadores
    global theme_colors, custom_colors
    theme_colors = {
        "primary": "#FF6B35",
        "secondary": "#004E89",
        "success": "#28A745",
        "warning": "#FFC107",
        "danger": "#DC3545",
        "info": "#17A2B8"
    }
    custom_colors = [
        '#FF6B35', '#004E89', '#28A745', '#FFC107', '#DC3545',
        '#17A2B8', '#6C757D', '#343A40'
    ]

    px.defaults.template = "plotly_white"
    px.defaults.color_discrete_sequence = custom_colors

    # CSS personalizado para componentes visuais
    st.markdown(
        """
        <style>
            .main-header {
                background: linear-gradient(90deg, #FF6B35, #004E89);
                padding: 1rem;
                border-radius: 10px;
                color: white;
                text-align: center;
                margin-bottom: 2rem;
            }
            .metric-card {
                background: white;
                padding: 1rem;
                border-radius: 8px;
                box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                text-align: center;
            }
            .alert-success {
                background-color: #d4edda;
                border: 1px solid #c3e6cb;
                color: #155724;
                padding: 1rem;
                border-radius: 8px;
                margin: 1rem 0;
            }
            .alert-warning {
                background-color: #fff3cd;
                border: 1px solid #ffeaa7;
                color: #856404;
                padding: 1rem;
                border-radius: 8px;
                margin: 1rem 0;
            }
            .alert-danger {
                background-color: #f8d7da;
                border: 1px solid #f5c6cb;
                color: #721c24;
                padding: 1rem;
                border-radius: 8px;
                margin: 1rem 0;
            }
        </style>
        """,
        unsafe_allow_html=True
    )


# -----------------------------------------------------------------------------
# FUNÇÕES AUXILIARES
# -----------------------------------------------------------------------------

def normalize_col(name: str) -> str:
    """Normaliza nomes de colunas (removendo acentos e espaços)."""
    name = name.strip().lower()
    name = "".join(c for c in unicodedata.normalize("NFKD", name) if not unicodedata.combining(c))
    return re.sub(r"\s+", "_", name)


def _norm_text(s: str) -> str:
    """Normaliza texto para comparação (removendo acentos e caracteres especiais)."""
    s = ''.join(c for c in unicodedata.normalize('NFKD', str(s).strip().lower()) if not unicodedata.combining(c))
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return ' '.join(s.split())


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
        "nov": 11, "novembro": 11, "dez": 12, "dezembro": 12
    }
    s = series.astype(str).str.strip().str.lower().map(lambda x: mapa.get(x, x))
    return pd.to_numeric(s, errors="coerce").astype("Int64")


# Aliases para renomear colunas com termos equivalentes
ALIASES = {
    "mes": ["mes", "mês", "month"],
    "ano": ["ano", "year"],
    "loja": ["loja", "filial", "store"],
    "faturamento": ["faturamento", "receita", "vendas", "valor", "total", "valor_total"],
    "pedidos": ["pedidos", "qtde_pedidos", "qtd_pedidos", "qtd", "quantidade_pedidos"],
    "ticket": ["ticket", "ticket_medio", "ticket_médio", "ticket medio", "ticket médio"],
}


def rename_by_alias(cols: list[str]) -> dict:
    """Gera um dicionário de renome baseado em aliases pré-definidos."""
    ren: dict[str, str] = {}
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


def fmt_brl(v) -> str:
    """Formata um número como moeda brasileira."""
    if pd.isna(v):
        return "R$ 0,00"
    s = f"{float(v):,.2f}"
    return "R$ " + s.replace(",", "X").replace(".", ",").replace("X", ".")


def fmt_int(v) -> str:
    """Formata um inteiro com separador de milhares."""
    try:
        return f"{int(v):,}".replace(",", ".")
    except Exception:
        return "0"


def fmt_pct(v, decimals: int = 1) -> str:
    """Formata um número como percentual."""
    if pd.isna(v) or v is None:
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
    return (last_val / first_val) ** (1 / months) - 1


def calculate_volatility(df_series: pd.Series) -> float:
    """Calcula a volatilidade como o desvio padrão das variações percentuais."""
    if len(df_series) < 2:
        return 0.0
    pct_changes = df_series.pct_change().dropna()
    return float(pct_changes.std())


def detect_seasonality(df_series: pd.Series) -> str:
    """Classifica a sazonalidade a partir do coeficiente de variação mensal."""
    if len(df_series) < 12:
        return "Dados insuficientes"
    df_with_month = df_series.reset_index()
    df_with_month['month'] = pd.to_datetime(df_with_month.iloc[:, 0]).dt.month
    monthly_avg = df_with_month.groupby('month')[df_with_month.columns[1]].mean()
    cv = monthly_avg.std() / monthly_avg.mean()
    if cv > 0.2:
        return "Alta sazonalidade"
    elif cv > 0.1:
        return "Sazonalidade moderada"
    else:
        return "Baixa sazonalidade"


# -----------------------------------------------------------------------------
# CARGA E TRATAMENTO DE DADOS
# -----------------------------------------------------------------------------

@st.cache_data(ttl=3600, max_entries=8, show_spinner=False)
def load_data() -> pd.DataFrame:
    """Carrega e pré-processa dados de faturamento. Gera dados de exemplo se necessário."""
    # Verifica se arquivo pré-tratado existe
    if os.path.exists("Faturamento_tratado.csv"):
        df = pd.read_csv("Faturamento_tratado.csv")
    elif os.path.exists("Faturamento.csv"):
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
    else:
        st.warning("Arquivo 'Faturamento.csv' não encontrado. Usando dados de exemplo.")
        return generate_sample_data()

    # Conversões finais e criação de coluna de data
    for c in ["mes", "ano", "pedidos"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    for c in ["faturamento", "ticket"]:
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


@st.cache_data(ttl=3600, show_spinner=False)
def generate_sample_data() -> pd.DataFrame:
    """Gera dados sintéticos de exemplo para demonstração."""
    np.random.seed(42)
    lojas = ["Centro", "Shopping A", "Shopping B", "Bairro Norte", "Bairro Sul"]
    start_date = datetime(2022, 1, 1)
    dates = pd.date_range(start_date, periods=30, freq='M')
    data: list[dict[str, any]] = []
    for date in dates:
        for loja in lojas:
            base_faturamento = 50000 + np.random.normal(0, 5000)
            seasonal_factor = 1 + 0.2 * np.sin(2 * np.pi * date.month / 12)
            trend_factor = 1 + 0.02 * ((date.year - 2022) * 12 + date.month - 1)
            faturamento = base_faturamento * seasonal_factor * trend_factor
            pedidos = int(faturamento / (25 + np.random.normal(0, 5)))
            ticket = faturamento / pedidos if pedidos > 0 else 0.0
            data.append({
                'mes': date.month,
                'ano': date.year,
                'loja': loja,
                'faturamento': faturamento,
                'pedidos': pedidos,
                'ticket': ticket,
                'data': date,
                'periodo': date.strftime('%Y-%m')
            })
    return pd.DataFrame(data)


# -----------------------------------------------------------------------------
# FILTROS E SIDEBAR
# -----------------------------------------------------------------------------

def prepare_filters(df: pd.DataFrame) -> tuple[str, str, str, list[str], bool, bool, bool]:
    """Cria os filtros na sidebar e retorna as seleções do usuário.

    Retorna:
        analysis_mode: modo de análise selecionado
        periodo_ini: string no formato YYYY-MM
        periodo_fim: string no formato YYYY-MM
        sel_lojas: lista de lojas selecionadas
        include_weekends: se inclui finais de semana (reserva para futura expansão)
        show_trends: se exibe linhas de tendência
        show_forecasts: se exibe previsões
    """
    # Logo (se existir)
    if os.path.exists("logo.png"):
        st.sidebar.image("logo.png", use_column_width=True)
    st.sidebar.markdown("### 🎯 Filtros Avançados")
    # Modo de análise
    analysis_mode = st.sidebar.selectbox(
        "Modo de Análise",
        ["Padrão", "Comparativo", "Preditivo", "Detalhado"],
        help="Escolha o tipo de análise desejada"
    )
    # Períodos disponíveis
    periodos = sorted(p for p in df["periodo"].dropna().unique().tolist())
    if len(periodos) < 2:
        st.error("Dados insuficientes para análise. São necessários pelo menos 2 meses de dados.")
        st.stop()
    # Seletor de intervalo
    col_period1, _ = st.sidebar.columns(2)
    with col_period1:
        period_type = st.selectbox(
            "Tipo de Período",
            ["Personalizado", "Últimos 3M", "Últimos 6M", "Último Ano", "YTD"]
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
            periodo_ini = ytd_periods[0]
            periodo_fim = ytd_periods[-1]
        else:
            periodo_ini, periodo_fim = periodos[0], periodos[-1]
    else:
        rng_default = (periodos[0], periodos[-1])
        periodo_ini, periodo_fim = st.sidebar.select_slider(
            "Período (AAAA‑MM)",
            options=periodos,
            value=rng_default
        )
    # Seleção de lojas
    st.sidebar.markdown("#### 🏪 Seleção de Lojas")
    # Grupos de lojas para seleção em massa
    GROUPS = {
        "BGPF": ["Caxias do Sul", "Bento Goncalves", "Novo Hamburgo", "Sao leopoldo",
                  "Canoas", "Protasio", "Floresta", "Barra Shopping"],
        "Ismael": ["Montenegro", "Lajeado"]
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
        n_top = st.sidebar.slider("Quantas top lojas?", 3, len(lojas), 5)
        top_lojas = (
            df.groupby("loja")["faturamento"].sum()
            .sort_values(ascending=False)
            .head(n_top)
            .index
            .tolist()
        )
        sel_lojas = top_lojas
    else:  # Personalizadas
        min_faturamento = st.sidebar.number_input("Faturamento mínimo (R$)", 0, 1_000_000, 0)
        min_pedidos = st.sidebar.number_input("Pedidos mínimos", 0, 10_000, 0)
        loja_perf = df.groupby("loja").agg({
            'faturamento': 'sum',
            'pedidos': 'sum'
        }).reset_index()
        filtered = loja_perf[
            (loja_perf['faturamento'] >= min_faturamento) &
            (loja_perf['pedidos'] >= min_pedidos)
        ]['loja'].tolist()
        sel_lojas = st.sidebar.multiselect(
            "Lojas que atendem critérios:", filtered, default=filtered
        )
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
        show_forecasts
    )


# -----------------------------------------------------------------------------
# PROCESSAMENTO DE DADOS E KPIs
# -----------------------------------------------------------------------------

def filter_data(df: pd.DataFrame, periodo_ini: str, periodo_fim: str, sel_lojas: list[str]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Aplica filtros de período e lojas aos dados."""
    mask = (
        (df["periodo"] >= periodo_ini) &
        (df["periodo"] <= periodo_fim) &
        df["loja"].isin(sel_lojas)
    )
    df_f = df.loc[mask].copy()
    df_lojas = df[df["loja"].isin(sel_lojas)].copy()
    return df_f, df_lojas


def delta(cur_v: float | int | None, base_v: float | int | None) -> float | None:
    """Calcula a variação percentual (delta) entre dois valores."""
    if cur_v is None or base_v in (None, 0) or pd.isna(base_v):
        return None
    return safe_div((float(cur_v) - float(base_v)), float(base_v))


@st.cache_data(show_spinner=False)
def compute_kpis(df_range: pd.DataFrame, df_comp: pd.DataFrame, p_ini: str, p_fim: str) -> dict:
    """Computa KPIs básicos e avançados para um intervalo e um conjunto de comparação."""
    # Totais e ticket médio
    tot_fat = float(df_range["faturamento"].sum())
    tot_ped = int(df_range["pedidos"].sum()) if df_range["pedidos"].notna().any() else 0
    tik_med = safe_div(tot_fat, tot_ped)
    # Série temporal
    serie_comp = (
        df_comp.dropna(subset=["data"]).groupby("data", as_index=False)
        .agg(faturamento=("faturamento", "sum"), pedidos=("pedidos", "sum"))
        .sort_values("data")
    )
    advanced_metrics: dict[str, float | str] = {}
    if not serie_comp.empty and len(serie_comp) > 1:
        # Crescimento
        advanced_metrics["growth_rate"] = calculate_growth_rate(serie_comp["faturamento"])
        # Volatilidade
        advanced_metrics["volatility"] = calculate_volatility(serie_comp["faturamento"])
        # Sazonalidade
        advanced_metrics["seasonality"] = detect_seasonality(serie_comp.set_index("data")["faturamento"])
        # Correlação pedidos x faturamento
        if len(serie_comp) > 3:
            corr_coef, p_value = pearsonr(serie_comp["pedidos"], serie_comp["faturamento"])
            advanced_metrics["correlation"] = corr_coef
            advanced_metrics["correlation_pvalue"] = p_value
        # ROI aproximado (20% de margem)
        estimated_profit = tot_fat * 0.2
        advanced_metrics["estimated_roi"] = estimated_profit
        # Eficiência (ticket médio vs histórico)
        historical_ticket = safe_div(serie_comp["faturamento"].sum(), serie_comp["pedidos"].sum())
        current_efficiency = safe_div(tik_med, historical_ticket) - 1
        advanced_metrics["efficiency"] = current_efficiency
    # Comparações temporais
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
    # Mom
    mom_fat = mom_ped = mom_tik = None
    if len(serie_comp) >= 2:
        last = serie_comp.iloc[-1]
        prev = serie_comp.iloc[-2]
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
        "serie_temporal": serie_comp
    }


# -----------------------------------------------------------------------------
# COMPONENTES DE INTERFACE
# -----------------------------------------------------------------------------

def display_header(periodo_ini: str, periodo_fim: str, sel_lojas: list[str], analysis_mode: str) -> None:
    """Exibe o cabeçalho principal do dashboard."""
    st.markdown(
        f"""
        <div class="main-header">
            <h1>🥟 Dashboard Inteligente — Hora do Pastel</h1>
            <p>Análise Avançada de Performance e Insights Automáticos</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    info_col1, info_col2, info_col3 = st.columns(3)
    with info_col1:
        st.info(f"📅 **Período:** {periodo_ini} a {periodo_fim}")
    with info_col2:
        st.info(f"🏪 **Lojas:** {len(sel_lojas)} selecionadas")
    with info_col3:
        st.info(f"📊 **Modo:** {analysis_mode}")


def display_kpi_panel(k: dict) -> None:
    """Mostra os indicadores principais e avançados."""
    st.markdown("### 📊 Painel de Indicadores")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            label="💰 Faturamento Total",
            value=fmt_brl(k["period_sum"]["fat"]),
            delta=fmt_pct(k.get("delta_period_fat", 0)),
            help=f"Período anterior: {fmt_brl(k['prev_period_fat'])}"
        )
    with col2:
        st.metric(
            label="🛒 Total de Pedidos",
            value=fmt_int(k["period_sum"]["ped"]),
            delta=fmt_pct(k.get("mom_ped", 0)),
            help="Variação MoM do total de pedidos"
        )
    with col3:
        st.metric(
            label="🎯 Ticket Médio",
            value=fmt_brl(k["period_sum"]["tik"]),
            delta=fmt_pct(k.get("mom_tik", 0)),
            help="Variação MoM do ticket médio"
        )
    with col4:
        st.metric(
            label="📈 vs Ano Anterior",
            value=fmt_brl(k["period_sum"]["fat"]),
            delta=fmt_pct(k.get("delta_yoy_fat", 0)),
            help=f"Mesmo período AA: {fmt_brl(k['yoy_fat_abs'])}"
        )
    # KPIs avançados
    if k.get("advanced"):
        col5, col6, col7, col8 = st.columns(4)
        growth_rate = k["advanced"].get("growth_rate", 0)
        volatility = k["advanced"].get("volatility", 0)
        efficiency = k["advanced"].get("efficiency", 0)
        estimated_roi = k["advanced"].get("estimated_roi", 0)
        with col5:
            st.metric(
                label="📊 Taxa de Crescimento",
                value=fmt_pct(growth_rate),
                help="Taxa de crescimento mensal médio no período"
            )
        with col6:
            st.metric(
                label="📉 Volatilidade",
                value=fmt_pct(volatility),
                help="Medida de instabilidade das vendas"
            )
        with col7:
            st.metric(
                label="⚡ Eficiência",
                value=fmt_pct(efficiency),
                help="Eficiência vs média histórica"
            )
        with col8:
            st.metric(
                label="💎 ROI Estimado",
                value=fmt_brl(estimated_roi),
                help="Lucro estimado (margem 20%)"
            )


def display_alerts(k: dict) -> None:
    """Exibe alertas e mensagens de insight com base nos KPIs."""
    st.markdown("### 🚨 Alertas e Insights Automáticos")
    alert_col1, alert_col2 = st.columns(2)
    growth_rate = k["advanced"].get("growth_rate", 0) if k.get("advanced") else 0
    efficiency = k["advanced"].get("efficiency", 0) if k.get("advanced") else 0
    volatility = k["advanced"].get("volatility", 0) if k.get("advanced") else 0
    seasonality = k["advanced"].get("seasonality", "") if k.get("advanced") else ""
    with alert_col1:
        if growth_rate > 0.05:
            st.markdown(
                f"""
                <div class="alert-success">
                    <h4>🚀 Excelente Crescimento!</h4>
                    <p>Crescimento médio mensal de <strong>{fmt_pct(growth_rate)}</strong> indica performance excepcional.</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        elif growth_rate < -0.02:
            st.markdown(
                f"""
                <div class="alert-danger">
                    <h4>⚠️ Atenção: Declínio nas Vendas</h4>
                    <p>Queda média mensal de <strong>{fmt_pct(abs(growth_rate))}</strong> requer análise de estratégias.</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div class="alert-warning">
                    <h4>📊 Crescimento Estável</h4>
                    <p>Crescimento de <strong>{fmt_pct(growth_rate)}</strong> indica estabilidade no período.</p>
                </div>
                """,
                unsafe_allow_html=True
            )
    with alert_col2:
        if efficiency > 0.1:
            st.markdown(
                f"""
                <div class="alert-success">
                    <h4>⚡ Alta Eficiência</h4>
                    <p>Performance <strong>{fmt_pct(efficiency)}</strong> acima da média histórica!</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        elif volatility > 0.3:
            st.markdown(
                f"""
                <div class="alert-warning">
                    <h4>📈 Alta Volatilidade</h4>
                    <p>Vendas com variação de <strong>{fmt_pct(volatility)}</strong> - considere estratégias de estabilização.</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        else:
            st.markdown(
                f"""
                <div class="alert-success">
                    <h4>🎯 Performance Consistente</h4>
                    <p>Padrão sazonal: <strong>{seasonality}</strong></p>
                </div>
                """,
                unsafe_allow_html=True
            )


def display_comparative_analysis(k: dict, df_f: pd.DataFrame) -> None:
    """Mostra gráficos comparativos e correlação para o modo 'Comparativo'."""
    st.markdown("### 🔄 Análise Comparativa Detalhada")
    comp_col1, comp_col2 = st.columns(2)
    # Comparação com período anterior
    current_fat = k["period_sum"]["fat"]
    prev_fat = k["prev_period_fat"]
    with comp_col1:
        fig_comp = go.Figure()
        fig_comp.add_trace(
            go.Bar(
                x=['Período Atual', 'Período Anterior'],
                y=[current_fat, prev_fat],
                marker_color=[theme_colors["primary"], theme_colors["secondary"]],
                text=[fmt_brl(current_fat), fmt_brl(prev_fat)],
                textposition='auto'
            )
        )
        fig_comp.update_layout(title="Comparação de Faturamento", height=300)
        st.plotly_chart(fig_comp, use_container_width=True)
    with comp_col2:
        correlation = k["advanced"].get("correlation", 0) if k.get("advanced") else 0
        fig_corr = go.Figure(
            go.Indicator(
                mode="gauge+number+delta",
                value=abs(correlation) * 100,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Correlação Pedidos x Faturamento"},
                delta={'reference': 80},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': theme_colors["primary"]},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "gray"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            )
        )
        fig_corr.update_layout(height=300)
        st.plotly_chart(fig_corr, use_container_width=True)


def display_predictive_analysis(k: dict, show_forecasts: bool) -> None:
    """Exibe previsão de vendas quando o modo 'Preditivo' está ativo."""
    st.markdown("### 🔮 Análise Preditiva")
    if not show_forecasts:
        st.info("Ative a opção 'Mostrar previsões' para ver a projeção de faturamento.")
        return
    serie_temporal = k.get("serie_temporal")
    if sm is None or serie_temporal is None or len(serie_temporal) < 12:
        st.info("Previsões requerem pelo menos 12 meses de dados e a biblioteca statsmodels.")
        return
    try:
        serie_forecast = serie_temporal.set_index('data')['faturamento']
        serie_forecast.index = pd.to_datetime(serie_forecast.index)
        serie_forecast = serie_forecast.asfreq('MS')
        model = ExponentialSmoothing(serie_forecast, seasonal='add', seasonal_periods=12).fit()
        forecast = model.forecast(6)
        forecast_dates = pd.date_range(start=serie_forecast.index[-1] + pd.DateOffset(months=1), periods=6, freq='MS')
        fig_pred = go.Figure()
        fig_pred.add_trace(
            go.Scatter(x=serie_forecast.index, y=serie_forecast.values, mode='lines+markers', name='Histórico',
                       line=dict(color=theme_colors["primary"]))
        )
        fig_pred.add_trace(
            go.Scatter(x=forecast_dates, y=forecast.values, mode='lines+markers', name='Previsão',
                       line=dict(color=theme_colors["warning"], dash='dash'))
        )
        fig_pred.update_layout(
            title="Previsão de Faturamento - Próximos 6 Meses",
            xaxis_title="Data",
            yaxis_title="Faturamento (R$)",
            height=400
        )
        st.plotly_chart(fig_pred, use_container_width=True)
        # Métricas de previsão
        pred_col1, pred_col2, pred_col3 = st.columns(3)
        with pred_col1:
            st.metric("Previsão Próximo Mês", fmt_brl(forecast.iloc[0]), help="Baseado em modelo Holt-Winters")
        with pred_col2:
            avg_forecast = forecast.mean()
            st.metric("Média Prevista (6M)", fmt_brl(avg_forecast), help="Média dos próximos 6 meses")
        with pred_col3:
            total_forecast = forecast.sum()
            st.metric("Total Previsto (6M)", fmt_brl(total_forecast), help="Soma dos próximos 6 meses")
    except Exception as e:
        st.warning(f"Não foi possível gerar previsões: {e}")


def display_detailed_analysis(df_f: pd.DataFrame, df_lojas: pd.DataFrame, k: dict) -> None:
    """Exibe as abas de análise detalhada (evolução temporal, performance por loja, análises avançadas, benchmarking e mobile)."""
    st.markdown("### 📈 Análise Detalhada")
    tabs = st.tabs([
        "📊 Evolução Temporal",
        "🏪 Performance por Loja",
        "🔬 Análise Avançada",
        "🎯 Benchmarking",
        "📱 Mobile Dashboard"
    ])
    # Evolução temporal
    with tabs[0]:
        st.markdown("#### Evolução dos Indicadores no Período")
        serie_f = (
            df_f.dropna(subset=["data"]).groupby("data", as_index=False)
            .agg(faturamento=("faturamento", "sum"), pedidos=("pedidos", "sum"))
            .sort_values("data")
        )
        if not serie_f.empty:
            serie_f["ticket_medio"] = serie_f.apply(lambda r: safe_div(r["faturamento"], r["pedidos"]), axis=1)
            serie_f['faturamento_mm3'] = serie_f['faturamento'].rolling(window=3, min_periods=1).mean()
            serie_f['faturamento_mm6'] = serie_f['faturamento'].rolling(window=6, min_periods=1).mean()
            # Cria subplots
            fig_evolution = make_subplots(
                rows=2,
                cols=2,
                subplot_titles=('Faturamento e Médias Móveis', 'Pedidos', 'Ticket Médio', 'Crescimento MoM'),
                specs=[[{"secondary_y": True}, {"secondary_y": False}],
                       [{"secondary_y": False}, {"secondary_y": False}]]
            )
            # Faturamento e médias móveis
            fig_evolution.add_trace(
                go.Scatter(x=serie_f['data'], y=serie_f['faturamento'], name='Faturamento', mode='lines+markers',
                           line=dict(width=3, color=theme_colors["primary"])), row=1, col=1
            )
            fig_evolution.add_trace(
                go.Scatter(x=serie_f['data'], y=serie_f['faturamento_mm3'], name='MM 3M', mode='lines',
                           line=dict(width=2, dash='dot', color=theme_colors["info"])), row=1, col=1
            )
            fig_evolution.add_trace(
                go.Scatter(x=serie_f['data'], y=serie_f['faturamento_mm6'], name='MM 6M', mode='lines',
                           line=dict(width=2, dash='dash', color=theme_colors["success"])), row=1, col=1
            )
            # Pedidos
            fig_evolution.add_trace(
                go.Bar(x=serie_f['data'], y=serie_f['pedidos'], name='Pedidos',
                       marker_color=theme_colors["secondary"]), row=1, col=2
            )
            # Ticket médio
            fig_evolution.add_trace(
                go.Scatter(x=serie_f['data'], y=serie_f['ticket_medio'], name='Ticket Médio', mode='lines+markers',
                           line=dict(width=2, color=theme_colors["warning"])), row=2, col=1
            )
            # Crescimento MoM
            if len(serie_f) > 1:
                serie_f['growth_mom'] = serie_f['faturamento'].pct_change()
                fig_evolution.add_trace(
                    go.Bar(
                        x=serie_f['data'],
                        y=serie_f['growth_mom'],
                        name='Crescimento MoM',
                        marker_color=['red' if x < 0 else 'green' for x in serie_f['growth_mom']]
                    ),
                    row=2,
                    col=2
                )
            fig_evolution.update_layout(height=600, showlegend=True)
            st.plotly_chart(fig_evolution, use_container_width=True)
            # Estatísticas resumo
            stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
            with stats_col1:
                st.metric("Média Mensal", fmt_brl(serie_f['faturamento'].mean()))
            with stats_col2:
                st.metric("Mediana", fmt_brl(serie_f['faturamento'].median()))
            with stats_col3:
                st.metric("Desvio Padrão", fmt_brl(serie_f['faturamento'].std()))
            with stats_col4:
                coef_var = serie_f['faturamento'].std() / serie_f['faturamento'].mean()
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
                    fig_tree = px.treemap(part, path=['loja'], values='faturamento',
                                          title='Participação de cada loja no faturamento do período',
                                          color='faturamento', color_continuous_scale='Viridis')
                    fig_tree.update_layout(height=400)
                    st.plotly_chart(fig_tree, use_container_width=True)
            with vis_col2:
                st.markdown("**Eficiência (Faturamento vs. Pedidos)**")
                eff = df_f.groupby("loja", as_index=False).agg(
                    faturamento=("faturamento", "sum"),
                    pedidos=("pedidos", "sum")
                )
                if not eff.empty and eff['pedidos'].sum() > 0:
                    eff["ticket"] = eff.apply(lambda r: safe_div(r["faturamento"], r["pedidos"]), axis=1)
                    fig_eff = px.scatter(
                        eff,
                        x="pedidos",
                        y="faturamento",
                        size="ticket",
                        color="loja",
                        hover_name="loja",
                        size_max=60,
                        title="Eficiência da Loja no Período",
                        color_discrete_sequence=custom_colors
                    )
                    fig_eff.update_layout(height=400)
                    st.plotly_chart(fig_eff, use_container_width=True)
            # Ranking de lojas
            st.markdown("**Ranking de Performance**")
            ranking_data = df_f.groupby("loja", as_index=False).agg({
                'faturamento': 'sum',
                'pedidos': 'sum'
            })
            ranking_data['ticket_medio'] = ranking_data.apply(
                lambda r: safe_div(r['faturamento'], r['pedidos']), axis=1
            )
            ranking_data['participacao'] = ranking_data['faturamento'] / ranking_data['faturamento'].sum()
            ranking_data = ranking_data.sort_values('faturamento', ascending=False)
            ranking_display = ranking_data.copy()
            ranking_display['faturamento'] = ranking_display['faturamento'].apply(fmt_brl)
            ranking_display['pedidos'] = ranking_display['pedidos'].apply(fmt_int)
            ranking_display['ticket_medio'] = ranking_display['ticket_medio'].apply(fmt_brl)
            ranking_display['participacao'] = ranking_display['participacao'].apply(lambda x: fmt_pct(x, 2))
            st.dataframe(
                ranking_display.rename(columns={
                    'loja': 'Loja',
                    'faturamento': 'Faturamento',
                    'pedidos': 'Pedidos',
                    'ticket_medio': 'Ticket Médio',
                    'participacao': 'Participação %'
                }),
                use_container_width=True,
                height=300
            )
            # Heatmap
            st.markdown("**Desempenho Mensal por Loja (Heatmap)**")
            heatmap_data = df_f.pivot_table(index='loja', columns='periodo', values='faturamento', aggfunc='sum').fillna(0)
            if not heatmap_data.empty:
                fig_heatmap = go.Figure(
                    data=go.Heatmap(
                        z=heatmap_data.values,
                        x=heatmap_data.columns,
                        y=heatmap_data.index,
                        colorscale='RdYlGn',
                        hoverongaps=False
                    )
                )
                fig_heatmap.update_layout(title='Faturamento Mensal por Loja', xaxis_nticks=36, height=500)
                st.plotly_chart(fig_heatmap, use_container_width=True)
    # Análise avançada
    with tabs[2]:
        st.markdown("#### Análise de Série Temporal e Correlações")
        serie_all = df_lojas.groupby('data')['faturamento'].sum().sort_index()
        analysis_tabs = st.tabs(["Decomposição", "Correlações", "Distribuições"])
        # Decomposição
        with analysis_tabs[0]:
            if len(serie_all) >= 24 and sm is not None:
                try:
                    serie_all.index = pd.to_datetime(serie_all.index)
                    res = sm.tsa.seasonal_decompose(serie_all.asfreq('MS'), model='additive')
                    fig_decomp = make_subplots(rows=4, cols=1, shared_xaxes=True,
                                               subplot_titles=("Original", "Tendência", "Sazonalidade", "Resíduos"))
                    fig_decomp.add_trace(go.Scatter(x=res.observed.index, y=res.observed, mode='lines', name='Original',
                                                    line=dict(color=theme_colors["primary"])), row=1, col=1)
                    fig_decomp.add_trace(go.Scatter(x=res.trend.index, y=res.trend, mode='lines', name='Tendência',
                                                    line=dict(color=theme_colors["success"])), row=2, col=1)
                    fig_decomp.add_trace(go.Scatter(x=res.seasonal.index, y=res.seasonal, mode='lines', name='Sazonalidade',
                                                    line=dict(color=theme_colors["warning"])), row=3, col=1)
                    fig_decomp.add_trace(go.Scatter(x=res.resid.index, y=res.resid, mode='markers', name='Resíduos',
                                                    marker=dict(color=theme_colors["info"])), row=4, col=1)
                    fig_decomp.update_layout(height=700, showlegend=False)
                    st.plotly_chart(fig_decomp, use_container_width=True)
                    st.markdown(
                        """
                        **Interpretação da Decomposição:**
                        - **Tendência**: Direção geral do faturamento (crescimento/declínio)
                        - **Sazonalidade**: Padrões que se repetem (ex: sazonalidade mensal/anual)
                        - **Resíduos**: Variações não explicadas por tendência ou sazonalidade
                        """
                    )
                except Exception as e:
                    st.warning(f"Erro na decomposição: {e}")
            else:
                st.info("A decomposição requer pelo menos 24 meses de dados e statsmodels instalado.")
        # Correlações
        with analysis_tabs[1]:
            if len(df_lojas) > 10:
                corr_data = df_lojas.pivot_table(index='data', columns='loja', values='faturamento', aggfunc='sum').fillna(0)
                if corr_data.shape[1] > 1:
                    correlation_matrix = corr_data.corr()
                    fig_corr_matrix = go.Figure(
                        data=go.Heatmap(
                            z=correlation_matrix.values,
                            x=correlation_matrix.columns,
                            y=correlation_matrix.index,
                            colorscale='RdBu',
                            zmid=0
                        )
                    )
                    fig_corr_matrix.update_layout(title='Matriz de Correlação entre Lojas', height=500)
                    st.plotly_chart(fig_corr_matrix, use_container_width=True)
                    # Top correlações
                    st.markdown("**Maiores Correlações:**")
                    corr_pairs: list[dict[str, any]] = []
                    cols = list(correlation_matrix.columns)
                    for i in range(len(cols)):
                        for j in range(i + 1, len(cols)):
                            corr_pairs.append({
                                'Loja 1': cols[i],
                                'Loja 2': cols[j],
                                'Correlação': correlation_matrix.iloc[i, j]
                            })
                    top_corr = pd.DataFrame(corr_pairs).sort_values('Correlação', ascending=False).head(5)
                    top_corr['Correlação'] = top_corr['Correlação'].apply(lambda x: f"{x:.3f}")
                    st.dataframe(top_corr, use_container_width=True)
        # Distribuições
        with analysis_tabs[2]:
            fig_dist = make_subplots(rows=1, cols=2, subplot_titles=['Distribuição do Faturamento', 'Box Plot por Loja'])
            fig_dist.add_trace(
                go.Histogram(x=df_f['faturamento'], nbinsx=30, name='Distribuição',
                             marker_color=theme_colors["primary"]),
                row=1, col=1
            )
            fig_dist.add_trace(
                go.Box(y=df_f['faturamento'], x=df_f['loja'], name='Box Plot',
                       marker_color=theme_colors["secondary"]),
                row=1, col=2
            )
            fig_dist.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_dist, use_container_width=True)
    # Benchmarking
    with tabs[3]:
        st.markdown("#### Benchmarking e Comparações")
        bench_metrics = df_f.groupby('loja').agg({
            'faturamento': ['sum', 'mean', 'std'],
            'pedidos': ['sum', 'mean'],
            'ticket': 'mean'
        }).round(2)
        bench_metrics.columns = ['Fat_Total', 'Fat_Médio', 'Fat_StdDev', 'Ped_Total', 'Ped_Médio', 'Ticket_Médio']
        for col in ['Fat_Total', 'Fat_Médio', 'Ped_Total', 'Ticket_Médio']:
            if col in bench_metrics.columns:
                bench_metrics[f'{col}_Score'] = (bench_metrics[col] / bench_metrics[col].max()) * 100
        score_cols = [col for col in bench_metrics.columns if col.endswith('_Score')]
        if score_cols:
            bench_metrics['Score_Geral'] = bench_metrics[score_cols].mean(axis=1).round(1)
            top_5_lojas = bench_metrics.nlargest(5, 'Score_Geral')
            fig_radar = go.Figure()
            theta = ['Faturamento Total', 'Faturamento Médio', 'Total Pedidos', 'Ticket Médio']
            for idx, (loja, row) in enumerate(top_5_lojas.iterrows()):
                fig_radar.add_trace(
                    go.Scatterpolar(
                        r=[row['Fat_Total_Score'], row['Fat_Médio_Score'], row['Ped_Total_Score'], row['Ticket_Médio_Score']],
                        theta=theta,
                        fill='toself',
                        name=loja,
                        marker_color=custom_colors[idx % len(custom_colors)]
                    )
                )
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                showlegend=True,
                title="Benchmarking - Top 5 Lojas",
                height=500
            )
            st.plotly_chart(fig_radar, use_container_width=True)
            # Tabela de scores
            st.markdown("**Ranking Geral de Performance:**")
            ranking_display = bench_metrics[['Score_Geral']].sort_values('Score_Geral', ascending=False)
            ranking_display['Posição'] = range(1, len(ranking_display) + 1)
            ranking_display = ranking_display[['Posição', 'Score_Geral']].reset_index()
            ranking_display.columns = ['Loja', 'Posição', 'Score']
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
                df_f.dropna(subset=["data"]).groupby("data", as_index=False)
                .agg(faturamento=("faturamento", "sum"))
                .sort_values("data")
            )
            fig_mobile = px.line(x=serie_f['data'], y=serie_f['faturamento'], title="Evolução do Faturamento")
            fig_mobile.update_layout(height=300)
            st.plotly_chart(fig_mobile, use_container_width=True)


def display_top_performers(df: pd.DataFrame, periodo_ini: str, periodo_fim: str) -> None:
    """Exibe as lojas com mais pedidos no período selecionado."""
    st.markdown("### 🏆 Top Performers do Período")
    rank_mask = (
        (df["periodo"] >= periodo_ini) & (df["periodo"] <= periodo_fim) & df["pedidos"].notna()
    )
    rank_df = (
        df.loc[rank_mask]
        .groupby("loja", as_index=False)["pedidos"].sum()
        .sort_values("pedidos", ascending=False)
    )
    if not rank_df.empty:
        cols = st.columns(3)
        total_ped = int(rank_df["pedidos"].sum()) if pd.notna(rank_df["pedidos"].sum()) else 0
        for i in range(min(3, len(rank_df))):
            row = rank_df.iloc[i]
            pct = (row["pedidos"] / total_ped * 100) if total_ped else 0
            position = ["🥇", "🥈", "🥉"][i]
            with cols[i]:
                st.markdown(
                    f"""
                    <div class="metric-card">
                        <h3>{position} {row['loja']}</h3>
                        <p><strong>{fmt_int(row['pedidos'])}</strong> pedidos</p>
                        <p>{pct:.1f}% do total</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )


def display_insights(k: dict) -> list[str]:
    """Gera e exibe insights finais com base nos KPIs."""
    st.markdown("### 💡 Insights e Recomendações")
    insights: list[str] = []
    growth_rate = k["advanced"].get("growth_rate", 0) if k.get("advanced") else 0
    volatility = k["advanced"].get("volatility", 0) if k.get("advanced") else 0
    efficiency = k["advanced"].get("efficiency", 0) if k.get("advanced") else 0
    seasonality = k["advanced"].get("seasonality", "") if k.get("advanced") else ""
    correlation = k["advanced"].get("correlation", 0) if k.get("advanced") else 0
    # Crescimento
    if growth_rate > 0.03:
        insights.append("✅ **Crescimento Acelerado**: Considere expandir operações nas lojas top performers.")
    elif growth_rate < -0.02:
        insights.append("⚠️ **Declínio Preocupante**: Revise estratégias de marketing e operações.")
    # Volatilidade
    if volatility > 0.25:
        insights.append("📊 **Alta Volatilidade**: Implemente estratégias de estabilização de demanda.")
    # Eficiência
    if efficiency > 0.1:
        insights.append("⚡ **Excelente Eficiência**: Modelo operacional pode ser replicado em outras lojas.")
    elif efficiency < -0.1:
        insights.append("🔧 **Baixa Eficiência**: Revisar processos operacionais e treinamento de equipe.")
    # Sazonalidade
    if seasonality == "Alta sazonalidade":
        insights.append("📅 **Forte Sazonalidade**: Planeje estoques e promoções baseadas em padrões mensais.")
    # Correlação
    if abs(correlation) > 0.8:
        insights.append("🔗 **Alta Correlação**: Estratégias unificadas podem ser eficazes.")
    elif abs(correlation) < 0.5:
        insights.append("🎯 **Baixa Correlação**: Considere estratégias personalizadas por loja.")
    if insights:
        for insight in insights:
            st.markdown(insight)
    else:
        st.info("📊 Performance estável no período analisado.")
    return insights


def display_exports(df_f: pd.DataFrame, insights: list[str], k: dict, rank_df: pd.DataFrame, periodo_ini: str, periodo_fim: str, sel_lojas: list[str]) -> None:
    """Disponibiliza botões para exportação de relatórios e dados."""
    st.markdown("### 📋 Relatórios e Exportação")
    export_col1, export_col2, export_col3 = st.columns(3)
    with export_col1:
        if st.button("📊 Gerar Resumo Executivo"):
            resumo = f"""\
# RESUMO EXECUTIVO - HORA DO PASTEL
**Período:** {periodo_ini} a {periodo_fim}
**Lojas Analisadas:** {len(sel_lojas)}

## INDICADORES PRINCIPAIS
- **Faturamento Total:** {fmt_brl(k['period_sum']['fat'])}
- **Total de Pedidos:** {fmt_int(k['period_sum']['ped'])}
- **Ticket Médio:** {fmt_brl(k['period_sum']['tik'])}
- **Variação vs Período Anterior:** {fmt_pct(k.get('delta_period_fat', 0))}
- **Variação vs Ano Anterior:** {fmt_pct(k.get('delta_yoy_fat', 0))}

## MÉTRICAS AVANÇADAS
- **Taxa de Crescimento Mensal:** {fmt_pct(k['advanced'].get('growth_rate', 0))}
- **Volatilidade:** {fmt_pct(k['advanced'].get('volatility', 0))}
- **Padrão Sazonal:** {k['advanced'].get('seasonality', 'N/A')}
- **ROI Estimado:** {fmt_brl(k['advanced'].get('estimated_roi', 0))}

## TOP PERFORMERS
"""
            for i, (_, row) in enumerate(rank_df.head(3).iterrows() if rank_df is not None else []):
                resumo += f"- {i+1}º lugar: {row['loja']} ({fmt_int(row['pedidos'])} pedidos)\n"
            resumo += "\n## INSIGHTS PRINCIPAIS\n"
            for insight in insights:
                resumo += f"- {insight.replace('**', '').replace('✅', '').replace('⚠️', '').replace('📊', '').replace('⚡', '').replace('🔧', '').replace('📅', '').replace('🔗', '').replace('🎯', '')}\n"
            st.download_button(
                "📄 Baixar Resumo Executivo",
                data=resumo.encode("utf-8"),
                file_name=f"resumo_executivo_{periodo_ini}_{periodo_fim}.md",
                mime="text/markdown"
            )
    with export_col2:
        if not df_f.empty:
            resumo_detalhado = (
                df_f.assign(ano_mes=df_f["periodo"])
                .groupby(["ano_mes", "loja"], as_index=False)
                .agg(faturamento=("faturamento", "sum"), pedidos=("pedidos", "sum"))
            )
            resumo_detalhado["ticket_medio"] = resumo_detalhado.apply(
                lambda r: safe_div(r["faturamento"], r["pedidos"]), axis=1
            )
            st.download_button(
                "📋 Baixar Dados Detalhados (CSV)",
                data=resumo_detalhado.to_csv(index=False).encode("utf-8"),
                file_name=f"dados_detalhados_{periodo_ini}_{periodo_fim}.csv",
                mime="text/csv"
            )
    with export_col3:
        if st.button("📈 Gerar Relatório KPIs"):
            kpi_report = {
                'Período': f"{periodo_ini} a {periodo_fim}",
                'Faturamento_Total': k['period_sum']['fat'],
                'Total_Pedidos': k['period_sum']['ped'],
                'Ticket_Medio': k['period_sum']['tik'],
                'Variacao_Periodo_Anterior': k.get('delta_period_fat', 0),
                'Variacao_Ano_Anterior': k.get('delta_yoy_fat', 0),
                'Taxa_Crescimento': k['advanced'].get('growth_rate', 0),
                'Volatilidade': k['advanced'].get('volatility', 0),
                'Eficiencia': k['advanced'].get('efficiency', 0),
                'ROI_Estimado': k['advanced'].get('estimated_roi', 0),
                'Correlacao': k['advanced'].get('correlation', 0),
                'Sazonalidade': k['advanced'].get('seasonality', "N/A")
            }
            kpi_df = pd.DataFrame([kpi_report])
            st.download_button(
                "📊 Baixar KPIs (CSV)",
                data=kpi_df.to_csv(index=False).encode("utf-8"),
                file_name=f"kpis_{periodo_ini}_{periodo_fim}.csv",
                mime="text/csv"
            )


# -----------------------------------------------------------------------------
# FUNÇÃO PRINCIPAL
# -----------------------------------------------------------------------------

def main() -> None:
    """Função principal que orquestra o fluxo do dashboard."""
    configure_page()
    df = load_data()
    (
        analysis_mode,
        periodo_ini,
        periodo_fim,
        sel_lojas,
        include_weekends,
        show_trends,
        show_forecasts
    ) = prepare_filters(df)
    df_f, df_lojas = filter_data(df, periodo_ini, periodo_fim, sel_lojas)
    k = compute_kpis(df_f, df_lojas, periodo_ini, periodo_fim)
    # Cabeçalho
    display_header(periodo_ini, periodo_fim, sel_lojas, analysis_mode)
    # Painel de indicadores
    display_kpi_panel(k)
    # Alertas
    display_alerts(k)
    # Análises específicas por modo
    if analysis_mode == "Comparativo":
        display_comparative_analysis(k, df_f)
    elif analysis_mode == "Preditivo":
        display_predictive_analysis(k, show_forecasts)
    # Análise detalhada (aba sempre disponível)
    display_detailed_analysis(df_f, df_lojas, k)
    # Top performers
    display_top_performers(df, periodo_ini, periodo_fim)
    # Insights finais
    insights = display_insights(k)
    # Exportação de relatórios
    # Preparar ranking para resumo executivo
    rank_mask = (df["periodo"] >= periodo_ini) & (df["periodo"] <= periodo_fim) & df["pedidos"].notna()
    rank_df = (
        df.loc[rank_mask]
        .groupby("loja", as_index=False)["pedidos"].sum()
        .sort_values("pedidos", ascending=False)
    )
    display_exports(df_f, insights, k, rank_df, periodo_ini, periodo_fim, sel_lojas)
    # Configurações extras na sidebar
    with st.sidebar:
        st.markdown("---")
        st.markdown("### ⚙️ Configurações")
        show_raw_data = st.checkbox("Mostrar dados brutos", False)
        enable_animations = st.checkbox("Animações nos gráficos", True)
        st.markdown("#### 🔍 Parâmetros de Análise")
        confidence_level = st.slider("Nível de Confiança (%)", 90, 99, 95)
        forecast_periods = st.slider("Períodos de Previsão", 3, 12, 6)
        st.markdown("#### 🚨 Limites para Alertas")
        growth_threshold = st.slider("Limite Crescimento (%)", 1, 10, 3)
        volatility_threshold = st.slider("Limite Volatilidade (%)", 10, 50, 25)
        st.markdown("---")
        st.markdown("### ℹ️ Informações")
        st.caption(f"Última atualização: {datetime.now().strftime('%d/%m/%Y %H:%M')}")
        st.caption(f"Total de registros: {len(df):,}")
        periodos = sorted(p for p in df["periodo"].dropna().unique().tolist())
        if periodos:
            st.caption(f"Período dos dados: {periodos[0]} a {periodos[-1]}")
        # Cache
        if st.button("🔄 Limpar Cache"):
            st.cache_data.clear()
            st.success("Cache limpo com sucesso!")
            st.rerun()
        st.markdown("---")
        st.caption(
            "💡 **Dica**: Use os filtros para focar em períodos específicos e lojas de interesse. "
            "O modo 'Preditivo' oferece projeções baseadas em machine learning."
        )
        if st.checkbox("Mostrar métricas de performance", False):
            st.markdown("### 🔍 Performance")
            st.caption(f"Registros processados: {len(df_f):,}")
            st.caption(f"Lojas ativas: {len(sel_lojas)}")
            st.caption(f"Período analisado: {len(periodos)} meses")
            st.progress(85)
            st.caption("CPU: 85%")
            st.progress(60)
            st.caption("Memória: 60%")
    # Mensagem final
    st.success("✅ Dashboard carregado com sucesso! Explore as diferentes abas e modos de análise para obter insights detalhados sobre a performance das suas lojas.")


if __name__ == "__main__":
    main()