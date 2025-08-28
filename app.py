"""
Dashboard Inteligente Modularizado (v2.2)
==================================================

Esta versão traz um conjunto de otimizações, refatorações e melhorias de legibilidade
para a aplicação de dashboard baseada em Streamlit. O objetivo principal foi
simplificar a estrutura de código, adicionar tipagem explícita, organizar
funções auxiliares e documentar melhor o fluxo de execução. Além disso,
mantemos total compatibilidade com a versão anterior (2.1) e preservamos todas as
funcionalidades existentes, incluindo as análises comparativa, preditiva,
detalhada e mobile.

Principais mudanças nesta revisão:

* **Tipagem aprimorada**: foram adicionadas anotações de tipo a praticamente
  todas as funções, melhorando a autocompletude em IDEs e a robustez durante
  o desenvolvimento.
* **Docstrings detalhadas**: cada função agora possui uma descrição clara de
  seu propósito, parâmetros e valor de retorno, facilitando a compreensão do
  código por novos desenvolvedores.
* **Organização e comentários**: o código foi reorganizado em seções mais
  coesas com comentários explicativos, proporcionando melhor leitura e
  manutenção.
* **Pequenas otimizações**: algumas operações de agrupamento e cálculos
  intermediários foram ajustados para reduzir repetições e melhorar a
  performance em cenários com conjuntos de dados maiores.
* **Compatibilidade estendida**: mantivemos as verificações defensivas para
  bibliotecas opcionais como `statsmodels`, garantindo que o dashboard funcione
  mesmo em ambientes onde essas dependências não estejam disponíveis.

Para executar o aplicativo utilize o comando:
```bash
streamlit run app.py
```

Dependências recomendadas: streamlit, pandas, plotly, python-dateutil,
statsmodels (opcional), numpy e scipy.
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

# Tipagem
from typing import Optional, Any, Dict, Tuple, List, Sequence, Callable

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
    """Configura a página Streamlit e define temas e estilos gerais.

    Esta função inicializa a configuração da página, definindo título,
    ícone, layout e itens de menu. Também declara paletas de cores globais
    utilizadas em gráficos e componentes visuais, e injeta CSS customizado
    para ajustar o estilo de elementos específicos como cabeçalho e cartões.
    """
    st.set_page_config(
        page_title="Dashboard Avançado — Hora do Pastel",
        page_icon="🥟",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            "Get Help": "https://www.streamlit.io/community",
            "Report a bug": "mailto:admin@horadopastel.com",
            "About": "### Dashboard Inteligente v2.2\nDesenvolvido para análise avançada de vendas.",
        },
    )

    # Paleta de cores global
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

    # Aplica template e cores padrão do Plotly
    px.defaults.template = "plotly_white"
    px.defaults.color_discrete_sequence = custom_colors

    # CSS adicional para componentes
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
                color: #212529;
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
    """Normaliza nomes de colunas removendo acentos, espaços e
    convertendo tudo para minúsculas.

    Args:
        name: Nome original da coluna.

    Returns:
        Nome normalizado, com acentos retirados e espaços substituídos por '_'.
    """
    name = name.strip().lower()
    name = "".join(
        c for c in unicodedata.normalize("NFKD", name) if not unicodedata.combining(c)
    )
    return re.sub(r"\s+", "_", name)


def _norm_text(s: str) -> str:
    """Normaliza uma string para comparação, removendo acentos e caracteres
    especiais, e convertendo para minúsculas.

    Args:
        s: Texto de entrada.

    Returns:
        Versão normalizada contendo apenas caracteres alfanuméricos e espaços.
    """
    s = "".join(
        c for c in unicodedata.normalize("NFKD", str(s).strip().lower()) if not unicodedata.combining(c)
    )
    s = re.sub(r"[^a-z0-9 ]", " ", s)
    return " ".join(s.split())


def br_to_float(series: pd.Series) -> pd.Series:
    """Converte strings de valores monetários brasileiros para floats.

    Remove símbolos, trata separadores de milhar e decimal e converte
    em tipo numérico. Valores inválidos são convertidos para NaN.

    Args:
        series: Série contendo strings como 'R$ 1.234,56'.

    Returns:
        Série numérica (float64) convertida.
    """
    s = series.astype(str).str.strip()
    s = s.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    s = s.str.replace(r"[^0-9,\.\-]", "", regex=True)
    has_comma = s.str.contains(",", na=False)
    s = s.mask(has_comma, s.str.replace(".", "", regex=False))
    s = s.mask(has_comma, s.str.replace(",", ".", regex=False))
    return pd.to_numeric(s, errors="coerce")


def month_to_int(series: pd.Series) -> pd.Series:
    """Mapeia nomes de meses em português para seu valor inteiro correspondente.

    Args:
        series: Série contendo nomes de meses, abreviações ou números.

    Returns:
        Série de inteiros representando meses (1 a 12). Valores
        desconhecidos são convertidos para NaN.
    """
    mapa = {
        "jan": 1, "janeiro": 1,
        "fev": 2, "fevereiro": 2,
        "mar": 3, "marco": 3, "março": 3,
        "abr": 4, "abril": 4,
        "mai": 5, "maio": 5,
        "jun": 6, "junho": 6,
        "jul": 7, "julho": 7,
        "ago": 8, "agosto": 8,
        "set": 9, "setembro": 9, "sep": 9,
        "out": 10, "outubro": 10,
        "nov": 11, "novembro": 11,
        "dez": 12, "dezembro": 12,
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


def rename_by_alias(cols: Sequence[str]) -> Dict[str, str]:
    """Gera um dicionário de renome baseado em aliases pré-definidos.

    Args:
        cols: Lista de nomes de colunas a serem avaliados.

    Returns:
        Mapeamento `{nome_original: nome_normalizado}` de acordo com `ALIASES`.
    """
    ren: Dict[str, str] = {}
    for c in cols:
        for target, opts in ALIASES.items():
            if c in opts:
                ren[c] = target
                break
    return ren


def safe_div(a: Optional[float | int], b: Optional[float | int]) -> float:
    """Realiza uma divisão segura entre dois valores numéricos.

    Retorna 0.0 caso o divisor seja nulo ou zero, evitando erros. Valores
    inválidos no numerador também resultam em 0.0.

    Args:
        a: Numerador.
        b: Denominador.

    Returns:
        Resultado da divisão ou 0.0 em casos inválidos.
    """
    try:
        if b in (0, None) or pd.isna(b):
            return 0.0
        return float(a) / float(b) if a not in (None, pd.NA) else 0.0
    except Exception:
        return 0.0


def fmt_brl(v: Any) -> str:
    """Formata um valor numérico como moeda brasileira (R$).

    Args:
        v: Valor numérico ou None/NaN.

    Returns:
        String formatada no padrão 'R$ 1.234,56'. Valores nulos retornam 'R$ 0,00'.
    """
    if pd.isna(v):
        return "R$ 0,00"
    s = f"{float(v):,.2f}"
    return "R$ " + s.replace(",", "X").replace(".", ",").replace("X", ".")


def fmt_int(v: Any) -> str:
    """Formata um valor como inteiro com separador de milhares.

    Args:
        v: Valor numérico.

    Returns:
        String representando o inteiro com separadores de milhar. Em caso
        de erro, retorna '0'.
    """
    try:
        return f"{int(v):,}".replace(",", ".")
    except Exception:
        return "0"


def fmt_pct(v: Optional[float], decimals: int = 1) -> str:
    """Formata um valor numérico como percentual.

    Args:
        v: Valor entre 0 e 1 (ou None).
        decimals: Número de casas decimais a exibir.

    Returns:
        String percentual no formato '0,0%'.
    """
    if v is None or pd.isna(v):
        return "0,0%"
    return f"{v * 100:,.{decimals}f}%".replace(".", ",")


def calculate_growth_rate(df_series: pd.Series) -> float:
    """Calcula a taxa de crescimento mensal médio de uma série temporal.

    A taxa é calculada como a raiz n-ésima da razão entre o último e o
    primeiro valor, onde n é o número de intervalos (len - 1). Se o
    valor inicial for menor ou igual a zero, retorna 0.0.

    Args:
        df_series: Série temporal ordenada cronologicamente.

    Returns:
        Taxa de crescimento médio mensal (float).
    """
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
    """Calcula a volatilidade como o desvio padrão das variações percentuais.

    Args:
        df_series: Série temporal de valores (e.g., faturamento).

    Returns:
        Desvio padrão das variações percentuais mensais. Retorna 0.0 se
        a série contiver menos de dois pontos ou variações inválidas.
    """
    if len(df_series) < 2:
        return 0.0
    pct_changes = df_series.pct_change().dropna()
    return float(pct_changes.std()) if not pct_changes.empty else 0.0


def detect_seasonality(series: pd.Series) -> str:
    """Classifica a sazonalidade de uma série temporal agregada por mês.

    Utiliza o coeficiente de variação da média mensal para determinar se
    existe forte sazonalidade (>20%), sazonalidade moderada (>10%) ou baixa
    sazonalidade. Caso a série possua menos de 12 pontos, retorna
    'Dados insuficientes'. Qualquer falha resulta em 'Indeterminado'.

    Args:
        series: Série temporal indexada por datas.

    Returns:
        String descrevendo o nível de sazonalidade.
    """
    if len(series) < 12:
        return "Dados insuficientes"
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
    """Carrega e pré-processa dados de faturamento.

    A função procura primeiramente por um arquivo já tratado
    (`Faturamento_tratado.csv`). Caso não exista, tenta carregar um arquivo
    bruto (`Faturamento.csv`) e realizar o tratamento necessário (normalização
    de colunas, conversão de tipos e criação de colunas derivadas). Quando
    nenhum arquivo é encontrado, gera um conjunto de dados sintético.

    Returns:
        Um DataFrame com colunas essenciais normalizadas e uma coluna 'data'
        de tipo datetime para uso nas análises temporais.
    """
    def _finalize(df: pd.DataFrame) -> pd.DataFrame:
        """Realiza conversões finais e cria coluna de data/periodo.

        Este helper garante que colunas relevantes estejam em tipos
        adequados (Int64 para mês, ano e pedidos; float para
        faturamento e ticket) e constrói uma coluna 'data' combinando
        ano, mês e dia 1. Também cria 'periodo' no formato 'YYYY-MM'.

        Args:
            df: DataFrame parcialmente processado.

        Returns:
            DataFrame pronto para uso nas análises.
        """
        # Conversões finais
        for c in ["mes", "ano", "pedidos"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
        for c in ["faturamento", "ticket"]:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce")
        if "loja" in df.columns:
            df["loja"] = df["loja"].astype(str).str.strip()
        # Cria coluna de data
        mask = df.get("ano").notna() & df.get("mes").notna()
        df["data"] = pd.NaT
        df.loc[mask, "data"] = pd.to_datetime(
            {"year": df.loc[mask, "ano"].astype(int), "month": df.loc[mask, "mes"].astype(int), "day": 1},
            errors="coerce",
        )
        # Cria coluna período AAAA-MM se não existir
        if "periodo" not in df.columns:
            df["periodo"] = pd.to_datetime(df["data"]).dt.to_period("M").astype(str)
        return df.dropna(subset=["data"]).copy()

    # Preferimos arquivo tratado, mas garantimos colunas essenciais
    if os.path.exists("Faturamento_tratado.csv"):
        df = pd.read_csv("Faturamento_tratado.csv")
        base_cols = {"mes", "ano", "loja", "faturamento", "pedidos", "ticket"}
        if not base_cols.issubset(set(map(normalize_col, df.columns))):
            # Normaliza e renomeia apenas se necessário
            df.columns = [normalize_col(c) for c in df.columns]
            df = df.rename(columns=rename_by_alias(list(df.columns)))
            for col in base_cols:
                if col not in df.columns:
                    df[col] = pd.NA
        return _finalize(df)

    # Arquivo bruto
    if os.path.exists("Faturamento.csv"):
        df = pd.read_csv("Faturamento.csv", sep=None, engine="python")
        # Normaliza colunas e remove duplicadas
        df.columns = [normalize_col(c) for c in df.columns]
        df = df.loc[:, ~df.columns.duplicated()].dropna(axis=1, how="all")
        df = df.rename(columns=rename_by_alias(list(df.columns)))
        # Garante colunas essenciais
        for col in ["mes", "ano", "loja", "faturamento", "pedidos", "ticket"]:
            if col not in df.columns:
                df[col] = pd.NA
        # Conversões específicas
        df["mes"] = month_to_int(df["mes"])
        df["ano"] = pd.to_numeric(df["ano"], errors="coerce").astype("Int64")
        df["faturamento"] = br_to_float(df["faturamento"])
        df["ticket"] = br_to_float(df["ticket"])
        df["pedidos"] = pd.to_numeric(df["pedidos"], errors="coerce").round().astype("Int64")
        return _finalize(df)

    # Caso nenhum arquivo exista, utiliza dados sintéticos
    st.warning("Arquivo 'Faturamento.csv' não encontrado. Usando dados de exemplo.")
    return generate_sample_data()


@st.cache_data(ttl=3600, show_spinner=False)
def generate_sample_data() -> pd.DataFrame:
    """Gera um conjunto de dados sintéticos para demonstração.

    Os dados simulam faturamento mensal de cinco lojas ao longo de 30 meses,
    incluindo efeitos de sazonalidade, tendência e um ticket médio derivado
    automaticamente a partir de valores simulados.

    Returns:
        DataFrame contendo colunas ['mes', 'ano', 'loja', 'faturamento',
        'pedidos', 'ticket', 'data', 'periodo'].
    """
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
    """Exibe controles de filtros na barra lateral e retorna seleções do usuário.

    Esta função cria a interface de filtragem na sidebar do Streamlit,
    permitindo escolher o modo de análise, delimitar o período (personalizado
    ou pré-definido), selecionar lojas através de diferentes critérios e
    habilitar opções adicionais como incluir finais de semana ou mostrar
    tendências/previsões.

    Args:
        df: DataFrame principal carregado com os dados de faturamento.

    Returns:
        Uma tupla contendo:
        - analysis_mode: modo de análise selecionado (Padrão, Comparativo, etc.).
        - periodo_ini: início do intervalo no formato AAAA-MM.
        - periodo_fim: final do intervalo no formato AAAA-MM.
        - sel_lojas: lista de lojas selecionadas.
        - include_weekends: flag para incluir finais de semana (reservado).
        - show_trends: flag para mostrar linhas de tendência (reservado).
        - show_forecasts: flag para habilitar previsões no modo preditivo.
    """
    # Exibe logo se presente no diretório
    if os.path.exists("logo.png"):
        st.sidebar.image("logo.png", use_container_width=True)

    st.sidebar.markdown("### 🎯 Filtros Avançados")

    analysis_mode: str = st.sidebar.selectbox(
        "Modo de Análise",
        ["Padrão", "Comparativo", "Preditivo", "Detalhado"],
        help="Escolha o tipo de análise desejada",
    )

    # Determinação de intervalo de período
    periodos = sorted(p for p in df["periodo"].dropna().unique().tolist())
    if len(periodos) < 2:
        st.error("Dados insuficientes para análise. São necessários pelo menos 2 meses de dados.")
        st.stop()

    period_type: str = st.sidebar.selectbox(
        "Tipo de Período",
        ["Personalizado", "Últimos 3M", "Últimos 6M", "Último Ano", "YTD"],
    )

    # Mapeia a seleção do tipo de período para um intervalo específico
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
    GROUPS: Dict[str, List[str]] = {
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
    selection_mode: str = st.sidebar.radio("Modo de Seleção", selection_modes)

    lojas = sorted(df["loja"].dropna().unique().tolist())
    # Mapeia nomes normalizados para a forma original
    map_norm_to_loja = {_norm_text(l): l for l in lojas}

    sel_lojas: List[str]
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
        # Modo personalizado por filtro de desempenho
        min_faturamento = st.sidebar.number_input("Faturamento mínimo (R$)", 0, 1_000_000, 0)
        min_pedidos = st.sidebar.number_input("Pedidos mínimos", 0, 10_000, 0)
        loja_perf = df.groupby("loja").agg({"faturamento": "sum", "pedidos": "sum"}).reset_index()
        filtered = loja_perf[
            (loja_perf["faturamento"] >= min_faturamento) & (loja_perf["pedidos"] >= min_pedidos)
        ]["loja"].tolist()
        sel_lojas = st.sidebar.multiselect("Lojas que atendem critérios:", filtered, default=filtered)

    # Garante que sempre haja pelo menos uma loja selecionada
    if not sel_lojas:
        sel_lojas = lojas[:3]

    # Filtros adicionais (reservados para futuras expansões)
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
    df: pd.DataFrame, periodo_ini: str, periodo_fim: str, sel_lojas: Sequence[str]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Aplica filtros de período e lojas aos dados originais.

    Args:
        df: DataFrame original com todos os registros.
        periodo_ini: Período inicial no formato AAAA-MM.
        periodo_fim: Período final no formato AAAA-MM.
        sel_lojas: Lista de lojas selecionadas para análise.

    Returns:
        Um tuple contendo dois DataFrames:
        - df_f: subconjunto filtrado pelo intervalo selecionado e lojas.
        - df_lojas: subconjunto contendo todos os dados das lojas selecionadas (independente do período).
    """
    mask = (df["periodo"] >= periodo_ini) & (df["periodo"] <= periodo_fim) & df["loja"].isin(sel_lojas)
    df_f = df.loc[mask].copy()
    df_lojas = df[df["loja"].isin(sel_lojas)].copy()
    return df_f, df_lojas


def delta(cur_v: Optional[float | int], base_v: Optional[float | int]) -> Optional[float]:
    """Calcula a variação percentual entre dois valores.

    Args:
        cur_v: Valor atual.
        base_v: Valor base de comparação.

    Returns:
        Percentual de variação ou None se não for possível calcular.
    """
    if cur_v is None or base_v in (None, 0) or pd.isna(base_v):
        return None
    return safe_div((float(cur_v) - float(base_v)), float(base_v))


@st.cache_data(show_spinner=False)
def compute_kpis(
    df_range: pd.DataFrame, df_comp: pd.DataFrame, p_ini: str, p_fim: str
) -> Dict[str, Any]:
    """Computa KPIs básicos e avançados para um intervalo e um conjunto de comparação.

    A função agrupa dados temporais, calcula totais, tickets médios, deltas
    (período anterior, YoY e MoM) e métricas avançadas como crescimento,
    volatilidade, sazonalidade, correlação, ROI estimado e eficiência.

    Args:
        df_range: DataFrame filtrado para o período em análise.
        df_comp: DataFrame contendo todos os dados das lojas selecionadas.
        p_ini: Início do período (AAAA-MM).
        p_fim: Fim do período (AAAA-MM).

    Returns:
        Um dicionário com métricas calculadas e séries temporais intermediárias.
    """
    # Totais e ticket médio
    tot_fat: float = float(df_range["faturamento"].sum())
    tot_ped: int = int(df_range["pedidos"].sum()) if df_range["pedidos"].notna().any() else 0
    tik_med: float = safe_div(tot_fat, tot_ped)

    # Série temporal agregada por data (todas as lojas selecionadas)
    serie_comp = (
        df_comp.dropna(subset=["data"]).groupby("data", as_index=False).agg(
            faturamento=("faturamento", "sum"), pedidos=("pedidos", "sum")
        ).sort_values("data")
    )

    advanced_metrics: Dict[str, Any] = {}
    if not serie_comp.empty and len(serie_comp) > 1:
        # Crescimento & Volatilidade (requerem pelo menos 3 pontos para robustez)
        advanced_metrics["growth_rate"] = calculate_growth_rate(serie_comp["faturamento"]) if len(serie_comp) > 2 else 0.0
        advanced_metrics["volatility"] = calculate_volatility(serie_comp["faturamento"]) if len(serie_comp) > 2 else 0.0
        # Sazonalidade
        advanced_metrics["seasonality"] = detect_seasonality(
            serie_comp.set_index("data")["faturamento"]
        )
        # Correlação pedidos x faturamento (apenas se variância > 0)
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
    prev_fat: float = float(df_prev_period["faturamento"].sum())
    delta_period_fat: Optional[float] = delta(tot_fat, prev_fat)

    # YoY
    yoy_start_date = start_date - relativedelta(years=1)
    yoy_end_date = end_date - relativedelta(years=1)
    mask_yoy = (df_comp["data"] >= yoy_start_date) & (df_comp["data"] <= yoy_end_date)
    df_yoy_period = df_comp[mask_yoy]
    yoy_fat: float = float(df_yoy_period["faturamento"].sum())
    delta_yoy_fat: Optional[float] = delta(tot_fat, yoy_fat)

    # MoM (mês a mês comparado ao mês anterior imediato)
    mom_fat: Optional[float] = None
    mom_ped: Optional[float] = None
    mom_tik: Optional[float] = None
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

def display_header(periodo_ini: str, periodo_fim: str, sel_lojas: Sequence[str], analysis_mode: str) -> None:
    """Exibe o cabeçalho principal com informações contextuais.

    O cabeçalho inclui o título do dashboard e três indicadores em destaque:
    período selecionado, quantidade de lojas analisadas e modo de análise ativo.

    Args:
        periodo_ini: Início do intervalo selecionado.
        periodo_fim: Fim do intervalo selecionado.
        sel_lojas: Lista de lojas selecionadas.
        analysis_mode: Modo de análise atual.
    """
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
    """Renderiza o painel principal de KPIs básicos e avançados.

    Os indicadores incluem faturamento total, total de pedidos, ticket médio,
    variação em relação ao período anterior, comparações YoY, crescimento,
    volatilidade, eficiência e ROI estimado.

    Args:
        k: Dicionário retornado por `compute_kpis` contendo todas as métricas
           necessárias.
    """
    st.markdown("### 📊 Painel de Indicadores")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric(
            label="💰 Faturamento Total",
            value=fmt_brl(k["period_sum"]["fat"]),
            delta=fmt_pct(k.get("delta_period_fat", 0) or 0),
            help=f"Período anterior: {fmt_brl(k['prev_period_fat'])}",
        )
    with col2:
        st.metric(
            label="🛒 Total de Pedidos",
            value=fmt_int(k["period_sum"]["ped"]),
            delta=fmt_pct(k.get("mom_ped", 0) or 0),
            help="Variação MoM do total de pedidos",
        )
    with col3:
        st.metric(
            label="🎯 Ticket Médio",
            value=fmt_brl(k["period_sum"]["tik"]),
            delta=fmt_pct(k.get("mom_tik", 0) or 0),
            help="Variação MoM do ticket médio",
        )
    with col4:
        st.metric(
            label="📈 vs Ano Anterior",
            value=fmt_brl(k["period_sum"]["fat"]),
            delta=fmt_pct(k.get("delta_yoy_fat", 0) or 0),
            help=f"Mesmo período AA: {fmt_brl(k['yoy_fat_abs'])}",
        )

    # KPIs avançados (growth, volatility, efficiency, ROI)
    if k.get("advanced"):
        col5, col6, col7, col8 = st.columns(4)
        growth_rate = k["advanced"].get("growth_rate", 0)
        volatility = k["advanced"].get("volatility", 0)
        efficiency = k["advanced"].get("efficiency", 0)
        estimated_roi = k["advanced"].get("estimated_roi", 0)
        with col5:
            st.metric("📊 Taxa de Crescimento", fmt_pct(growth_rate), help="Crescimento médio mensal")
        with col6:
            st.metric("📉 Volatilidade", fmt_pct(volatility), help="Instabilidade das vendas")
        with col7:
            st.metric("⚡ Eficiência", fmt_pct(efficiency), help="Eficiência vs média histórica")
        with col8:
            st.metric("💎 ROI Estimado", fmt_brl(estimated_roi), help="Lucro estimado (margem 20%)")


def display_alerts(k: Dict[str, Any]) -> None:
    """Apresenta alertas e insights rápidos baseados nas métricas avançadas.

    A função utiliza as métricas de crescimento, eficiência, volatilidade e
    sazonalidade para exibir cartões visuais que destacam pontos de atenção
    ou conquistas. O conteúdo varia conforme os thresholds definidos.

    Args:
        k: Dicionário retornado por `compute_kpis` contendo as métricas.
    """
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
    """Exibe análises comparativas entre períodos e indicadores derivados.

    Gera um gráfico de barras comparando faturamento do período atual com o
    período anterior e um indicador tipo gauge para a correlação entre
    pedidos e faturamento.

    Args:
        k: Dicionário de KPIs retornado por `compute_kpis`.
        df_f: DataFrame filtrado contendo os dados do período atual.
    """
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
    """Mostra análise preditiva de faturamento utilizando Holt-Winters.

    Caso a opção de mostrar previsões esteja habilitada e existam pelo menos
    12 meses de dados, a função ajusta um modelo `ExponentialSmoothing` para
    gerar projeções. Também exibe métricas resumidas da previsão.

    Args:
        k: Dicionário de KPIs e séries temporais.
        show_forecasts: Flag indicando se as previsões devem ser exibidas.
    """
    st.markdown("### 🔮 Análise Preditiva")
    if not show_forecasts:
        st.info("Ative a opção 'Mostrar previsões' para ver a projeção de faturamento.")
        return

    serie_temporal: Optional[pd.DataFrame] = k.get("serie_temporal")
    if sm is None or ExponentialSmoothing is None or serie_temporal is None or len(serie_temporal) < 12:
        st.info("Previsões requerem pelo menos 12 meses de dados e a biblioteca statsmodels.")
        return

    # Controles locais para previsão
    forecast_periods: int = st.slider("Períodos de Previsão (meses)", 3, 12, 6, key="forecast_periods_slider")
    seasonal_mode: str = st.selectbox("Sazonalidade", ["aditiva", "multiplicativa"], index=0)

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
    """Apresenta a análise detalhada em diversas abas.

    Inclui a evolução temporal dos indicadores, comparações entre lojas,
    análises avançadas de decomposição e correlação, benchmarking e um resumo
    para dispositivos móveis. Esta função é responsável por montar a maior
    parte das visualizações interativas do dashboard.

    Args:
        df_f: DataFrame filtrado pelo período atual.
        df_lojas: DataFrame contendo todos os dados das lojas selecionadas.
        k: Dicionário de métricas gerado por `compute_kpis`.
    """
    st.markdown("### 📈 Análise Detalhada")
    tabs = st.tabs([
        "📊 Evolução Temporal",
        "🏪 Performance por Loja",
        "🔬 Análise Avançada",
        "🎯 Benchmarking",
        "📱 Mobile Dashboard",
    ])

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
                subplot_titles=(
                    "Faturamento e Médias Móveis",
                    "Pedidos",
                    "Ticket Médio",
                    "Crescimento MoM",
                ),
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
                go.Bar(
                    x=serie_f["data"],
                    y=serie_f["pedidos"],
                    name="Pedidos",
                    marker_color=theme_colors["secondary"],
                ),
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
        bench_metrics = df_f.groupby("loja").agg({
            "faturamento": ["sum", "mean", "std"],
            "pedidos": ["sum", "mean"],
            "ticket": "mean",
        }).round(2)
        bench_metrics.columns = ["Fat_Total", "Fat_Médio", "Fat_StdDev", "Ped_Total", "Ped_Médio", "Ticket_Médio"]

        # Calcula scores normalizados para cada métrica
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
                        r=[
                            row.get("Fat_Total_Score", 0),
                            row.get("Fat_Médio_Score", 0),
                            row.get("Ped_Total_Score", 0),
                            row.get("Ticket_Médio_Score", 0),
                        ],
                        theta=theta,
                        fill="toself",
                        name=loja,
                        marker_color=custom_colors[idx % len(custom_colors)],
                    )
                )
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                showlegend=True,
                title="Benchmarking - Top 5 Lojas",
                height=500,
            )
            st.plotly_chart(fig_radar, use_container_width=True)

            st.markdown("**Ranking Geral de Performance:**")
            ranking_display = bench_metrics[["Score_Geral"]].sort_values("Score_Geral", ascending=False)
            ranking_display["Posição"] = range(1, len(ranking_display) + 1)
            ranking_display = ranking_display[["Posição", "Score_Geral"]].reset_index().rename(
                columns={"index": "Loja", "Score_Geral": "Score"}
            )
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
            serie_f_mobile = (
                df_f.dropna(subset=["data"]).groupby("data", as_index=False).agg(faturamento=("faturamento", "sum")).sort_values("data")
            )
            fig_mobile = px.line(x=serie_f_mobile["data"], y=serie_f_mobile["faturamento"], title="Evolução do Faturamento")
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
    """Exibe rankings de desempenho e pódio por métrica.

    A função apresenta um ranking Top‑N baseado no número de pedidos e um
    pódio por três categorias: Faturamento, Pedidos e Ticket Médio.

    Args:
        df: DataFrame original com todos os registros.
        periodo_ini: Início do período selecionado.
        periodo_fim: Fim do período selecionado.
        top_n: Quantidade de lojas a serem exibidas no ranking.
        show_podium: Indica se o pódio por categoria deve ser exibido.
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

    # Ranking top‑N por pedidos
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

    # Pódio por métrica
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
            html: List[str] = ['<div class="podium-container">']
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
    """Gera e exibe lista de insights e recomendações baseados nas métricas.

    Interpreta os valores de crescimento, volatilidade, eficiência, sazonalidade
    e correlação para apresentar mensagens propositivas que auxiliam na
    tomada de decisão.

    Args:
        k: Dicionário de métricas retornado por `compute_kpis`.

    Returns:
        Lista de strings com recomendações geradas.
    """
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
    """Função principal que orquestra todo o fluxo do dashboard.

    Esta função inicializa a configuração, carrega dados, aplica filtros,
    computa KPIs, chama componentes de interface e gerencia a barra lateral
    com utilidades adicionais. É a função invocada quando a aplicação é
    executada diretamente.
    """
    configure_page()
    df = load_data()

    (
        analysis_mode,
        periodo_ini,
        periodo_fim,
        sel_lojas,
        include_weekends,  # reservado para uso futuro
        show_trends,       # reservado para uso futuro
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
    """
Dashboard Inteligente Modularizado (v3.0) - Versão Otimizada
=============================================================

Versão melhorada com:
- KPIs inteligentes com explicações interativas
- Top Performers com visualizações aprimoradas
- Algoritmos de análise avançada (clustering, outliers, tendências)
- Interface mais visual e informativa
- Componentes modulares e reutilizáveis
- Performance otimizada com caching inteligente
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
from scipy.stats import pearsonr, zscore

# Tentativa de importação dos módulos de clustering. Caso não estejam disponíveis,
# definimos indicadores nulos para evitar erros de importação.
try:
    from sklearn.cluster import KMeans  # type: ignore
    from sklearn.preprocessing import StandardScaler  # type: ignore
    HAS_SKLEARN = True
except ImportError:
    KMeans = None  # type: ignore
    StandardScaler = None  # type: ignore
    HAS_SKLEARN = False

from typing import Optional, Any, Dict, Tuple, List, Sequence

# Statsmodels com fallback
try:
    import statsmodels.api as sm
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    from statsmodels.stats.diagnostic import acorr_ljungbox
    HAS_STATSMODELS = True
except ImportError:
    sm = None
    ExponentialSmoothing = None
    HAS_STATSMODELS = False

# =============================================================================
# CONFIGURAÇÕES AVANÇADAS E CONSTANTES
# =============================================================================

# Constantes de negócio
BENCHMARK_TARGETS = {
    "growth_rate_excellent": 0.05,
    "growth_rate_good": 0.02,
    "volatility_high": 0.30,
    "volatility_moderate": 0.15,
    "efficiency_excellent": 0.15,
    "efficiency_good": 0.05,
    "correlation_strong": 0.70,
    "outlier_zscore": 2.0,
}

# Paletas de cores avançadas
COLOR_SCHEMES = {
    "primary": ["#FF6B35", "#004E89", "#28A745", "#FFC107", "#DC3545", "#17A2B8"],
    "performance": ["#2E8B57", "#FFD700", "#FF4500", "#8B0000"],
    "gradient": ["#667eea", "#764ba2", "#f093fb", "#f5576c"],
    "business": ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"],
}

def configure_advanced_page() -> None:
    """Configuração avançada da página com tema personalizado."""
    st.set_page_config(
        page_title="Dashboard Inteligente v3.0 — Análise Avançada",
        page_icon="📊",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            "Get Help": "https://www.streamlit.io/community",
            "Report a bug": "mailto:dashboard@empresa.com",
            "About": "### Dashboard Inteligente v3.0\nAnálise avançada com ML e visualizações interativas.",
        },
    )

    # CSS avançado com animações e componentes modernos
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
        
        .main { font-family: 'Inter', sans-serif; }
        
        .hero-header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            padding: 2rem; border-radius: 15px; color: white;
            text-align: center; margin-bottom: 2rem;
            box-shadow: 0 10px 30px rgba(0,0,0,0.1);
        }
        
        .kpi-card {
            background: white; padding: 1.5rem; border-radius: 12px;
            box-shadow: 0 4px 15px rgba(0,0,0,0.1);
            border-left: 4px solid #667eea;
            transition: transform 0.2s ease-in-out;
        }
        .kpi-card:hover { transform: translateY(-2px); }
        
        .insight-card {
            background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
            color: white; padding: 1rem; border-radius: 10px;
            margin: 0.5rem 0; box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        }
        
        .top-performer-gold {
            background: linear-gradient(135deg, #FFD700 0%, #FFA500 100%);
            color: #333; padding: 1.5rem; border-radius: 12px;
            text-align: center; box-shadow: 0 8px 25px rgba(255,215,0,0.3);
            border: 2px solid #FFD700; position: relative;
        }
        
        .top-performer-silver {
            background: linear-gradient(135deg, #C0C0C0 0%, #A8A8A8 100%);
            color: #333; padding: 1.5rem; border-radius: 12px;
            text-align: center; box-shadow: 0 6px 20px rgba(192,192,192,0.3);
            border: 2px solid #C0C0C0;
        }
        
        .top-performer-bronze {
            background: linear-gradient(135deg, #CD7F32 0%, #B8860B 100%);
            color: white; padding: 1.5rem; border-radius: 12px;
            text-align: center; box-shadow: 0 6px 20px rgba(205,127,50,0.3);
            border: 2px solid #CD7F32;
        }
        
        .metric-explanation {
            background: #f8f9fa; border-left: 4px solid #007bff;
            padding: 1rem; border-radius: 8px; margin: 1rem 0;
        }
        
        .progress-bar {
            background: #e9ecef; border-radius: 10px; overflow: hidden;
            height: 20px; margin: 0.5rem 0;
        }
        
        .progress-fill {
            height: 100%; background: linear-gradient(90deg, #28a745, #20c997);
            transition: width 0.5s ease-in-out;
        }
        
        .alert-success { background: linear-gradient(135deg, #d4edda 0%, #c3e6cb 100%); }
        .alert-warning { background: linear-gradient(135deg, #fff3cd 0%, #ffeaa7 100%); }
        .alert-danger { background: linear-gradient(135deg, #f8d7da 0%, #f5c6cb 100%); }
        
        .performance-badge {
            display: inline-block; padding: 0.25rem 0.75rem;
            border-radius: 20px; font-size: 0.875rem; font-weight: 500;
        }
        .badge-excellent { background: #28a745; color: white; }
        .badge-good { background: #ffc107; color: #212529; }
        .badge-poor { background: #dc3545; color: white; }
        
        .interactive-tooltip {
            background: #333; color: white; padding: 0.5rem;
            border-radius: 5px; font-size: 0.8rem; position: relative;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

# =============================================================================
# FUNÇÕES AUXILIARES OTIMIZADAS
# =============================================================================

def normalize_col(name: str) -> str:
    """Normaliza nomes de colunas de forma mais robusta."""
    name = str(name).strip().lower()
    name = "".join(c for c in unicodedata.normalize("NFKD", name) if not unicodedata.combining(c))
    name = re.sub(r"[^\w\s]", "", name)
    return re.sub(r"\s+", "_", name)

def calculate_percentile_rank(value: float, series: pd.Series) -> float:
    """Calcula o percentil de um valor dentro de uma série."""
    if pd.isna(value) or series.empty:
        return 0.0
    return float((series <= value).sum() / len(series) * 100)

def detect_outliers(series: pd.Series, method: str = "zscore") -> pd.Series:
    """Detecta outliers usando Z-score ou IQR."""
    if method == "zscore":
        z_scores = np.abs(zscore(series.dropna()))
        return pd.Series(z_scores > BENCHMARK_TARGETS["outlier_zscore"], index=series.dropna().index)
    elif method == "iqr":
        Q1, Q3 = series.quantile([0.25, 0.75])
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        return (series < lower_bound) | (series > upper_bound)
    return pd.Series(False, index=series.index)

def classify_performance(value: float, metric_type: str) -> Tuple[str, str, str]:
    """Classifica performance e retorna (classe, badge_class, icon)."""
    thresholds = {
        "growth": (BENCHMARK_TARGETS["growth_rate_excellent"], BENCHMARK_TARGETS["growth_rate_good"]),
        "efficiency": (BENCHMARK_TARGETS["efficiency_excellent"], BENCHMARK_TARGETS["efficiency_good"]),
        "volatility": (BENCHMARK_TARGETS["volatility_moderate"], BENCHMARK_TARGETS["volatility_high"]),
    }
    
    if metric_type in thresholds:
        excellent, good = thresholds[metric_type]
        if metric_type == "volatility":  # Menor é melhor
            if value <= excellent:
                return "Excelente", "badge-excellent", "🟢"
            elif value <= good:
                return "Bom", "badge-good", "🟡"
            else:
                return "Atenção", "badge-poor", "🔴"
        else:  # Maior é melhor
            if value >= excellent:
                return "Excelente", "badge-excellent", "🟢"
            elif value >= good:
                return "Bom", "badge-good", "🟡"
            else:
                return "Atenção", "badge-poor", "🔴"
    
    return "Neutro", "badge-good", "⚪"

# =============================================================================
# CARREGAMENTO E PROCESSAMENTO DE DADOS
# =============================================================================

@st.cache_data(ttl=3600, max_entries=10, show_spinner=False)
def load_enhanced_data() -> pd.DataFrame:
    """Carrega dados com validações e enriquecimento automático."""
    
    def _enhance_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """Enriquece DataFrame com colunas calculadas."""
        # Normalização de colunas
        df.columns = [normalize_col(c) for c in df.columns]
        
        # Mapeamento de aliases
        aliases = {
            "mes": ["mes", "mês", "month", "mm"],
            "ano": ["ano", "year", "yyyy", "aa"],
            "loja": ["loja", "filial", "store", "unidade"],
            "faturamento": ["faturamento", "receita", "vendas", "valor_total", "revenue"],
            "pedidos": ["pedidos", "qtde_pedidos", "qtd_pedidos", "orders", "quantidade"],
            "ticket": ["ticket", "ticket_medio", "ticket_médio", "average_order"],
        }
        
        rename_dict = {}
        for target, variations in aliases.items():
            for col in df.columns:
                if col in variations and target not in df.columns:
                    rename_dict[col] = target
                    break
        
        df = df.rename(columns=rename_dict)
        
        # Conversões e validações
        if "mes" in df.columns:
            df["mes"] = pd.to_numeric(df["mes"], errors="coerce").astype("Int64")
        if "ano" in df.columns:
            df["ano"] = pd.to_numeric(df["ano"], errors="coerce").astype("Int64")
        if "faturamento" in df.columns:
            df["faturamento"] = pd.to_numeric(df["faturamento"], errors="coerce")
        if "pedidos" in df.columns:
            df["pedidos"] = pd.to_numeric(df["pedidos"], errors="coerce").astype("Int64")
        if "ticket" in df.columns:
            df["ticket"] = pd.to_numeric(df["ticket"], errors="coerce")
        
        # Criação de colunas derivadas
        if "mes" in df.columns and "ano" in df.columns:
            valid_mask = df["ano"].notna() & df["mes"].notna()
            df.loc[valid_mask, "data"] = pd.to_datetime({
                "year": df.loc[valid_mask, "ano"].astype(int),
                "month": df.loc[valid_mask, "mes"].astype(int),
                "day": 1
            }, errors="coerce")
            df["periodo"] = df["data"].dt.strftime("%Y-%m")
            df["trimestre"] = df["data"].dt.quarter
            df["semestre"] = ((df["data"].dt.month - 1) // 6) + 1
            df["dia_semana"] = df["data"].dt.dayofweek
            df["nome_mes"] = df["data"].dt.month_name()
        
        # Cálculos de ticket se não existir
        if "ticket" not in df.columns and "faturamento" in df.columns and "pedidos" in df.columns:
            df["ticket"] = df["faturamento"] / df["pedidos"].replace(0, np.nan)
        
        # Limpeza final
        df = df.dropna(subset=["data"]).copy()
        
        return df
    
    # Tentativa de carregamento dos arquivos
    for filename in ["Faturamento_tratado.csv", "Faturamento.csv"]:
        if os.path.exists(filename):
            try:
                df = pd.read_csv(filename, sep=None, engine="python")
                return _enhance_dataframe(df)
            except Exception as e:
                st.warning(f"Erro ao carregar {filename}: {e}")
                continue
    
    # Dados sintéticos mais realistas
    return generate_enhanced_sample_data()

def generate_enhanced_sample_data() -> pd.DataFrame:
    """Gera dados sintéticos mais realistas e variados."""
    np.random.seed(42)
    
    # Configuração de lojas com características distintas
    lojas_config = {
        "Centro": {"base_fat": 75000, "volatility": 0.15, "growth_trend": 0.02},
        "Shopping Norte": {"base_fat": 85000, "volatility": 0.10, "growth_trend": 0.03},
        "Shopping Sul": {"base_fat": 90000, "volatility": 0.12, "growth_trend": 0.025},
        "Bairro Leste": {"base_fat": 45000, "volatility": 0.25, "growth_trend": 0.01},
        "Bairro Oeste": {"base_fat": 55000, "volatility": 0.20, "growth_trend": 0.015},
        "Aeroporto": {"base_fat": 65000, "volatility": 0.30, "growth_trend": 0.035},
    }
    
    start_date = datetime(2022, 1, 1)
    periods = 36  # 3 anos de dados
    dates = pd.date_range(start_date, periods=periods, freq="MS")
    
    data = []
    for i, date in enumerate(dates):
        for loja, config in lojas_config.items():
            # Fatores de influência
            seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * date.month / 12) + 0.1 * np.cos(2 * np.pi * date.month / 6)
            trend_factor = (1 + config["growth_trend"]) ** i
            random_factor = 1 + np.random.normal(0, config["volatility"])
            
            # Eventos especiais (Black Friday, Natal, etc.)
            special_events = {11: 1.5, 12: 1.8, 1: 0.7, 2: 0.8}  # Nov, Dez, Jan, Fev
            event_factor = special_events.get(date.month, 1.0)
            
            # Cálculo do faturamento
            faturamento = (config["base_fat"] * seasonal_factor * trend_factor * 
                          random_factor * event_factor)
            
            # Pedidos com correlação imperfeita
            ticket_base = 35 + np.random.normal(0, 5)
            ticket_base = max(15, ticket_base)  # Mínimo de R$ 15
            pedidos = int(max(1, faturamento / ticket_base + np.random.normal(0, 50)))
            ticket_real = faturamento / pedidos
            
            data.append({
                "mes": date.month,
                "ano": date.year,
                "loja": loja,
                "faturamento": round(faturamento, 2),
                "pedidos": pedidos,
                "ticket": round(ticket_real, 2),
                "data": date,
                "periodo": date.strftime("%Y-%m"),
                "trimestre": date.quarter,
                "semestre": ((date.month - 1) // 6) + 1,
            })
    
    return pd.DataFrame(data)

# =============================================================================
# KPIs INTELIGENTES E EXPLICAÇÕES
# =============================================================================

class IntelligentKPI:
    """Classe para KPIs com explicações automáticas e análise inteligente."""
    
    def __init__(self, name: str, value: float, comparison_value: Optional[float] = None,
                 format_func: Optional[callable] = None, icon: str = "📊"):
        self.name = name
        self.value = value
        self.comparison_value = comparison_value
        self.format_func = format_func or (lambda x: f"{x:,.2f}")
        self.icon = icon
        self.delta = self._calculate_delta()
        self.performance_class = self._classify_performance()
    
    def _calculate_delta(self) -> Optional[float]:
        """Calcula variação percentual."""
        if self.comparison_value and self.comparison_value != 0:
            return (self.value - self.comparison_value) / abs(self.comparison_value)
        return None
    
    def _classify_performance(self) -> Tuple[str, str]:
        """Classifica performance baseada no delta."""
        if self.delta is None:
            return "neutro", "⚪"
        elif self.delta > 0.10:
            return "excelente", "🟢"
        elif self.delta > 0.05:
            return "bom", "🟡"
        elif self.delta > -0.05:
            return "estavel", "🟠"
        else:
            return "atencao", "🔴"
    
    def get_explanation(self) -> str:
        """Gera explicação automática do KPI."""
        explanations = {
            "Faturamento Total": "Soma de todas as receitas no período. Indica o volume financeiro movimentado.",
            "Pedidos Totais": "Número total de transações realizadas. Reflete o volume operacional.",
            "Ticket Médio": "Valor médio por pedido (Faturamento ÷ Pedidos). Indica o valor por transação.",
            "Taxa de Crescimento": "Crescimento médio mensal composto. Mostra a tendência de evolução.",
            "Volatilidade": "Variabilidade das vendas. Menor valor indica maior previsibilidade.",
            "Eficiência Operacional": "Performance vs. média histórica. Maior valor indica melhor eficiência.",
        }
        return explanations.get(self.name, "KPI de análise de performance.")
    
    def render(self) -> None:
        """Renderiza o KPI com explicação interativa."""
        delta_text = f"{self.delta*100:+.1f}%" if self.delta else "N/A"
        performance_class, icon = self.performance_class
        
        with st.container():
            col1, col2 = st.columns([3, 1])
            with col1:
                st.metric(
                    label=f"{self.icon} {self.name}",
                    value=self.format_func(self.value),
                    delta=delta_text if self.delta else None,
                )
            with col2:
                if st.button("ℹ️", key=f"info_{self.name}"):
                    st.info(self.get_explanation())

def compute_advanced_kpis(df_filtered: pd.DataFrame, df_historical: pd.DataFrame) -> Dict[str, Any]:
    """Computa KPIs avançados com análise estatística."""
    
    # Métricas básicas
    total_faturamento = float(df_filtered["faturamento"].sum())
    total_pedidos = int(df_filtered["pedidos"].sum())
    ticket_medio = total_faturamento / max(1, total_pedidos)
    
    # Série temporal para comparações
    serie_atual = df_filtered.groupby("data")["faturamento"].sum().sort_index()
    serie_historica = df_historical.groupby("data")["faturamento"].sum().sort_index()
    
    # Comparações temporais
    periodo_anterior_valor = 0.0
    if len(serie_historica) > len(serie_atual):
        periodo_anterior = serie_historica.iloc[-(len(serie_atual)*2):-len(serie_atual)]
        if not periodo_anterior.empty:
            periodo_anterior_valor = float(periodo_anterior.sum())
    
    # Métricas avançadas
    kpis: Dict[str, IntelligentKPI] = {}
    
    # Taxa de crescimento composto
    if len(serie_atual) > 1:
        growth_rate = ((serie_atual.iloc[-1] / max(1, serie_atual.iloc[0])) ** (1/(len(serie_atual)-1))) - 1
        kpis["growth_rate"] = IntelligentKPI(
            "Taxa de Crescimento", growth_rate, 0.02, lambda x: f"{x*100:+.2f}%", "📈"
        )
    
    # Volatilidade (coeficiente de variação)
    if len(serie_atual) > 1:
        cv = serie_atual.std() / max(1e-9, serie_atual.mean())
        kpis["volatility"] = IntelligentKPI(
            "Volatilidade", cv, BENCHMARK_TARGETS["volatility_moderate"], 
            lambda x: f"{x*100:.1f}%", "📊"
        )
    
    # Eficiência vs histórico
    ticket_historico = df_historical["faturamento"].sum() / max(1, df_historical["pedidos"].sum())
    eficiencia = (ticket_medio / max(1, ticket_historico)) - 1
    kpis["efficiency"] = IntelligentKPI(
        "Eficiência Operacional", eficiencia, 0.0, lambda x: f"{x*100:+.1f}%", "⚡"
    )
    
    # Consistência (baseada no desvio padrão normalizado)
    if len(serie_atual) > 2:
        consistencia = 1 - (serie_atual.std() / max(1e-9, serie_atual.max()))
        kpis["consistency"] = IntelligentKPI(
            "Consistência", consistencia, 0.7, lambda x: f"{x*100:.1f}%", "🎯"
        )
    
    # Momentum (aceleração recente)
    if len(serie_atual) >= 6:
        recent = serie_atual.tail(3).mean()
        previous = serie_atual.iloc[-6:-3].mean()
        momentum = (recent - previous) / max(1, previous)
        kpis["momentum"] = IntelligentKPI(
            "Momentum", momentum, 0.0, lambda x: f"{x*100:+.1f}%", "🚀"
        )
    
    # Retorna estrutura completa
    return {
        "basic_metrics": {
            "faturamento": total_faturamento,
            "pedidos": total_pedidos,
            "ticket_medio": ticket_medio,
            "periodo_anterior": periodo_anterior_valor,
        },
        "intelligent_kpis": kpis,
        "time_series": serie_atual,
        "historical_series": serie_historica,
        "outliers": detect_outliers(serie_atual) if len(serie_atual) > 5 else pd.Series(dtype=bool),
    }

# =============================================================================
# COMPONENTES VISUAIS AVANÇADOS
# =============================================================================

def render_hero_section(periodo_ini: str, periodo_fim: str, analysis_mode: str) -> None:
    """Renderiza seção hero com informações principais."""
    st.markdown(
        f"""
        <div class="hero-header">
            <h1>📊 Dashboard Inteligente v3.0</h1>
            <h3>Análise Avançada de Performance com Machine Learning</h3>
            <p><strong>Período:</strong> {periodo_ini} até {periodo_fim} | <strong>Modo:</strong> {analysis_mode}</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

def render_intelligent_kpi_panel(kpis_data: Dict[str, Any]) -> None:
    """Renderiza painel de KPIs com explicações interativas."""
    st.markdown("### 📊 Indicadores Inteligentes")
    
    # KPIs básicos
    basic = kpis_data["basic_metrics"]
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        kpi_fat = IntelligentKPI(
            "Faturamento Total", basic["faturamento"], basic["periodo_anterior"],
            lambda x: f"R$ {x:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."), "💰"
        )
        kpi_fat.render()
    
    with col2:
        kpi_ped = IntelligentKPI(
            "Pedidos Totais", basic["pedidos"], None,
            lambda x: f"{int(x):,}".replace(",", "."), "🛒"
        )
        kpi_ped.render()
    
    with col3:
        kpi_ticket = IntelligentKPI(
            "Ticket Médio", basic["ticket_medio"], None,
            lambda x: f"R$ {x:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."), "🎯"
        )
        kpi_ticket.render()
    
    with col4:
        # Percentil de performance
        serie_hist = kpis_data["historical_series"]
        if not serie_hist.empty:
            percentil = calculate_percentile_rank(basic["faturamento"], serie_hist)
            st.metric("📈 Percentil Performance", f"{percentil:.0f}%")
    
    # KPIs inteligentes avançados
    if kpis_data["intelligent_kpis"]:
        st.markdown("#### KPIs Avançados")
        intell_cols = st.columns(len(kpis_data["intelligent_kpis"]))
        
        for i, (key, kpi) in enumerate(kpis_data["intelligent_kpis"].items()):
            with intell_cols[i]:
                kpi.render()

def render_enhanced_top_performers(df: pd.DataFrame, periodo_ini: str, periodo_fim: str) -> None:
    """Renderiza seção de top performers com visualizações aprimoradas."""
    st.markdown("### 🏆 Top Performers - Análise Multidimensional")
    
    # Filtrar dados do período
    mask = (df["periodo"] >= periodo_ini) & (df["periodo"] <= periodo_fim)
    df_period = df[mask].copy()
    
    if df_period.empty:
        st.warning("Não há dados para o período selecionado.")
        return
    
    # Agregação por loja com métricas avançadas
    agg_data = df_period.groupby("loja").agg({
        "faturamento": ["sum", "mean", "std"],
        "pedidos": ["sum", "mean"],
        "ticket": ["mean", "std"]
    }).round(2)
    
    # Flatten columns
    agg_data.columns = [f"{col[0]}_{col[1]}" for col in agg_data.columns]
    agg_data = agg_data.reset_index()
    
    # Cálculo de scores compostos
    metrics_to_score = ["faturamento_sum", "pedidos_sum", "ticket_mean"]
    for metric in metrics_to_score:
        if metric in agg_data.columns:
            max_val = agg_data[metric].max()
            if max_val > 0:
                agg_data[f"{metric}_score"] = (agg_data[metric] / max_val) * 100
    
    # Score geral (média ponderada)
    score_cols = [col for col in agg_data.columns if col.endswith("_score")]
    if score_cols:
        weights = {"faturamento_sum_score": 0.4, "pedidos_sum_score": 0.3, "ticket_mean_score": 0.3}
        agg_data["score_geral"] = 0
        for col in score_cols:
            weight = weights.get(col, 1/len(score_cols))
            agg_data["score_geral"] += agg_data[col] * weight
    
    # Ordenar por score geral
    agg_data = agg_data.sort_values("score_geral", ascending=False)
    
    # Visualização do pódio aprimorado
    st.markdown("#### Pódio de Performance")
    
    top_3 = agg_data.head(3)
    if len(top_3) >= 3:
        podium_col1, podium_col2, podium_col3 = st.columns([1, 1.2, 1])
        
        # 2º Lugar (Prata)
        with podium_col1:
            silver = top_3.iloc[1]
            st.markdown(
                f"""
                <div class="top-performer-silver">
                    <div style="font-size: 2rem;">🥈</div>
                    <h3>{silver['loja']}</h3>
                    <p><strong>Score: {silver['score_geral']:.1f}</strong></p>
                    <p>Faturamento: R$ {silver['faturamento_sum']:,.0f}</p>
                    <p>Pedidos: {silver['pedidos_sum']:,.0f}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        
        # 1º Lugar (Ouro) - Maior destaque
        with podium_col2:
            gold = top_3.iloc[0]
            st.markdown(
                f"""
                <div class="top-performer-gold">
                    <div style="font-size: 3rem;">👑</div>
                    <h2>{gold['loja']}</h2>
                    <p><strong>Score: {gold['score_geral']:.1f}</strong></p>
                    <p>Faturamento: R$ {gold['faturamento_sum']:,.0f}</p>
                    <p>Pedidos: {gold['pedidos_sum']:,.0f}</p>
                    <p>Ticket: R$ {gold['ticket_mean']:,.2f}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        
        # 3º Lugar (Bronze)
        with podium_col3:
            bronze = top_3.iloc[2]
            st.markdown(
                f"""
                <div class="top-performer-bronze">
                    <div style="font-size: 2rem;">🥉</div>
                    <h3>{bronze['loja']}</h3>
                    <p><strong>Score: {bronze['score_geral']:.1f}</strong></p>
                    <p>Faturamento: R$ {bronze['faturamento_sum']:,.0f}</p>
                    <p>Pedidos: {bronze['pedidos_sum']:,.0f}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
    
    # Gráfico radar comparativo dos top 5
    st.markdown("#### Análise Radar - Top 5 Lojas")
    
    top_5 = agg_data.head(5)
    fig_radar = go.Figure()
    
    categories = ["Faturamento", "Volume Pedidos", "Ticket Médio", "Consistência"]
    
    for idx, (_, row) in enumerate(top_5.iterrows()):
        # Normalizar métricas para 0-100
        fat_norm = (row["faturamento_sum"] / top_5["faturamento_sum"].max()) * 100
        ped_norm = (row["pedidos_sum"] / top_5["pedidos_sum"].max()) * 100  
        ticket_norm = (row["ticket_mean"] / top_5["ticket_mean"].max()) * 100
        consist_norm = 100 - ((row.get("faturamento_std", 0) / row["faturamento_mean"]) * 50) if row["faturamento_mean"] > 0 else 50
        
        fig_radar.add_trace(go.Scatterpolar(
            r=[fat_norm, ped_norm, ticket_norm, max(0, consist_norm)],
            theta=categories,
            fill='toself',
            name=row['loja'],
            line_color=COLOR_SCHEMES["primary"][idx % len(COLOR_SCHEMES["primary"])],
        ))
    
    fig_radar.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
        showlegend=True,
        title="Comparação Multidimensional - Top 5",
        height=500
    )
    st.plotly_chart(fig_radar, use_container_width=True)
    
    # Tabela detalhada com badges de performance
    st.markdown("#### Ranking Detalhado")
    
    display_data = agg_data.copy()
    
    # Adicionar badges de performance
    def get_performance_badge(score: float) -> str:
        if score >= 80:
            return '<span class="performance-badge badge-excellent">Excelente</span>'
        elif score >= 60:
            return '<span class="performance-badge badge-good">Bom</span>'
        else:
            return '<span class="performance-badge badge-poor">Melhorar</span>'
    
    display_data["Performance"] = display_data["score_geral"].apply(get_performance_badge)
    display_data["Faturamento"] = display_data["faturamento_sum"].apply(
        lambda x: f"R$ {x:,.0f}".replace(",", "X").replace(".", ",").replace("X", ".")
    )
    display_data["Pedidos"] = display_data["pedidos_sum"].apply(lambda x: f"{x:,.0f}".replace(",", "."))
    display_data["Ticket"] = display_data["ticket_mean"].apply(
        lambda x: f"R$ {x:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")
    )
    
    # Mostrar apenas colunas relevantes
    cols_to_show = ["loja", "Performance", "Faturamento", "Pedidos", "Ticket", "score_geral"]
    final_display = display_data[cols_to_show].rename(columns={
        "loja": "Loja",
        "score_geral": "Score"
    })
    
    st.markdown(final_display.to_html(escape=False, index=False), unsafe_allow_html=True)

def render_clustering_analysis(df: pd.DataFrame) -> None:
    """Análise de clustering para segmentação automática de lojas."""
    st.markdown("### 🎯 Segmentação Inteligente de Lojas")
    
    # Se a biblioteca scikit-learn não estiver instalada, informamos o usuário e encerramos a função.
    if not HAS_SKLEARN:
        st.info("Clustering requer a biblioteca scikit-learn. Instale-a para habilitar este módulo.")
        return
    
    # Preparar dados para clustering
    features_data = df.groupby("loja").agg({
        "faturamento": ["mean", "std"],
        "pedidos": ["mean", "std"], 
        "ticket": ["mean", "std"]
    })
    
    features_data.columns = [f"{col[0]}_{col[1]}" for col in features_data.columns]
    features_data = features_data.fillna(0)
    
    if len(features_data) < 3:
        st.info("Clustering requer pelo menos 3 lojas para análise.")
        return
    
    # Normalizar features
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(features_data)
    
    # Determinar número ótimo de clusters
    n_clusters = min(4, len(features_data) // 2 + 1)
    
    # K-means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    clusters = kmeans.fit_predict(features_scaled)
    
    features_data["cluster"] = clusters
    features_data["loja"] = features_data.index
    
    # Visualização 3D dos clusters
    fig_cluster = px.scatter_3d(
        features_data,
        x="faturamento_mean",
        y="pedidos_mean", 
        z="ticket_mean",
        color="cluster",
        hover_name="loja",
        title="Segmentação Automática de Lojas",
        labels={
            "faturamento_mean": "Faturamento Médio",
            "pedidos_mean": "Pedidos Médios",
            "ticket_mean": "Ticket Médio"
        },
        color_continuous_scale="Viridis"
    )
    fig_cluster.update_layout(height=600)
    st.plotly_chart(fig_cluster, use_container_width=True)
    
    # Análise dos clusters
    st.markdown("#### Características dos Segmentos")
    
    cluster_analysis = features_data.groupby("cluster").agg({
        "faturamento_mean": "mean",
        "pedidos_mean": "mean",
        "ticket_mean": "mean"
    }).round(0)
    
    cluster_names = {
        0: "High Volume",
        1: "Premium",
        2: "Balanced",
        3: "Growing"
    }
    
    for cluster_id, row in cluster_analysis.iterrows():
        cluster_name = cluster_names.get(cluster_id, f"Segmento {cluster_id}")
        lojas_no_cluster = features_data[features_data["cluster"] == cluster_id]["loja"].tolist()
        
        st.markdown(f"**{cluster_name}** ({len(lojas_no_cluster)} lojas)")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Fat. Médio", f"R$ {row['faturamento_mean']:,.0f}".replace(",", "."))
        with col2:
            st.metric("Ped. Médios", f"{row['pedidos_mean']:,.0f}".replace(",", "."))
        with col3:
            st.metric("Ticket Médio", f"R$ {row['ticket_mean']:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
        
        st.caption(f"Lojas: {', '.join(lojas_no_cluster)}")
        st.markdown("---")

def render_forecasting_module(kpis_data: Dict[str, Any], periods: int = 6) -> None:
    """Módulo avançado de previsão com múltiplos modelos."""
    st.markdown("### 🔮 Previsões Inteligentes")
    
    serie = kpis_data["time_series"]
    
    if len(serie) < 12:
        st.warning("Previsões requerem pelo menos 12 meses de dados históricos.")
        return
    
    # Controles de previsão
    forecast_col1, forecast_col2 = st.columns(2)
    with forecast_col1:
        forecast_periods = st.slider("Períodos para Prever", 3, 12, periods)
    with forecast_col2:
        confidence_level = st.selectbox("Nível de Confiança", [80, 90, 95], index=1)
    
    if not HAS_STATSMODELS:
        st.error("Módulo de previsão requer statsmodels. Instale com: pip install statsmodels")
        return
    
    try:
        # Preparar série temporal
        ts_data = serie.copy()
        ts_data.index = pd.to_datetime(ts_data.index)
        ts_data = ts_data.asfreq("MS").fillna(method="ffill")
        
        # Modelo Holt-Winters
        model = ExponentialSmoothing(
            ts_data,
            trend="add",
            seasonal="add",
            seasonal_periods=12
        ).fit()
        
        forecast = model.forecast(forecast_periods)
        forecast_dates = pd.date_range(
            start=ts_data.index[-1] + pd.DateOffset(months=1),
            periods=forecast_periods,
            freq="MS"
        )
        
        # Intervalo de confiança (aproximado)
        forecast_std = ts_data.std()
        z_score = {80: 1.28, 90: 1.64, 95: 1.96}[confidence_level]
        confidence_interval = z_score * forecast_std
        
        # Visualização
        fig_forecast = go.Figure()
        
        # Dados históricos
        fig_forecast.add_trace(go.Scatter(
            x=ts_data.index,
            y=ts_data.values,
            mode="lines+markers",
            name="Histórico",
            line=dict(color="#667eea", width=3),
        ))
        
        # Previsão
        fig_forecast.add_trace(go.Scatter(
            x=forecast_dates,
            y=forecast.values,
            mode="lines+markers", 
            name="Previsão",
            line=dict(color="#f5576c", width=3, dash="dash"),
        ))
        
        # Intervalo de confiança
        fig_forecast.add_trace(go.Scatter(
            x=list(forecast_dates) + list(forecast_dates[::-1]),
            y=list(forecast + confidence_interval) + list((forecast - confidence_interval)[::-1]),
            fill="toself",
            fillcolor="rgba(245,87,108,0.2)",
            line=dict(color="rgba(255,255,255,0)"),
            name=f"IC {confidence_level}%",
            showlegend=True,
        ))
        
        fig_forecast.update_layout(
            title=f"Previsão de Faturamento - Próximos {forecast_periods} Meses",
            xaxis_title="Período",
            yaxis_title="Faturamento (R$)",
            height=500,
            hovermode="x unified"
        )
        
        st.plotly_chart(fig_forecast, use_container_width=True)
        
        # Métricas de previsão
        forecast_col1, forecast_col2, forecast_col3, forecast_col4 = st.columns(4)
        
        with forecast_col1:
            st.metric(
                "Próximo Mês",
                f"R$ {forecast.iloc[0]:,.0f}".replace(",", "."),
                help="Previsão para o próximo período"
            )
        with forecast_col2:
            st.metric(
                "Média Prevista", 
                f"R$ {forecast.mean():,.0f}".replace(",", "."),
                help="Média dos próximos períodos"
            )
        with forecast_col3:
            total_previsto = forecast.sum()
            total_atual = ts_data.tail(forecast_periods).sum()
            variacao = ((total_previsto - total_atual) / total_atual) * 100 if total_atual > 0 else 0
            st.metric(
                "Total Previsto",
                f"R$ {total_previsto:,.0f}".replace(",", "."),
                f"{variacao:+.1f}%",
                help="Soma dos próximos períodos vs. últimos períodos"
            )
        with forecast_col4:
            acuracia = 100 - abs(forecast_std / ts_data.mean()) * 100 if ts_data.mean() > 0 else 0
            st.metric(
                "Confiabilidade",
                f"{max(0, acuracia):.0f}%",
                help="Estimativa de confiabilidade do modelo"
            )
        
        # Explicação do modelo
        with st.expander("🔬 Como funciona a previsão?"):
            st.markdown("""
            **Modelo Holt-Winters (Suavização Exponencial):**
            
            - **Tendência**: Captura a direção geral de crescimento/declínio
            - **Sazonalidade**: Identifica padrões recorrentes (ex: meses mais fortes)
            - **Suavização**: Reduz ruído mantendo padrões importantes
            
            **Intervalo de Confiança**: Indica a faixa provável onde o valor real pode estar.
            Maior intervalo = maior incerteza na previsão.
            
            **Limitações**: Assume que padrões passados se repetirão. Eventos externos 
            (crises, promoções, mudanças sazonais) podem afetar a precisão.
            """)
    except Exception as e:
        st.error(f"Erro na geração de previsões: {str(e)}")

def render_anomaly_detection(kpis_data: Dict[str, Any]) -> None:
    """Detecta e visualiza anomalias na série temporal."""
    st.markdown("### 🔍 Detecção de Anomalias")
    
    serie = kpis_data["time_series"]
    outliers = kpis_data["outliers"]
    
    if serie.empty or len(serie) < 6:
        st.info("Detecção de anomalias requer pelo menos 6 pontos de dados.")
        return
    
    # Calcular limites de controle estatístico
    mean_value = serie.mean()
    std_value = serie.std()
    upper_limit = mean_value + 2 * std_value
    lower_limit = mean_value - 2 * std_value
    
    # Detectar anomalias por diferentes métodos
    z_scores = np.abs(zscore(serie))
    anomalias_zscore = z_scores > 2.0
    
    # Visualização
    fig_anomaly = go.Figure()
    
    # Série normal
    normal_mask = ~anomalias_zscore
    fig_anomaly.add_trace(go.Scatter(
        x=serie.index[normal_mask],
        y=serie.values[normal_mask],
        mode="lines+markers",
        name="Normal",
        line=dict(color="#28a745", width=2),
        marker=dict(size=6)
    ))
    
    # Anomalias
    if anomalias_zscore.any():
        fig_anomaly.add_trace(go.Scatter(
            x=serie.index[anomalias_zscore],
            y=serie.values[anomalias_zscore],
            mode="markers",
            name="Anomalias",
            marker=dict(color="#dc3545", size=12, symbol="x")
        ))
    
    # Limites de controle
    fig_anomaly.add_hline(y=upper_limit, line_dash="dash", line_color="#ffc107", 
                          annotation_text="Limite Superior")
    fig_anomaly.add_hline(y=lower_limit, line_dash="dash", line_color="#ffc107",
                          annotation_text="Limite Inferior")
    fig_anomaly.add_hline(y=mean_value, line_dash="dot", line_color="#6c757d",
                          annotation_text="Média")
    
    fig_anomaly.update_layout(
        title="Detecção de Anomalias - Controle Estatístico",
        xaxis_title="Período",
        yaxis_title="Faturamento (R$)",
        height=400
    )
    
    st.plotly_chart(fig_anomaly, use_container_width=True)
    
    # Resumo das anomalias
    if anomalias_zscore.any():
        anomaly_dates = serie.index[anomalias_zscore]
        anomaly_values = serie.values[anomalias_zscore]
        
        st.markdown("#### Períodos Anômalos Detectados")
        for date, value in zip(anomaly_dates, anomaly_values):
            deviation = ((value - mean_value) / mean_value) * 100
            alert_type = "success" if deviation > 0 else "warning"
            direction = "acima" if deviation > 0 else "abaixo"
            
            st.markdown(
                f"""
                <div class="alert-{alert_type}">
                    <strong>{date.strftime('%m/%Y')}</strong>: 
                    R$ {value:,.0f} ({deviation:+.1f}% {direction} da média)
                </div>
                """.replace(",", "."),
                unsafe_allow_html=True
            )
    else:
        st.success("✅ Nenhuma anomalia significativa detectada no período.")

def render_business_insights_engine(kpis_data: Dict[str, Any], df: pd.DataFrame) -> None:
    """Motor de insights de negócio baseado em regras e padrões."""
    st.markdown("### 💡 Engine de Insights de Negócio")
    
    insights = []
    
    # Análise dos KPIs inteligentes
    intelligent_kpis = kpis_data.get("intelligent_kpis", {})
    
    # Growth insights
    if "growth_rate" in intelligent_kpis:
        growth_kpi = intelligent_kpis["growth_rate"]
        growth_class, _ = growth_kpi.performance_class
        
        if growth_class == "excelente":
            insights.append({
                "type": "success",
                "title": "Crescimento Acelerado",
                "description": f"Taxa de crescimento de {growth_kpi.value*100:.1f}% indica momentum forte. Considere expandir capacidade.",
                "action": "Avaliar abertura de novas unidades ou ampliação do mix de produtos.",
                "priority": "alta"
            })
        elif growth_class == "atencao":
            insights.append({
                "type": "warning", 
                "title": "Crescimento Desacelerado",
                "description": f"Taxa de {growth_kpi.value*100:.1f}% sugere necessidade de revisão estratégica.",
                "action": "Implementar campanhas de retenção e análise de concorrência.",
                "priority": "alta"
            })
    
    # Volatility insights
    if "volatility" in intelligent_kpis:
        vol_kpi = intelligent_kpis["volatility"]
        vol_class, _ = vol_kpi.performance_class
        
        if vol_class == "atencao":
            insights.append({
                "type": "warning",
                "title": "Alta Variabilidade",
                "description": f"Volatilidade de {vol_kpi.value*100:.1f}% indica vendas instáveis.",
                "action": "Implementar estratégias de estabilização da demanda.",
                "priority": "media"
            })
    
    # Efficiency insights
    if "efficiency" in intelligent_kpis:
        eff_kpi = intelligent_kpis["efficiency"]
        eff_class, _ = eff_kpi.performance_class
        
        if eff_class == "excelente":
            insights.append({
                "type": "success",
                "title": "Eficiência Operacional Superior", 
                "description": f"Performance {eff_kpi.value*100:+.1f}% acima do histórico.",
                "action": "Documentar melhores práticas para replicar em outras lojas.",
                "priority": "media"
            })
    
    # Análise sazonal
    serie = kpis_data["time_series"]
    if len(serie) >= 12:
        monthly_avg = serie.groupby(serie.index.month).mean()
        best_month = monthly_avg.idxmax()
        worst_month = monthly_avg.idxmin()
        
        month_names = {1: "Jan", 2: "Fev", 3: "Mar", 4: "Abr", 5: "Mai", 6: "Jun",
                      7: "Jul", 8: "Ago", 9: "Set", 10: "Out", 11: "Nov", 12: "Dez"}
        
        insights.append({
            "type": "info",
            "title": "Padrão Sazonal Identificado",
            "description": f"Melhor mês: {month_names[best_month]} | Pior mês: {month_names[worst_month]}",
            "action": "Planejar campanhas e estoques baseados no ciclo sazonal.",
            "priority": "baixa"
        })
    
    # Renderizar insights
    if insights:
        # Separar por prioridade
        high_priority = [i for i in insights if i["priority"] == "alta"]
        medium_priority = [i for i in insights if i["priority"] == "media"]
        low_priority = [i for i in insights if i["priority"] == "baixa"]
        
        for priority_group, title in [(high_priority, "🚨 Alta Prioridade"), 
                                      (medium_priority, "⚠️ Média Prioridade"),
                                      (low_priority, "💡 Informativo")]:
            if priority_group:
                st.markdown(f"#### {title}")
                for insight in priority_group:
                    st.markdown(
                        f"""
                        <div class="alert-{insight['type']}">
                            <h5>{insight['title']}</h5>
                            <p>{insight['description']}</p>
                            <small><strong>Ação sugerida:</strong> {insight['action']}</small>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
    else:
        st.info("Nenhum insight crítico identificado no momento.")

def render_interactive_explanations() -> None:
    """Seção de explicações interativas sobre KPIs e métricas."""
    st.markdown("### 📚 Central de Conhecimento")
    
    with st.expander("📊 Explicação dos KPIs Principais"):
        tab1, tab2, tab3 = st.tabs(["Métricas Básicas", "KPIs Avançados", "Algoritmos ML"])
        
        with tab1:
            st.markdown("""
            **Faturamento Total**: Soma de todas as receitas no período
            - **Cálculo**: Σ(vendas_período)
            - **Uso**: Medir volume de negócio e comparar períodos
            
            **Ticket Médio**: Valor médio por transação
            - **Cálculo**: Faturamento Total ÷ Número de Pedidos  
            - **Uso**: Entender valor por cliente e identificar oportunidades de upsell
            
            **Taxa de Conversão**: Eficiência em converter visitantes em vendas
            - **Cálculo**: (Pedidos ÷ Visitantes) × 100
            - **Uso**: Otimizar processos de venda
            """)
        
        with tab2:
            st.markdown("""
            **Taxa de Crescimento Composto**: Crescimento médio por período
            - **Cálculo**: ((Valor_Final ÷ Valor_Inicial)^(1/períodos)) - 1
            - **Uso**: Projetar crescimento sustentável
            
            **Volatilidade (Coeficiente de Variação)**: Estabilidade das vendas
            - **Cálculo**: Desvio_Padrão ÷ Média
            - **Uso**: Avaliar previsibilidade e risco operacional
            
            **Eficiência Operacional**: Performance vs. benchmark histórico
            - **Cálculo**: (Ticket_Atual ÷ Ticket_Histórico) - 1
            - **Uso**: Identificar melhorias operacionais
            """)
        
        with tab3:
            st.markdown("""
            **K-Means Clustering**: Segmentação automática de lojas
            - **Algoritmo**: Agrupa lojas por similaridade de performance
            - **Uso**: Estratégias diferenciadas por segmento
            
            **Z-Score (Detecção de Anomalias)**: Identifica períodos atípicos
            - **Cálculo**: (Valor - Média) ÷ Desvio_Padrão
            - **Uso**: Investigar causas de variações extremas
            
            **Holt-Winters**: Previsão com tendência e sazonalidade
            - **Algoritmo**: Suavização exponencial tripla
            - **Uso**: Planejamento de demanda e estoques
            """)

# =============================================================================
# FUNÇÃO PRINCIPAL OTIMIZADA
# =============================================================================

def main() -> None:
    """Função principal com fluxo otimizado e componentes modulares."""
    
    # Configuração inicial
    configure_advanced_page()
    
    # Carregamento de dados com cache inteligente
    df = load_enhanced_data()
    
    if df.empty:
        st.error("Não foi possível carregar dados. Verifique os arquivos CSV.")
        st.stop()
    
    # Sidebar com filtros avançados
    with st.sidebar:
        st.markdown("### 🎛️ Controles Avançados")
        
        # Seleção de modo
        analysis_modes = {
            "Dashboard Geral": "Visão completa com todos os módulos",
            "Top Performers": "Foco em ranking e segmentação", 
            "Previsões": "Análise preditiva e tendências",
            "Anomalias": "Detecção de padrões atípicos"
        }
        
        selected_mode = st.selectbox(
            "Modo de Análise",
            list(analysis_modes.keys()),
            help="Escolha o foco da análise"
        )
        
        st.caption(analysis_modes[selected_mode])
        
        # Filtros de período
        periodos = sorted(df["periodo"].unique())
        if len(periodos) < 2:
            st.error("Dados insuficientes para análise.")
            st.stop()
        
        periodo_range = st.select_slider(
            "Período de Análise",
            options=periodos,
            value=(periodos[max(0, len(periodos)-12)], periodos[-1]),
            help="Selecione o intervalo de meses para análise"
        )
        
        periodo_ini, periodo_fim = periodo_range
        
        # Seleção de lojas
        st.markdown("#### Filtro de Lojas")
        lojas_disponiveis = sorted(df["loja"].unique())
        
        selection_mode = st.radio(
            "Modo de Seleção",
            ["Todas", "Top Performers", "Manual", "Por Score"],
            help="Como selecionar as lojas para análise"
        )
        
        if selection_mode == "Todas":
            selected_lojas = lojas_disponiveis
        elif selection_mode == "Top Performers":
            n_top = st.slider("Quantas lojas?", 3, len(lojas_disponiveis), 5)
            # Calcular top por faturamento no período
            period_mask = (df["periodo"] >= periodo_ini) & (df["periodo"] <= periodo_fim)
            top_lojas_fat = (df[period_mask].groupby("loja")["faturamento"]
                           .sum().nlargest(n_top).index.tolist())
            selected_lojas = top_lojas_fat
        elif selection_mode == "Manual":
            selected_lojas = st.multiselect(
                "Escolha as lojas:",
                lojas_disponiveis,
                default=lojas_disponiveis[:3]
            )
        else:  # Por Score
            min_score = st.slider("Score mínimo", 0, 100, 70)
            # Calcular scores rapidamente
            period_mask = (df["periodo"] >= periodo_ini) & (df["periodo"] <= periodo_fim)
            scores = df[period_mask].groupby("loja").agg({
                "faturamento": "sum",
                "pedidos": "sum"
            })
            scores["score"] = (scores["faturamento"] / scores["faturamento"].max() * 50 + 
                             scores["pedidos"] / scores["pedidos"].max() * 50)
            qualified_lojas = scores[scores["score"] >= min_score].index.tolist()
            selected_lojas = st.multiselect(
                "Lojas qualificadas:",
                qualified_lojas,
                default=qualified_lojas
            )
        
        if not selected_lojas:
            selected_lojas = lojas_disponiveis[:3]
            st.warning("Nenhuma loja selecionada. Usando padrão.")
        
        # Opções avançadas
        st.markdown("#### Opções Avançadas")
        show_forecasts = st.checkbox("Habilitar Previsões", value=True)
        show_clustering = st.checkbox("Análise de Segmentação", value=True)
        show_anomalies = st.checkbox("Detecção de Anomalias", value=True)
        
        # Informações do dataset
        st.markdown("---")
        st.markdown("### Informações do Dataset")
        st.metric("Total de Registros", len(df))
        st.metric("Lojas Disponíveis", len(lojas_disponiveis))
        st.metric("Período dos Dados", f"{periodos[0]} a {periodos[-1]}")
        
        if st.button("Limpar Cache"):
            st.cache_data.clear()
            st.success("Cache limpo!")
            st.rerun()
    
    # Filtrar dados
    period_mask = ((df["periodo"] >= periodo_ini) & 
                   (df["periodo"] <= periodo_fim) & 
                   df["loja"].isin(selected_lojas))
    df_filtered = df[period_mask].copy()
    df_historical = df[df["loja"].isin(selected_lojas)].copy()
    
    if df_filtered.empty:
        st.error("Nenhum dado encontrado para os filtros selecionados.")
        st.stop()
    
    # Header principal
    render_hero_section(periodo_ini, periodo_fim, selected_mode)
    
    # Computar KPIs
    kpis_data = compute_advanced_kpis(df_filtered, df_historical)
    
    # Renderização baseada no modo selecionado
    if selected_mode == "Dashboard Geral":
        render_intelligent_kpi_panel(kpis_data)
        render_enhanced_top_performers(df, periodo_ini, periodo_fim)
        
        if show_clustering:
            render_clustering_analysis(df_filtered)
        
        if show_anomalies:
            render_anomaly_detection(kpis_data)
        
        if show_forecasts:
            render_forecasting_module(kpis_data)
        
        render_business_insights_engine(kpis_data, df_filtered)
        render_interactive_explanations()
        
    elif selected_mode == "Top Performers":
        render_intelligent_kpi_panel(kpis_data)
        render_enhanced_top_performers(df, periodo_ini, periodo_fim)
        
        if show_clustering:
            render_clustering_analysis(df_filtered)
        
    elif selected_mode == "Previsões":
        render_intelligent_kpi_panel(kpis_data)
        render_forecasting_module(kpis_data)
        render_business_insights_engine(kpis_data, df_filtered)
        
    elif selected_mode == "Anomalias":
        render_intelligent_kpi_panel(kpis_data)
        render_anomaly_detection(kpis_data)
        render_business_insights_engine(kpis_data, df_filtered)
    
    # Seção de performance da aplicação
    with st.expander("⚡ Performance da Aplicação"):
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Registros Processados", len(df_filtered))
        with col2:
            st.metric("Lojas Analisadas", len(selected_lojas))
        with col3:
            cache_info = st.cache_data.get_stats()
            st.metric("Cache Hits", len(cache_info))

def render_advanced_evolution_charts(kpis_data: Dict[str, Any]) -> None:
    """Gráficos avançados de evolução temporal com análises estatísticas."""
    st.markdown("### 📈 Evolução Temporal Avançada")
    
    serie = kpis_data["time_series"]
    if serie.empty:
        st.info("Não há dados suficientes para análise temporal.")
        return
    
    # Preparar dados para visualizações
    df_evolution = pd.DataFrame({
        "data": serie.index,
        "faturamento": serie.values
    })
    
    # Adicionar médias móveis e tendências
    df_evolution["mm_3"] = df_evolution["faturamento"].rolling(3).mean()
    df_evolution["mm_6"] = df_evolution["faturamento"].rolling(6).mean()
    
    # Regressão linear para tendência
    if len(df_evolution) > 3:
        x = np.arange(len(df_evolution))
        coeffs = np.polyfit(x, df_evolution["faturamento"], 1)
        df_evolution["tendencia"] = np.poly1d(coeffs)(x)
    
    # Gráfico principal de evolução
    fig_evolution = make_subplots(
        rows=2, cols=2,
        subplot_titles=("Faturamento e Tendências", "Velocidade de Mudança", 
                       "Análise de Distribuição", "Autocorrelação"),
        specs=[[{"secondary_y": True}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Gráfico 1: Faturamento principal
    fig_evolution.add_trace(
        go.Scatter(x=df_evolution["data"], y=df_evolution["faturamento"],
                  mode="lines+markers", name="Faturamento",
                  line=dict(color="#667eea", width=3)), row=1, col=1
    )
    
    if "mm_3" in df_evolution.columns:
        fig_evolution.add_trace(
            go.Scatter(x=df_evolution["data"], y=df_evolution["mm_3"],
                      mode="lines", name="MM 3M", 
                      line=dict(color="#f093fb", dash="dot")), row=1, col=1
        )
    
    if "tendencia" in df_evolution.columns:
        fig_evolution.add_trace(
            go.Scatter(x=df_evolution["data"], y=df_evolution["tendencia"],
                      mode="lines", name="Tendência Linear",
                      line=dict(color="#f5576c", dash="dash")), row=1, col=1
        )
    
    # Gráfico 2: Velocidade de mudança (derivada)
    if len(df_evolution) > 1:
        mudanca = df_evolution["faturamento"].pct_change() * 100
        fig_evolution.add_trace(
            go.Bar(x=df_evolution["data"], y=mudanca,
                  name="Variação %", marker_color="#28a745"), row=1, col=2
        )
    
    # Gráfico 3: Distribuição
    fig_evolution.add_trace(
        go.Histogram(x=df_evolution["faturamento"], nbinsx=20,
                    name="Distribuição", marker_color="#17a2b8"), row=2, col=1
    )
    
    # Gráfico 4: Autocorrelação (se statsmodels disponível)
    if HAS_STATSMODELS and len(serie) > 10:
        try:
            lags = min(10, len(serie) // 3)
            autocorr = [serie.autocorr(lag=i) for i in range(1, lags + 1)]
            fig_evolution.add_trace(
                go.Bar(x=list(range(1, lags + 1)), y=autocorr,
                      name="Autocorrelação", marker_color="#fd7e14"), row=2, col=2
            )
        except:
            pass
    
    fig_evolution.update_layout(height=700, showlegend=True)
    st.plotly_chart(fig_evolution, use_container_width=True)

# =============================================================================
# EXECUÇÃO PRINCIPAL
# =============================================================================

if __name__ == "__main__":
    main()
