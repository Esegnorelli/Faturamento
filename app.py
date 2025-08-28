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
