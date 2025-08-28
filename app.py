"""
Dashboard Inteligente Modularizado - Hora do Pastel v3.0
========================================================

Versão otimizada e totalmente dinâmica do dashboard de vendas.
Estrutura orientada a objetos com configurações centralizadas,
cache inteligente e componentes reutilizáveis.

Para executar: streamlit run app.py

Dependências: streamlit, pandas, plotly, python-dateutil,
statsmodels (opcional), numpy, scipy.
"""

import os
import re
import unicodedata
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional, Any, Union
import warnings

warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import streamlit as st
from scipy.stats import pearsonr

# Importação opcional do statsmodels
try:
    import statsmodels.api as sm
    from statsmodels.tsa.holtwinters import ExponentialSmoothing
    STATSMODELS_AVAILABLE = True
except ModuleNotFoundError:
    sm = None
    STATSMODELS_AVAILABLE = False


# =============================================================================
# CONFIGURAÇÕES E CONSTANTES
# =============================================================================

@dataclass
class DashboardConfig:
    """Configurações centralizadas do dashboard."""
    
    # Configurações da página
    page_title: str = "Dashboard Avançado — Hora do Pastel"
    page_icon: str = "🥟"
    layout: str = "wide"
    
    # Cores do tema
    colors: Dict[str, str] = field(default_factory=lambda: {
        "primary": "#FF6B35",
        "secondary": "#004E89", 
        "success": "#28A745",
        "warning": "#FFC107",
        "danger": "#DC3545",
        "info": "#17A2B8",
        "light": "#F8F9FA",
        "dark": "#343A40"
    })
    
    # Ícones otimizados
    icons: Dict[str, str] = field(default_factory=lambda: {
        # Métricas principais
        "faturamento": "💰",
        "pedidos": "🛒", 
        "ticket": "🎯",
        "crescimento": "📈",
        
        # Top performers (sequência solicitada)
        "rank_faturamento": "🏆",  # Troféu - 1º lugar
        "rank_pedidos": "📊",      # Gráfico - 2º lugar  
        "rank_ticket": "💎",       # Diamante - 3º lugar
        
        # Pódio tradicional
        "primeiro": "🥇",
        "segundo": "🥈", 
        "terceiro": "🥉",
        
        # Outros ícones
        "alerta_sucesso": "✅",
        "alerta_atencao": "⚠️",
        "alerta_perigo": "🚨",
        "insight": "💡",
        "config": "⚙️",
        "loja": "🏪",
        "calendario": "📅",
        "relatorio": "📋",
        "grafico": "📊",
        "filtro": "🎯"
    })
    
    # Configurações de cache
    cache_ttl: int = 3600
    cache_max_entries: int = 10
    
    # Configurações de análise
    min_data_points: int = 2
    forecast_periods: int = 6
    confidence_level: float = 0.95
    
    # Limites para alertas
    high_growth_threshold: float = 0.05
    decline_threshold: float = -0.02
    high_volatility_threshold: float = 0.25
    efficiency_threshold: float = 0.1


@dataclass
class UIComponents:
    """Componentes de interface reutilizáveis."""
    
    @staticmethod
    def get_custom_css() -> str:
        """Retorna CSS customizado otimizado."""
        return """
        <style>
            .main-header {
                background: linear-gradient(135deg, #FF6B35, #004E89);
                padding: 1.5rem;
                border-radius: 15px;
                color: white;
                text-align: center;
                margin-bottom: 2rem;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
            }
            
            .metric-card {
                background: white;
                padding: 1.5rem;
                border-radius: 12px;
                box-shadow: 0 2px 8px rgba(0,0,0,0.1);
                text-align: center;
                color: #212529;
                border-left: 4px solid #FF6B35;
                transition: transform 0.2s ease;
            }
            
            .metric-card:hover {
                transform: translateY(-2px);
                box-shadow: 0 4px 12px rgba(0,0,0,0.15);
            }
            
            .alert {
                padding: 1rem;
                border-radius: 10px;
                margin: 1rem 0;
                border-left: 4px solid;
            }
            
            .alert-success {
                background-color: #d4edda;
                border-color: #28a745;
                color: #155724;
            }
            
            .alert-warning {
                background-color: #fff3cd;
                border-color: #ffc107;
                color: #856404;
            }
            
            .alert-danger {
                background-color: #f8d7da;
                border-color: #dc3545;
                color: #721c24;
            }
            
            .podium-container {
                display: flex;
                justify-content: center;
                align-items: flex-end;
                gap: 1rem;
                margin: 2rem 0;
                padding: 1rem;
            }
            
            .podium-item {
                flex: 1;
                padding: 1.5rem;
                border-radius: 12px;
                box-shadow: 0 4px 8px rgba(0,0,0,0.1);
                text-align: center;
                background: white;
                color: #212529;
                transition: all 0.3s ease;
                position: relative;
            }
            
            .podium-item.first {
                transform: translateY(-20px) scale(1.05);
                background: linear-gradient(135deg, #FFD700, #FFA500);
                color: #333;
                border: 2px solid #FF6B35;
            }
            
            .podium-item.second {
                transform: translateY(-10px);
                background: linear-gradient(135deg, #C0C0C0, #A0A0A0);
            }
            
            .podium-item.third {
                background: linear-gradient(135deg, #CD7F32, #A0522D);
                color: white;
            }
            
            .kpi-grid {
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
                gap: 1rem;
                margin: 1rem 0;
            }
            
            .sidebar-section {
                background: #f8f9fa;
                padding: 1rem;
                border-radius: 10px;
                margin-bottom: 1rem;
            }
        </style>
        """


# =============================================================================
# UTILITÁRIOS E HELPERS
# =============================================================================

class DataUtils:
    """Utilitários para manipulação de dados."""
    
    @staticmethod
    def normalize_text(text: str) -> str:
        """Normaliza texto removendo acentos e caracteres especiais."""
        text = unicodedata.normalize('NFKD', str(text).strip().lower())
        text = ''.join(c for c in text if not unicodedata.combining(c))
        return re.sub(r'[^a-z0-9 ]', ' ', text).strip()
    
    @staticmethod
    def normalize_column(name: str) -> str:
        """Normaliza nomes de colunas."""
        name = DataUtils.normalize_text(name)
        return re.sub(r'\s+', '_', name)
    
    @staticmethod
    def safe_divide(numerator: Union[float, int, None], 
                   denominator: Union[float, int, None]) -> float:
        """Divisão segura evitando divisão por zero."""
        try:
            if denominator in (0, None) or pd.isna(denominator):
                return 0.0
            if numerator in (None, pd.NA) or pd.isna(numerator):
                return 0.0
            return float(numerator) / float(denominator)
        except (TypeError, ValueError, ZeroDivisionError):
            return 0.0
    
    @staticmethod
    def convert_br_currency(series: pd.Series) -> pd.Series:
        """Converte strings monetárias brasileiras para float."""
        s = series.astype(str).str.strip()
        s = s.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
        s = s.str.replace(r"[^0-9,\.\-]", "", regex=True)
        
        has_comma = s.str.contains(",", na=False)
        s = s.mask(has_comma, s.str.replace(".", "", regex=False))
        s = s.mask(has_comma, s.str.replace(",", ".", regex=False))
        
        return pd.to_numeric(s, errors="coerce")
    
    @staticmethod
    def convert_month_names(series: pd.Series) -> pd.Series:
        """Converte nomes de meses para números."""
        month_map = {
            "jan": 1, "janeiro": 1, "fev": 2, "fevereiro": 2,
            "mar": 3, "março": 3, "marco": 3, "abr": 4, "abril": 4,
            "mai": 5, "maio": 5, "jun": 6, "junho": 6,
            "jul": 7, "julho": 7, "ago": 8, "agosto": 8,
            "set": 9, "setembro": 9, "sep": 9, "out": 10, "outubro": 10,
            "nov": 11, "novembro": 11, "dez": 12, "dezembro": 12
        }
        
        normalized = series.astype(str).str.strip().str.lower()
        mapped = normalized.map(lambda x: month_map.get(x, x))
        return pd.to_numeric(mapped, errors="coerce").astype("Int64")


class Formatters:
    """Formatadores de valores."""
    
    @staticmethod
    def currency(value: Union[float, int, None]) -> str:
        """Formata valor como moeda brasileira."""
        if pd.isna(value) or value is None:
            return "R$ 0,00"
        try:
            formatted = f"{float(value):,.2f}"
            return "R$ " + formatted.replace(",", "X").replace(".", ",").replace("X", ".")
        except (TypeError, ValueError):
            return "R$ 0,00"
    
    @staticmethod
    def integer(value: Union[float, int, None]) -> str:
        """Formata inteiro com separador de milhares."""
        if pd.isna(value) or value is None:
            return "0"
        try:
            return f"{int(value):,}".replace(",", ".")
        except (TypeError, ValueError):
            return "0"
    
    @staticmethod
    def percentage(value: Union[float, int, None], decimals: int = 1) -> str:
        """Formata como percentual."""
        if pd.isna(value) or value is None:
            return "0,0%"
        try:
            formatted = f"{value * 100:,.{decimals}f}%"
            return formatted.replace(".", ",")
        except (TypeError, ValueError):
            return "0,0%"


class Analytics:
    """Funções de análise e cálculos estatísticos."""
    
    @staticmethod
    def calculate_growth_rate(series: pd.Series) -> float:
        """Calcula taxa de crescimento composta."""
        if len(series) < 2:
            return 0.0
        
        first_val = series.iloc[0]
        last_val = series.iloc[-1]
        periods = len(series) - 1
        
        if first_val <= 0:
            return 0.0
        
        try:
            return (last_val / first_val) ** (1 / periods) - 1
        except (TypeError, ValueError, ZeroDivisionError):
            return 0.0
    
    @staticmethod
    def calculate_volatility(series: pd.Series) -> float:
        """Calcula volatilidade como desvio padrão das variações."""
        if len(series) < 2:
            return 0.0
        
        pct_changes = series.pct_change().dropna()
        return float(pct_changes.std()) if len(pct_changes) > 0 else 0.0
    
    @staticmethod
    def detect_seasonality(series: pd.Series) -> str:
        """Detecta padrões sazonais."""
        if len(series) < 12:
            return "Dados insuficientes"
        
        try:
            # Extrai mês das datas do índice
            if hasattr(series.index, 'month'):
                monthly_avg = series.groupby(series.index.month).mean()
            else:
                # Fallback para séries sem índice de data
                df_temp = series.reset_index()
                df_temp['month'] = pd.to_datetime(df_temp.iloc[:, 0]).dt.month
                monthly_avg = df_temp.groupby('month')[df_temp.columns[1]].mean()
            
            cv = monthly_avg.std() / monthly_avg.mean()
            
            if cv > 0.2:
                return "Alta sazonalidade"
            elif cv > 0.1:
                return "Sazonalidade moderada"
            else:
                return "Baixa sazonalidade"
                
        except Exception:
            return "Não determinada"
    
    @staticmethod
    def calculate_delta(current: Union[float, int, None], 
                       previous: Union[float, int, None]) -> Optional[float]:
        """Calcula variação percentual entre dois valores."""
        if current is None or previous in (None, 0) or pd.isna(previous):
            return None
        
        try:
            return (float(current) - float(previous)) / float(previous)
        except (TypeError, ValueError, ZeroDivisionError):
            return None


# =============================================================================
# CLASSE PRINCIPAL DO DASHBOARD
# =============================================================================

class DashboardManager:
    """Gerenciador principal do dashboard."""
    
    def __init__(self):
        self.config = DashboardConfig()
        self.ui = UIComponents()
        self._setup_page()
        self._load_data()
    
    def _setup_page(self):
        """Configura a página do Streamlit."""
        st.set_page_config(
            page_title=self.config.page_title,
            page_icon=self.config.page_icon,
            layout=self.config.layout,
            initial_sidebar_state="expanded",
            menu_items={
                'Get Help': 'https://www.streamlit.io/community',
                'Report a bug': 'mailto:admin@horadopastel.com',
                'About': '### Dashboard Inteligente v3.0\nVersão otimizada e totalmente dinâmica.'
            }
        )
        
        # Aplica CSS customizado
        st.markdown(self.ui.get_custom_css(), unsafe_allow_html=True)
        
        # Configura tema do Plotly
        px.defaults.template = "plotly_white"
        px.defaults.color_discrete_sequence = list(self.config.colors.values())[:6]
    
    @st.cache_data(ttl=3600, show_spinner=False)
    def _load_sample_data(_self) -> pd.DataFrame:
        """Gera dados sintéticos otimizados."""
        np.random.seed(42)
        
        lojas = ["Centro", "Shopping A", "Shopping B", "Bairro Norte", "Bairro Sul", 
                "Protásio Alves", "Floresta", "Caxias do Sul", "Novo Hamburgo"]
        
        start_date = datetime(2022, 1, 1)
        dates = pd.date_range(start_date, periods=36, freq='M')
        
        data = []
        for date in dates:
            for loja in lojas:
                # Simulação mais realista com múltiplos fatores
                base_faturamento = np.random.normal(45000, 8000)
                seasonal_factor = 1 + 0.3 * np.sin(2 * np.pi * date.month / 12)
                trend_factor = 1 + 0.015 * ((date.year - 2022) * 12 + date.month - 1)
                
                # Fator específico por loja
                loja_factors = {
                    "Centro": 1.2, "Shopping A": 1.1, "Shopping B": 0.9,
                    "Protásio Alves": 1.15, "Caxias do Sul": 0.85
                }
                loja_factor = loja_factors.get(loja, 1.0)
                
                faturamento = max(base_faturamento * seasonal_factor * trend_factor * loja_factor, 1000)
                ticket_base = np.random.normal(28, 6)
                pedidos = max(int(faturamento / max(ticket_base, 10)), 1)
                ticket = DataUtils.safe_divide(faturamento, pedidos)
                
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
    
    @st.cache_data(ttl=3600, show_spinner=False) 
    def _load_data(_self) -> pd.DataFrame:
        """Carrega dados com tratamento otimizado."""
        # Mapeamento de colunas por aliases
        column_aliases = {
            "mes": ["mes", "mês", "month"],
            "ano": ["ano", "year"], 
            "loja": ["loja", "filial", "store"],
            "faturamento": ["faturamento", "receita", "vendas", "valor", "total"],
            "pedidos": ["pedidos", "qtde_pedidos", "qtd_pedidos", "quantidade"],
            "ticket": ["ticket", "ticket_medio", "ticket_médio"]
        }
        
        # Tenta carregar arquivo tratado primeiro
        if os.path.exists("Faturamento_tratado.csv"):
            return pd.read_csv("Faturamento_tratado.csv")
        
        # Carrega arquivo original
        if os.path.exists("Faturamento.csv"):
            try:
                df = pd.read_csv("Faturamento.csv", sep=None, engine="python")
                
                # Normaliza nomes das colunas
                df.columns = [DataUtils.normalize_column(c) for c in df.columns]
                df = df.loc[:, ~df.columns.duplicated()].dropna(axis=1, how="all")
                
                # Renomeia colunas usando aliases
                rename_map = {}
                for target, aliases in column_aliases.items():
                    for col in df.columns:
                        if col in aliases:
                            rename_map[col] = target
                            break
                df = df.rename(columns=rename_map)
                
                # Converte tipos de dados
                if "mes" in df.columns:
                    df["mes"] = DataUtils.convert_month_names(df["mes"])
                if "ano" in df.columns:
                    df["ano"] = pd.to_numeric(df["ano"], errors="coerce").astype("Int64")
                if "faturamento" in df.columns:
                    df["faturamento"] = DataUtils.convert_br_currency(df["faturamento"])
                if "ticket" in df.columns:
                    df["ticket"] = DataUtils.convert_br_currency(df["ticket"])
                if "pedidos" in df.columns:
                    df["pedidos"] = pd.to_numeric(df["pedidos"], errors="coerce").astype("Int64")
                if "loja" in df.columns:
                    df["loja"] = df["loja"].astype(str).str.strip()
                
                # Cria coluna de data
                mask = df["ano"].notna() & df["mes"].notna()
                df["data"] = pd.NaT
                if mask.any():
                    df.loc[mask, "data"] = pd.to_datetime({
                        "year": df.loc[mask, "ano"].astype(int),
                        "month": df.loc[mask, "mes"].astype(int), 
                        "day": 1
                    }, errors="coerce")
                
                df["periodo"] = df["data"].dt.to_period("M").astype(str)
                return df.dropna(subset=['data'])
                
            except Exception as e:
                st.error(f"Erro ao carregar dados: {e}")
        
        # Fallback para dados sintéticos
        st.warning("Usando dados sintéticos para demonstração.")
        return _self._load_sample_data()
    
    def _create_filters(self) -> Tuple[str, str, str, List[str], Dict[str, Any]]:
        """Cria interface de filtros otimizada."""
        with st.sidebar:
            # Logo
            if os.path.exists("logo.png"):
                st.image("logo.png", use_container_width=True)
            
            st.markdown(f"### {self.config.icons['filtro']} Filtros Avançados")
            
            # Modo de análise
            analysis_mode = st.selectbox(
                f"{self.config.icons['config']} Modo de Análise",
                ["Padrão", "Comparativo", "Preditivo", "Detalhado"],
                help="Escolha o tipo de análise desejada"
            )
            
            # Seleção de período
            st.markdown("#### 📅 Período")
            periodos = sorted(self.data["periodo"].dropna().unique())
            
            if len(periodos) < 2:
                st.error("Dados insuficientes. Mínimo 2 períodos necessários.")
                st.stop()
            
            period_type = st.selectbox(
                "Tipo de Período",
                ["Personalizado", "Últimos 3M", "Últimos 6M", "Último Ano", "YTD", "Tudo"]
            )
            
            # Determina intervalo baseado na seleção
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
            elif period_type == "Tudo":
                periodo_ini, periodo_fim = periodos[0], periodos[-1]
            else:  # Personalizado
                periodo_ini, periodo_fim = st.select_slider(
                    "Intervalo",
                    options=periodos,
                    value=(periodos[max(0, len(periodos) - 6)], periodos[-1])
                )
            
            # Seleção de lojas otimizada
            st.markdown(f"#### {self.config.icons['loja']} Lojas")
            lojas = sorted(self.data["loja"].dropna().unique())
            
            selection_mode = st.radio(
                "Modo de Seleção",
                ["Todas", "Manual", "Top N", "Personalizadas"],
                horizontal=True
            )
            
            if selection_mode == "Todas":
                sel_lojas = lojas
            elif selection_mode == "Manual":
                sel_lojas = st.multiselect("Escolher lojas:", lojas, default=lojas[:5])
            elif selection_mode == "Top N":
                n_top = st.slider("Número de lojas:", 3, len(lojas), 5)
                top_lojas = (
                    self.data.groupby("loja")["faturamento"].sum()
                    .nlargest(n_top).index.tolist()
                )
                sel_lojas = top_lojas
            else:  # Personalizadas
                min_faturamento = st.number_input("Faturamento mín. (R$):", 0, 1000000, 0)
                filtered_lojas = (
                    self.data.groupby("loja")["faturamento"].sum()
                    .loc[lambda x: x >= min_faturamento].index.tolist()
                )
                sel_lojas = st.multiselect("Lojas filtradas:", filtered_lojas, default=filtered_lojas)
            
            if not sel_lojas:
                sel_lojas = lojas[:3]
            
            # Configurações avançadas
            st.markdown("#### ⚙️ Opções Avançadas")
            opcoes_avancadas = {
                "show_trends": st.checkbox("Linhas de tendência", True),
                "show_forecasts": st.checkbox("Previsões", False),
                "enable_animations": st.checkbox("Animações", True),
                "show_confidence": st.checkbox("Intervalos de confiança", False)
            }
            
            # Informações do sistema
            st.markdown("---")
            st.markdown("### ℹ️ Sistema")
            col1, col2 = st.columns(2)
            with col1:
                st.metric("Registros", f"{len(self.data):,}")
            with col2:
                st.metric("Lojas", len(lojas))
            
            if st.button("🔄 Atualizar Cache"):
                st.cache_data.clear()
                st.success("Cache limpo!")
                st.rerun()
        
        return analysis_mode, periodo_ini, periodo_fim, sel_lojas, opcoes_avancadas
    
    def run(self):
        """Executa o dashboard principal."""
        # Carrega dados
        self.data = self._load_data()
        
        # Cria filtros
        analysis_mode, periodo_ini, periodo_fim, sel_lojas, opcoes = self._create_filters()
        
        # Filtra dados
        df_filtered, df_lojas = self._filter_data(periodo_ini, periodo_fim, sel_lojas)
        
        # Computa KPIs
        kpis = self._compute_kpis(df_filtered, df_lojas, periodo_ini, periodo_fim)
        
        # Interface principal
        self._render_header(periodo_ini, periodo_fim, sel_lojas, analysis_mode)
        self._render_kpis(kpis)
        self._render_alerts(kpis)
        
        # Análises específicas por modo
        if analysis_mode == "Comparativo":
            self._render_comparative_analysis(kpis, df_filtered)
        elif analysis_mode == "Preditivo":
            self._render_predictive_analysis(kpis, opcoes.get("show_forecasts", False))
        elif analysis_mode == "Detalhado":
            self._render_detailed_analysis(df_filtered, df_lojas, kpis)
        
        # Sempre mostra top performers e insights
        self._render_top_performers(periodo_ini, periodo_fim)
        self._render_insights(kpis)
    
    def _filter_data(self, periodo_ini: str, periodo_fim: str, sel_lojas: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Filtra dados por período e lojas."""
        mask = (
            (self.data["periodo"] >= periodo_ini) &
            (self.data["periodo"] <= periodo_fim) &
            self.data["loja"].isin(sel_lojas)
        )
        df_filtered = self.data.loc[mask].copy()
        df_lojas = self.data[self.data["loja"].isin(sel_lojas)].copy()
        return df_filtered, df_lojas
    
    def _compute_kpis(self, df_range: pd.DataFrame, df_comp: pd.DataFrame, 
                     periodo_ini: str, periodo_fim: str) -> Dict[str, Any]:
        """Computa KPIs otimizados."""
        # KPIs básicos
        tot_faturamento = float(df_range["faturamento"].sum())
        tot_pedidos = int(df_range["pedidos"].sum()) if df_range["pedidos"].notna().any() else 0
        ticket_medio = DataUtils.safe_divide(tot_faturamento, tot_pedidos)
        
        # Série temporal para análises avançadas
        serie_temporal = (
            df_comp.dropna(subset=["data"])
            .groupby("data", as_index=False)
            .agg(faturamento=("faturamento", "sum"), pedidos=("pedidos", "sum"))
            .sort_values("data")
        )
        
        # Métricas avançadas
        metricas_avancadas = {}
        if not serie_temporal.empty and len(serie_temporal) > 1:
            metricas_avancadas.update({
                "growth_rate": Analytics.calculate_growth_rate(serie_temporal["faturamento"]),
                "volatility": Analytics.calculate_volatility(serie_temporal["faturamento"]),
                "seasonality": Analytics.detect_seasonality(serie_temporal.set_index("data")["faturamento"]),
                "estimated_roi": tot_faturamento * 0.2,  # Margem estimada de 20%
                "efficiency": 0.0  # Será calculado abaixo
            })
            
            # Correlação pedidos vs faturamento
            if len(serie_temporal) > 3:
                try:
                    corr_coef, p_value = pearsonr(serie_temporal["pedidos"], serie_temporal["faturamento"])
                    metricas_avancadas.update({
                        "correlation": corr_coef,
                        "correlation_pvalue": p_value
                    })
                except Exception:
                    metricas_avancadas.update({"correlation": 0.0, "correlation_pvalue": 1.0})
            
            # Eficiência vs histórico
            ticket_historico = DataUtils.safe_divide(
                serie_temporal["faturamento"].sum(),
                serie_temporal["pedidos"].sum()
            )
            if ticket_historico > 0:
                metricas_avancadas["efficiency"] = DataUtils.safe_divide(ticket_medio, ticket_historico) - 1
        
        # Comparações temporais
        start_date = pd.to_datetime(periodo_ini)
        end_date = pd.to_datetime(periodo_fim) 
        num_months = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month) + 1
        
        # Período anterior
        prev_end_date = start_date - relativedelta(months=1)
        prev_start_date = prev_end_date - relativedelta(months=num_months - 1)
        mask_prev = (df_comp["data"] >= prev_start_date) & (df_comp["data"] <= prev_end_date)
        prev_faturamento = float(df_comp[mask_prev]["faturamento"].sum())
        
        # Year over Year
        yoy_start_date = start_date - relativedelta(years=1)
        yoy_end_date = end_date - relativedelta(years=1)
        mask_yoy = (df_comp["data"] >= yoy_start_date) & (df_comp["data"] <= yoy_end_date)
        yoy_faturamento = float(df_comp[mask_yoy]["faturamento"].sum())
        
        # Month over Month
        mom_metrics = {}
        if len(serie_temporal) >= 2:
            last_month = serie_temporal.iloc[-1]
            prev_month = serie_temporal.iloc[-2]
            
            mom_metrics.update({
                "mom_faturamento": Analytics.calculate_delta(last_month["faturamento"], prev_month["faturamento"]),
                "mom_pedidos": Analytics.calculate_delta(last_month["pedidos"], prev_month["pedidos"]),
                "mom_ticket": Analytics.calculate_delta(
                    DataUtils.safe_divide(last_month["faturamento"], last_month["pedidos"]),
                    DataUtils.safe_divide(prev_month["faturamento"], prev_month["pedidos"])
                )
            })
        
        return {
            "totais": {
                "faturamento": tot_faturamento,
                "pedidos": tot_pedidos,
                "ticket_medio": ticket_medio
            },
            "comparacoes": {
                "prev_period_faturamento": prev_faturamento,
                "yoy_faturamento": yoy_faturamento,
                "delta_period": Analytics.calculate_delta(tot_faturamento, prev_faturamento),
                "delta_yoy": Analytics.calculate_delta(tot_faturamento, yoy_faturamento)
            },
            "mom": mom_metrics,
            "avancadas": metricas_avancadas,
            "serie_temporal": serie_temporal
        }
    
    def _render_header(self, periodo_ini: str, periodo_fim: str, sel_lojas: List[str], analysis_mode: str):
        """Renderiza cabeçalho otimizado."""
        st.markdown(
            f"""
            <div class="main-header">
                <h1>{self.config.page_icon} Dashboard Inteligente — Hora do Pastel</h1>
                <p>Análise Avançada de Performance e Insights Automáticos v3.0</p>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.info(f"{self.config.icons['calendario']} **Período:** {periodo_ini} a {periodo_fim}")
        with col2:
            st.info(f"{self.config.icons['loja']} **Lojas:** {len(sel_lojas)} selecionadas")
        with col3:
            st.info(f"{self.config.icons['config']} **Modo:** {analysis_mode}")
        with col4:
            st.info(f"{self.config.icons['grafico']} **Registros:** {len(self.data):,}")
    
    def _render_kpis(self, kpis: Dict[str, Any]):
        """Renderiza painel de KPIs otimizado."""
        st.markdown("### Painel de Indicadores")
        
        # KPIs principais
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                label=f"{self.config.icons['faturamento']} Faturamento Total",
                value=Formatters.currency(kpis["totais"]["faturamento"]),
                delta=Formatters.percentage(kpis["comparacoes"].get("delta_period")),
                help=f"Período anterior: {Formatters.currency(kpis['comparacoes']['prev_period_faturamento'])}"
            )
        
        with col2:
            st.metric(
                label=f"{self.config.icons['pedidos']} Total de Pedidos",
                value=Formatters.integer(kpis["totais"]["pedidos"]),
                delta=Formatters.percentage(kpis["mom"].get("mom_pedidos")),
                help="Variação MoM do total de pedidos"
            )
        
        with col3:
            st.metric(
                label=f"{self.config.icons['ticket']} Ticket Médio",
                value=Formatters.currency(kpis["totais"]["ticket_medio"]),
                delta=Formatters.percentage(kpis["mom"].get("mom_ticket")),
                help="Variação MoM do ticket médio"
            )
        
        with col4:
            st.metric(
                label=f"{self.config.icons['crescimento']} vs Ano Anterior",
                value=Formatters.currency(kpis["totais"]["faturamento"]),
                delta=Formatters.percentage(kpis["comparacoes"].get("delta_yoy")),
                help=f"Mesmo período AA: {Formatters.currency(kpis['comparacoes']['yoy_faturamento'])}"
            )
        
        # KPIs avançados
        if kpis.get("avancadas"):
            st.markdown("#### Métricas Avançadas")
            col5, col6, col7, col8 = st.columns(4)
            
            with col5:
                growth_rate = kpis["avancadas"].get("growth_rate", 0)
                st.metric(
                    label="📈 Taxa de Crescimento",
                    value=Formatters.percentage(growth_rate),
                    help="Taxa de crescimento mensal médio"
                )
            
            with col6:
                volatility = kpis["avancadas"].get("volatility", 0)
                st.metric(
                    label="📊 Volatilidade", 
                    value=Formatters.percentage(volatility),
                    help="Medida de instabilidade das vendas"
                )
            
            with col7:
                efficiency = kpis["avancadas"].get("efficiency", 0)
                st.metric(
                    label="⚡ Eficiência",
                    value=Formatters.percentage(efficiency),
                    help="Eficiência vs média histórica"
                )
            
            with col8:
                roi = kpis["avancadas"].get("estimated_roi", 0)
                st.metric(
                    label="💎 ROI Estimado",
                    value=Formatters.currency(roi),
                    help="Lucro estimado (margem 20%)"
                )
    
    def _render_alerts(self, kpis: Dict[str, Any]):
        """Renderiza alertas inteligentes."""
        st.markdown("### Alertas e Insights Automáticos")
        
        col1, col2 = st.columns(2)
        
        growth_rate = kpis["avancadas"].get("growth_rate", 0) if kpis.get("avancadas") else 0
        efficiency = kpis["avancadas"].get("efficiency", 0) if kpis.get("avancadas") else 0
        volatility = kpis["avancadas"].get("volatility", 0) if kpis.get("avancadas") else 0
        seasonality = kpis["avancadas"].get("seasonality", "") if kpis.get("avancadas") else ""
        
        with col1:
            if growth_rate > self.config.high_growth_threshold:
                st.markdown(
                    f"""
                    <div class="alert alert-success">
                        <h4>{self.config.icons['alerta_sucesso']} Excelente Crescimento!</h4>
                        <p>Taxa de crescimento de <strong>{Formatters.percentage(growth_rate)}</strong> indica performance excepcional.</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            elif growth_rate < self.config.decline_threshold:
                st.markdown(
                    f"""
                    <div class="alert alert-danger">
                        <h4>{self.config.icons['alerta_perigo']} Atenção: Declínio nas Vendas</h4>
                        <p>Queda de <strong>{Formatters.percentage(abs(growth_rate))}</strong> requer análise de estratégias.</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"""
                    <div class="alert alert-warning">
                        <h4>{self.config.icons['alerta_atencao']} Crescimento Estável</h4>
                        <p>Taxa de <strong>{Formatters.percentage(growth_rate)}</strong> indica estabilidade.</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
        
        with col2:
            if efficiency > self.config.efficiency_threshold:
                st.markdown(
                    f"""
                    <div class="alert alert-success">
                        <h4>⚡ Alta Eficiência</h4>
                        <p>Performance <strong>{Formatters.percentage(efficiency)}</strong> acima da média!</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            elif volatility > self.config.high_volatility_threshold:
                st.markdown(
                    f"""
                    <div class="alert alert-warning">
                        <h4>📈 Alta Volatilidade</h4>
                        <p>Variação de <strong>{Formatters.percentage(volatility)}</strong> - considere estratégias de estabilização.</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            else:
                st.markdown(
                    f"""
                    <div class="alert alert-success">
                        <h4>🎯 Performance Consistente</h4>
                        <p>Padrão sazonal: <strong>{seasonality}</strong></p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
    
    def _render_comparative_analysis(self, kpis: Dict[str, Any], df_filtered: pd.DataFrame):
        """Análise comparativa detalhada."""
        st.markdown("### Análise Comparativa Detalhada")
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Gráfico comparativo de faturamento
            current_fat = kpis["totais"]["faturamento"]
            prev_fat = kpis["comparacoes"]["prev_period_faturamento"]
            
            fig_comp = go.Figure()
            fig_comp.add_trace(
                go.Bar(
                    x=['Período Atual', 'Período Anterior'],
                    y=[current_fat, prev_fat],
                    marker_color=[self.config.colors["primary"], self.config.colors["secondary"]],
                    text=[Formatters.currency(current_fat), Formatters.currency(prev_fat)],
                    textposition='auto'
                )
            )
            fig_comp.update_layout(
                title="Comparação de Faturamento",
                height=350,
                showlegend=False
            )
            st.plotly_chart(fig_comp, use_container_width=True)
        
        with col2:
            # Gauge de correlação
            correlation = kpis["avancadas"].get("correlation", 0) if kpis.get("avancadas") else 0
            
            fig_gauge = go.Figure(
                go.Indicator(
                    mode="gauge+number+delta",
                    value=abs(correlation) * 100,
                    domain={'x': [0, 1], 'y': [0, 1]},
                    title={'text': "Correlação Pedidos x Faturamento"},
                    delta={'reference': 80},
                    gauge={
                        'axis': {'range': [None, 100]},
                        'bar': {'color': self.config.colors["primary"]},
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
            fig_gauge.update_layout(height=350)
            st.plotly_chart(fig_gauge, use_container_width=True)
    
    def _render_predictive_analysis(self, kpis: Dict[str, Any], show_forecasts: bool):
        """Análise preditiva com forecasting."""
        st.markdown("### Análise Preditiva")
        
        if not show_forecasts:
            st.info("Ative a opção 'Previsões' nos filtros para ver projeções.")
            return
        
        serie_temporal = kpis.get("serie_temporal")
        if not STATSMODELS_AVAILABLE or serie_temporal is None or len(serie_temporal) < 12:
            st.info("Previsões requerem pelo menos 12 meses de dados e statsmodels.")
            return
        
        try:
            # Prepara série para forecasting
            serie_forecast = serie_temporal.set_index('data')['faturamento']
            serie_forecast.index = pd.to_datetime(serie_forecast.index)
            serie_forecast = serie_forecast.asfreq('MS')
            
            # Modelo Holt-Winters
            model = ExponentialSmoothing(
                serie_forecast,
                seasonal='add',
                seasonal_periods=min(12, len(serie_forecast) // 2)
            ).fit()
            
            forecast = model.forecast(self.config.forecast_periods)
            forecast_dates = pd.date_range(
                start=serie_forecast.index[-1] + pd.DateOffset(months=1),
                periods=self.config.forecast_periods,
                freq='MS'
            )
            
            # Gráfico de previsão
            fig_pred = go.Figure()
            
            fig_pred.add_trace(
                go.Scatter(
                    x=serie_forecast.index,
                    y=serie_forecast.values,
                    mode='lines+markers',
                    name='Histórico',
                    line=dict(color=self.config.colors["primary"], width=2)
                )
            )
            
            fig_pred.add_trace(
                go.Scatter(
                    x=forecast_dates,
                    y=forecast.values,
                    mode='lines+markers',
                    name='Previsão',
                    line=dict(color=self.config.colors["warning"], width=2, dash='dash')
                )
            )
            
            fig_pred.update_layout(
                title=f"Previsão de Faturamento - Próximos {self.config.forecast_periods} Meses",
                xaxis_title="Data",
                yaxis_title="Faturamento (R$)",
                height=500,
                hovermode='x unified'
            )
            
            st.plotly_chart(fig_pred, use_container_width=True)
            
            # Métricas de previsão
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Previsão Próximo Mês",
                    Formatters.currency(forecast.iloc[0]),
                    help="Baseado em modelo Holt-Winters"
                )
            
            with col2:
                avg_forecast = forecast.mean()
                st.metric(
                    f"Média Prevista ({self.config.forecast_periods}M)",
                    Formatters.currency(avg_forecast),
                    help=f"Média dos próximos {self.config.forecast_periods} meses"
                )
            
            with col3:
                total_forecast = forecast.sum()
                st.metric(
                    f"Total Previsto ({self.config.forecast_periods}M)",
                    Formatters.currency(total_forecast),
                    help=f"Soma dos próximos {self.config.forecast_periods} meses"
                )
                
        except Exception as e:
            st.warning(f"Erro na geração de previsões: {e}")
    
    def _render_detailed_analysis(self, df_filtered: pd.DataFrame, df_lojas: pd.DataFrame, kpis: Dict[str, Any]):
        """Análise detalhada com múltiplas abas."""
        st.markdown("### Análise Detalhada")
        
        tabs = st.tabs([
            "📊 Evolução Temporal",
            "🏪 Performance por Loja", 
            "🔬 Análise Estatística",
            "🎯 Benchmarking"
        ])
        
        # Aba 1: Evolução Temporal
        with tabs[0]:
            self._render_temporal_evolution(df_filtered)
        
        # Aba 2: Performance por Loja
        with tabs[1]:
            self._render_store_performance(df_filtered)
        
        # Aba 3: Análise Estatística
        with tabs[2]:
            self._render_statistical_analysis(df_lojas, kpis)
        
        # Aba 4: Benchmarking
        with tabs[3]:
            self._render_benchmarking(df_filtered)
    
    def _render_temporal_evolution(self, df_filtered: pd.DataFrame):
        """Renderiza evolução temporal."""
        if df_filtered.empty:
            st.warning("Nenhum dado disponível para o período selecionado.")
            return
        
        # Agrupa por data
        serie_temporal = (
            df_filtered.dropna(subset=["data"])
            .groupby("data", as_index=False)
            .agg(
                faturamento=("faturamento", "sum"),
                pedidos=("pedidos", "sum")
            )
            .sort_values("data")
        )
        
        if serie_temporal.empty:
            st.warning("Dados insuficientes para análise temporal.")
            return
        
        # Calcula métricas derivadas
        serie_temporal["ticket_medio"] = serie_temporal.apply(
            lambda r: DataUtils.safe_divide(r["faturamento"], r["pedidos"]), axis=1
        )
        serie_temporal['faturamento_mm3'] = serie_temporal['faturamento'].rolling(window=3, min_periods=1).mean()
        serie_temporal['growth_mom'] = serie_temporal['faturamento'].pct_change()
        
        # Cria subplots
        fig_evolution = make_subplots(
            rows=2, cols=2,
            subplot_titles=[
                'Faturamento e Média Móvel',
                'Volume de Pedidos', 
                'Ticket Médio',
                'Crescimento MoM'
            ],
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        # Faturamento
        fig_evolution.add_trace(
            go.Scatter(
                x=serie_temporal['data'],
                y=serie_temporal['faturamento'],
                name='Faturamento',
                mode='lines+markers',
                line=dict(width=3, color=self.config.colors["primary"])
            ),
            row=1, col=1
        )
        
        fig_evolution.add_trace(
            go.Scatter(
                x=serie_temporal['data'],
                y=serie_temporal['faturamento_mm3'],
                name='MM 3M',
                mode='lines',
                line=dict(width=2, dash='dot', color=self.config.colors["info"])
            ),
            row=1, col=1
        )
        
        # Pedidos
        fig_evolution.add_trace(
            go.Bar(
                x=serie_temporal['data'],
                y=serie_temporal['pedidos'],
                name='Pedidos',
                marker_color=self.config.colors["secondary"]
            ),
            row=1, col=2
        )
        
        # Ticket médio
        fig_evolution.add_trace(
            go.Scatter(
                x=serie_temporal['data'],
                y=serie_temporal['ticket_medio'],
                name='Ticket Médio',
                mode='lines+markers',
                line=dict(width=2, color=self.config.colors["warning"])
            ),
            row=2, col=1
        )
        
        # Crescimento MoM
        colors = ['red' if x < 0 else 'green' for x in serie_temporal['growth_mom'].fillna(0)]
        fig_evolution.add_trace(
            go.Bar(
                x=serie_temporal['data'],
                y=serie_temporal['growth_mom'],
                name='Crescimento MoM',
                marker_color=colors
            ),
            row=2, col=2
        )
        
        fig_evolution.update_layout(height=700, showlegend=False)
        st.plotly_chart(fig_evolution, use_container_width=True)
        
        # Estatísticas resumo
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Média Mensal", Formatters.currency(serie_temporal['faturamento'].mean()))
        with col2:
            st.metric("Mediana", Formatters.currency(serie_temporal['faturamento'].median()))
        with col3:
            st.metric("Desvio Padrão", Formatters.currency(serie_temporal['faturamento'].std()))
        with col4:
            cv = serie_temporal['faturamento'].std() / serie_temporal['faturamento'].mean()
            st.metric("Coef. Variação", Formatters.percentage(cv))
    
    def _render_store_performance(self, df_filtered: pd.DataFrame):
        """Renderiza performance por loja."""
        if df_filtered.empty:
            st.warning("Nenhum dado disponível.")
            return
        
        col1, col2 = st.columns(2)
        
        # Treemap de participação
        with col1:
            st.markdown("**Participação no Faturamento**")
            participacao = df_filtered.groupby("loja")["faturamento"].sum().reset_index()
            
            if not participacao.empty:
                fig_tree = px.treemap(
                    participacao,
                    path=['loja'],
                    values='faturamento',
                    title='Contribuição por Loja',
                    color='faturamento',
                    color_continuous_scale='Viridis'
                )
                fig_tree.update_layout(height=400)
                st.plotly_chart(fig_tree, use_container_width=True)
        
        # Scatter de eficiência
        with col2:
            st.markdown("**Eficiência Operacional**")
            eficiencia = df_filtered.groupby("loja").agg({
                'faturamento': 'sum',
                'pedidos': 'sum'
            }).reset_index()
            
            if not eficiencia.empty:
                eficiencia['ticket'] = eficiencia.apply(
                    lambda r: DataUtils.safe_divide(r['faturamento'], r['pedidos']), axis=1
                )
                
                fig_scatter = px.scatter(
                    eficiencia,
                    x='pedidos',
                    y='faturamento',
                    size='ticket',
                    color='loja',
                    hover_name='loja',
                    title='Faturamento vs Volume',
                    size_max=60
                )
                fig_scatter.update_layout(height=400)
                st.plotly_chart(fig_scatter, use_container_width=True)
        
        # Ranking detalhado
        st.markdown("**Ranking Completo de Performance**")
        ranking = df_filtered.groupby("loja").agg({
            'faturamento': 'sum',
            'pedidos': 'sum'
        }).reset_index()
        
        ranking['ticket_medio'] = ranking.apply(
            lambda r: DataUtils.safe_divide(r['faturamento'], r['pedidos']), axis=1
        )
        ranking['participacao'] = ranking['faturamento'] / ranking['faturamento'].sum()
        ranking = ranking.sort_values('faturamento', ascending=False)
        
        # Formatar para exibição
        ranking_display = ranking.copy()
        ranking_display['faturamento'] = ranking_display['faturamento'].apply(Formatters.currency)
        ranking_display['pedidos'] = ranking_display['pedidos'].apply(Formatters.integer)
        ranking_display['ticket_medio'] = ranking_display['ticket_medio'].apply(Formatters.currency)
        ranking_display['participacao'] = ranking_display['participacao'].apply(
            lambda x: Formatters.percentage(x, 2)
        )
        
        st.dataframe(
            ranking_display.rename(columns={
                'loja': 'Loja',
                'faturamento': 'Faturamento',
                'pedidos': 'Pedidos',
                'ticket_medio': 'Ticket Médio',
                'participacao': 'Participação %'
            }),
            use_container_width=True,
            height=400
        )
    
    def _render_statistical_analysis(self, df_lojas: pd.DataFrame, kpis: Dict[str, Any]):
        """Renderiza análises estatísticas avançadas."""
        if df_lojas.empty:
            st.warning("Dados insuficientes para análise estatística.")
            return
        
        serie_completa = df_lojas.groupby('data')['faturamento'].sum().sort_index()
        
        # Decomposição sazonal (se disponível)
        if STATSMODELS_AVAILABLE and len(serie_completa) >= 24:
            st.markdown("**Decomposição Sazonal**")
            try:
                serie_completa.index = pd.to_datetime(serie_completa.index)
                decomp = sm.tsa.seasonal_decompose(serie_completa.asfreq('MS'), model='additive')
                
                fig_decomp = make_subplots(
                    rows=4, cols=1,
                    shared_xaxes=True,
                    subplot_titles=["Original", "Tendência", "Sazonalidade", "Resíduos"]
                )
                
                components = [
                    (decomp.observed, "Original", self.config.colors["primary"]),
                    (decomp.trend, "Tendência", self.config.colors["success"]),
                    (decomp.seasonal, "Sazonalidade", self.config.colors["warning"]),
                    (decomp.resid, "Resíduos", self.config.colors["info"])
                ]
                
                for i, (component, name, color) in enumerate(components, 1):
                    mode = 'lines' if name != "Resíduos" else 'markers'
                    fig_decomp.add_trace(
                        go.Scatter(
                            x=component.index,
                            y=component,
                            mode=mode,
                            name=name,
                            line=dict(color=color) if mode == 'lines' else dict(),
                            marker=dict(color=color) if mode == 'markers' else dict()
                        ),
                        row=i, col=1
                    )
                
                fig_decomp.update_layout(height=800, showlegend=False)
                st.plotly_chart(fig_decomp, use_container_width=True)
                
            except Exception as e:
                st.warning(f"Erro na decomposição sazonal: {e}")
        
        # Matriz de correlação entre lojas
        st.markdown("**Correlação entre Lojas**")
        corr_data = df_lojas.pivot_table(
            index='data',
            columns='loja', 
            values='faturamento',
            aggfunc='sum'
        ).fillna(0)
        
        if corr_data.shape[1] > 1:
            correlation_matrix = corr_data.corr()
            
            fig_corr = go.Figure(
                data=go.Heatmap(
                    z=correlation_matrix.values,
                    x=correlation_matrix.columns,
                    y=correlation_matrix.index,
                    colorscale='RdBu',
                    zmid=0,
                    text=correlation_matrix.round(2),
                    texttemplate="%{text}",
                    textfont={"size": 10}
                )
            )
            )
            fig_corr.update_layout(
                title='Matriz de Correlação entre Lojas',
                height=500
            )
            st.plotly_chart(fig_corr, use_container_width=True)
        
        # Distribuições e outliers
        st.markdown("**Análise de Distribuições**")
        col1, col2 = st.columns(2)
        
        with col1:
            fig_hist = px.histogram(
                df_lojas,
                x='faturamento',
                nbins=30,
                title='Distribuição do Faturamento',
                color_discrete_sequence=[self.config.colors["primary"]]
            )
            fig_hist.update_layout(height=350)
            st.plotly_chart(fig_hist, use_container_width=True)
        
        with col2:
            fig_box = px.box(
                df_lojas,
                x='loja',
                y='faturamento',
                title='Box Plot por Loja'
            )
            fig_box.update_layout(height=350)
            fig_box.update_xaxes(tickangle=45)
            st.plotly_chart(fig_box, use_container_width=True)
    
    def _render_benchmarking(self, df_filtered: pd.DataFrame):
        """Renderiza análise de benchmarking."""
        if df_filtered.empty:
            st.warning("Dados insuficientes para benchmarking.")
            return
        
        # Calcula métricas por loja
        bench_metrics = df_filtered.groupby('loja').agg({
            'faturamento': ['sum', 'mean', 'std'],
            'pedidos': ['sum', 'mean'],
        }).round(2)
        
        bench_metrics.columns = ['Fat_Total', 'Fat_Medio', 'Fat_StdDev', 'Ped_Total', 'Ped_Medio']
        
        # Calcula ticket médio
        bench_metrics['Ticket_Medio'] = bench_metrics.apply(
            lambda row: DataUtils.safe_divide(row['Fat_Total'], row['Ped_Total']), axis=1
        )
        
        # Calcula scores normalizados (0-100)
        score_columns = ['Fat_Total', 'Fat_Medio', 'Ped_Total', 'Ticket_Medio']
        for col in score_columns:
            if col in bench_metrics.columns and bench_metrics[col].max() > 0:
                bench_metrics[f'{col}_Score'] = (bench_metrics[col] / bench_metrics[col].max()) * 100
        
        # Score geral
        score_cols = [col for col in bench_metrics.columns if col.endswith('_Score')]
        if score_cols:
            bench_metrics['Score_Geral'] = bench_metrics[score_cols].mean(axis=1).round(1)
            
            # Top 5 lojas para radar chart
            top_5_lojas = bench_metrics.nlargest(5, 'Score_Geral')
            
            # Radar chart
            fig_radar = go.Figure()
            
            categories = ['Faturamento Total', 'Faturamento Médio', 'Total Pedidos', 'Ticket Médio']
            
            for idx, (loja, row) in enumerate(top_5_lojas.iterrows()):
                values = [
                    row['Fat_Total_Score'],
                    row['Fat_Medio_Score'],
                    row['Ped_Total_Score'],
                    row['Ticket_Medio_Score']
                ]
                
                fig_radar.add_trace(
                    go.Scatterpolar(
                        r=values,
                        theta=categories,
                        fill='toself',
                        name=loja,
                        marker_color=list(self.config.colors.values())[idx % len(self.config.colors)]
                    )
                )
            
            fig_radar.update_layout(
                polar=dict(radialaxis=dict(visible=True, range=[0, 100])),
                showlegend=True,
                title="Benchmarking - Top 5 Lojas",
                height=500
            )
            
            st.plotly_chart(fig_radar, use_container_width=True)
            
            # Ranking geral
            st.markdown("**Ranking Geral de Performance**")
            ranking_display = bench_metrics[['Score_Geral']].sort_values('Score_Geral', ascending=False)
            ranking_display['Posicao'] = range(1, len(ranking_display) + 1)
            ranking_display = ranking_display[['Posicao', 'Score_Geral']].reset_index()
            ranking_display.columns = ['Loja', 'Posição', 'Score']
            ranking_display['Score'] = ranking_display['Score'].apply(lambda x: f"{x:.1f}")
            
            st.dataframe(ranking_display, use_container_width=True, height=300)
    
    def _render_top_performers(self, periodo_ini: str, periodo_fim: str):
        """Renderiza top performers com pódio otimizado."""
        st.markdown("### Top Performers do Período")
        
        # Filtra dados do período
        mask = (
            (self.data["periodo"] >= periodo_ini) &
            (self.data["periodo"] <= periodo_fim) &
            self.data["pedidos"].notna()
        )
        
        period_data = self.data.loc[mask]
        
        if period_data.empty:
            st.warning("Nenhum dado disponível para o período selecionado.")
            return
        
        # Top 3 por volume de pedidos (ranking tradicional)
        top_pedidos = (
            period_data.groupby("loja")["pedidos"].sum()
            .sort_values(ascending=False)
            .head(3)
        )
        
        if not top_pedidos.empty:
            st.markdown("#### Ranking por Volume de Pedidos")
            cols = st.columns(3)
            total_pedidos = int(period_data["pedidos"].sum())
            
            for i, (loja, pedidos) in enumerate(top_pedidos.items()):
                participacao = (pedidos / total_pedidos * 100) if total_pedidos > 0 else 0
                posicoes = [self.config.icons["primeiro"], self.config.icons["segundo"], self.config.icons["terceiro"]]
                
                with cols[i]:
                    st.markdown(
                        f"""
                        <div class="metric-card">
                            <h3>{posicoes[i]} {loja}</h3>
                            <p><strong>{Formatters.integer(pedidos)}</strong> pedidos</p>
                            <p>{participacao:.1f}% do total</p>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )
        
        # Pódio por métricas específicas (sequência solicitada: Faturamento, Pedidos, Ticket)
        st.markdown("#### Pódio de Excelência por Categoria")
        
        agg_data = period_data.groupby("loja").agg({
            'faturamento': 'sum',
            'pedidos': 'sum'
        })
        agg_data['ticket'] = agg_data.apply(
            lambda r: DataUtils.safe_divide(r['faturamento'], r['pedidos']), axis=1
        )
        
        if not agg_data.empty:
            # Identifica os melhores em cada categoria
            top_faturamento = agg_data.loc[agg_data['faturamento'].idxmax()]
            top_pedidos_metric = agg_data.loc[agg_data['pedidos'].idxmax()]
            top_ticket = agg_data.loc[agg_data['ticket'].idxmax()]
            
            # Dados para o pódio (sequência: Faturamento, Pedidos, Ticket)
            metrics_data = {
                'Faturamento': (top_faturamento.name, top_faturamento['faturamento'], self.config.icons["rank_faturamento"]),
                'Pedidos': (top_pedidos_metric.name, top_pedidos_metric['pedidos'], self.config.icons["rank_pedidos"]),
                'Ticket': (top_ticket.name, top_ticket['ticket'], self.config.icons["rank_ticket"])
            }
            
            # HTML do pódio com destaque para faturamento (centro)
            podium_html = '<div class="podium-container">'
            
            # Ordem visual: Pedidos (esquerda), Faturamento (centro/primeiro), Ticket (direita)
            ordered_metrics = ['Pedidos', 'Faturamento', 'Ticket']
            
            for idx, metric_name in enumerate(ordered_metrics):
                store_name, value, icon = metrics_data[metric_name]
                
                # Formatação baseada na métrica
                if metric_name in ['Faturamento', 'Ticket']:
                    value_fmt = Formatters.currency(value)
                else:
                    value_fmt = Formatters.integer(value)
                
                # Classes CSS para posicionamento do pódio
                css_class = 'podium-item'
                if metric_name == 'Faturamento':  # Primeiro lugar no centro
                    css_class += ' first'
                elif metric_name == 'Pedidos':   # Segundo lugar
                    css_class += ' second'
                else:  # Terceiro lugar (Ticket)
                    css_class += ' third'
                
                podium_html += f"""
                    <div class="{css_class}">
                        <h4>{icon} {metric_name}</h4>
                        <p><strong>{store_name}</strong></p>
                        <p>{value_fmt}</p>
                    </div>
                """
            
            podium_html += '</div>'
            st.markdown(podium_html, unsafe_allow_html=True)
    
    def _render_insights(self, kpis: Dict[str, Any]):
        """Renderiza insights automáticos otimizados."""
        st.markdown(f"### {self.config.icons['insight']} Insights e Recomendações")
        
        insights = []
        
        if kpis.get("avancadas"):
            growth_rate = kpis["avancadas"].get("growth_rate", 0)
            volatility = kpis["avancadas"].get("volatility", 0)
            efficiency = kpis["avancadas"].get("efficiency", 0)
            seasonality = kpis["avancadas"].get("seasonality", "")
            correlation = kpis["avancadas"].get("correlation", 0)
            
            # Análise de crescimento
            if growth_rate > self.config.high_growth_threshold:
                insights.append(
                    f"{self.config.icons['alerta_sucesso']} **Crescimento Acelerado**: "
                    f"Taxa de {Formatters.percentage(growth_rate)} permite expansão estratégica."
                )
            elif growth_rate < self.config.decline_threshold:
                insights.append(
                    f"{self.config.icons['alerta_perigo']} **Atenção ao Declínio**: "
                    f"Queda de {Formatters.percentage(abs(growth_rate))} requer ação imediata."
                )
            
            # Análise de volatilidade
            if volatility > self.config.high_volatility_threshold:
                insights.append(
                    f"{self.config.icons['alerta_atencao']} **Alta Volatilidade**: "
                    f"Variação de {Formatters.percentage(volatility)} indica necessidade de estabilização."
                )
            
            # Análise de eficiência
            if efficiency > self.config.efficiency_threshold:
                insights.append(
                    f"⚡ **Excelente Eficiência**: "
                    f"Performance {Formatters.percentage(efficiency)} acima da média pode ser replicada."
                )
            elif efficiency < -self.config.efficiency_threshold:
                insights.append(
                    f"🔧 **Oportunidade de Melhoria**: "
                    f"Eficiência {Formatters.percentage(efficiency)} abaixo da média requer otimização."
                )
            
            # Análise de sazonalidade
            if seasonality == "Alta sazonalidade":
                insights.append(
                    f"📅 **Forte Padrão Sazonal**: "
                    f"Planejamento baseado em sazonalidade pode otimizar resultados."
                )
            
            # Análise de correlação
            if abs(correlation) > 0.8:
                insights.append(
                    f"🔗 **Alta Sinergia**: "
                    f"Correlação de {correlation:.2f} favorece estratégias integradas."
                )
            elif abs(correlation) < 0.5:
                insights.append(
                    f"🎯 **Estratégias Personalizadas**: "
                    f"Baixa correlação sugere abordagens específicas por loja."
                )
        
        # Exibe insights
        if insights:
            for insight in insights:
                st.markdown(insight)
        else:
            st.info("📊 Performance estável no período analisado.")
        
        # Recomendações estratégicas
        st.markdown("#### Recomendações Estratégicas")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Ações de Curto Prazo:**
            - Monitorar lojas com alta volatilidade
            - Implementar campanhas para lojas em declínio
            - Aproveitar momentum das lojas em crescimento
            """)
        
        with col2:
            st.markdown("""
            **Estratégias de Longo Prazo:**
            - Expandir modelo das lojas eficientes
            - Desenvolver planos sazonais específicos
            - Investir em análise preditiva avançada
            """)


# =============================================================================
# PONTO DE ENTRADA PRINCIPAL
# =============================================================================

def main():
    """Função principal que inicializa o dashboard."""
    try:
        dashboard = DashboardManager()
        dashboard.run()
        
        # Mensagem de sucesso
        st.success("✅ Dashboard carregado com sucesso! Explore os diferentes modos de análise para insights detalhados.")
        
    except Exception as e:
        st.error(f"Erro na inicialização do dashboard: {e}")
        st.info("Verifique se todos os dados necessários estão disponíveis e tente novamente.")


if __name__ == "__main__":
    main()
                "
