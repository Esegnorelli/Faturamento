import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

# Configuração da página
st.set_page_config(
    page_title="Dashboard Hora do Pastel",
    page_icon="🥟",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS customizado
st.markdown("""
<style>
    /* Header principal */
    .main-header {
        font-size: 3rem;
        color: #FF6B35;
        text-align: center;
        margin-bottom: 2rem;
        font-weight: bold;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    /* Container do logo centralizado */
    .logo-container {
        display: flex;
        justify-content: center;
        align-items: center;
        margin-bottom: 1rem;
    }
    
    /* Caixas de insight - compatível com tema escuro */
    .insight-box {
        background: linear-gradient(135deg, #1e3a8a, #3b82f6);
        color: white;
        padding: 1.2rem;
        border-radius: 12px;
        border-left: 4px solid #60a5fa;
        margin: 1rem 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    .insight-box h4 {
        color: #fbbf24 !important;
        margin-bottom: 0.5rem;
    }
    
    /* Caixas de explicação - tema escuro compatível */
    .explanation-box {
        background: linear-gradient(135deg, #d97706, #f59e0b);
        color: #1f2937;
        padding: 1.2rem;
        border-radius: 12px;
        border-left: 4px solid #f97316;
        margin: 1rem 0;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    .explanation-box h4 {
        color: #1f2937 !important;
        margin-bottom: 0.5rem;
        font-weight: bold;
    }
    
    /* Melhorar contraste no tema escuro */
    .stSelectbox label, .stMultiSelect label {
        font-weight: 600;
        color: var(--text-color) !important;
    }
    
    /* Métricas com melhor visual */
    div[data-testid="metric-container"] {
        background: linear-gradient(135deg, #f8fafc, #e2e8f0);
        padding: 1rem;
        border-radius: 12px;
        border-left: 5px solid #FF6B35;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    /* Sidebar melhorada */
    .css-1d391kg {
        background: linear-gradient(180deg, #f1f5f9, #e2e8f0);
    }
    
    /* Botões e elementos interativos */
    .stButton > button {
        background: linear-gradient(135deg, #FF6B35, #ff8f65);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 12px rgba(255, 107, 53, 0.4);
    }
    
    /* Melhorar tabelas */
    .dataframe {
        border-radius: 8px;
        overflow: hidden;
        box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    }
    
    /* Título das seções */
    .stMarkdown h1, .stMarkdown h2 {
        color: #FF6B35;
        border-bottom: 2px solid #FF6B35;
        padding-bottom: 0.5rem;
    }
    
    /* Tema escuro específico */
    @media (prefers-color-scheme: dark) {
        .explanation-box {
            background: linear-gradient(135deg, #92400e, #b45309);
            color: #fbbf24;
        }
        
        .explanation-box h4 {
            color: #fbbf24 !important;
        }
        
        div[data-testid="metric-container"] {
            background: linear-gradient(135deg, #374151, #4b5563);
            color: white;
        }
    }
    
    /* Responsividade */
    @media (max-width: 768px) {
        .main-header {
            font-size: 2rem;
        }
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Carrega e processa os dados de faturamento"""
    try:
        # Lê o arquivo CSV
        df = pd.read_csv('Faturamento.csv', sep=',')
        
        # Remove a primeira coluna se for apenas um índice
        if df.columns[0].isdigit() or 'Unnamed' in df.columns[0]:
            df = df.drop(df.columns[0], axis=1)
        
        # Limpa os nomes das colunas
        df.columns = [col.strip() for col in df.columns]
        
        # Converte tipos de dados
        df['faturamento'] = pd.to_numeric(df['faturamento'], errors='coerce')
        df['pedidos'] = pd.to_numeric(df['pedidos'], errors='coerce')
        df['ticket'] = pd.to_numeric(df['ticket'], errors='coerce')
        df['mes'] = pd.to_numeric(df['mes'], errors='coerce')
        df['ano'] = pd.to_numeric(df['ano'], errors='coerce')
        
        # Cria data completa
        df['data'] = pd.to_datetime(df['periodo'], format='%Y-%m', errors='coerce')
        
        # Remove linhas com dados inválidos
        df = df.dropna(subset=['faturamento', 'pedidos', 'mes', 'ano'])
        
        # Calcula métricas adicionais
        df['ticket_calculado'] = df['faturamento'] / df['pedidos']
        df['receita_por_dia'] = df['faturamento'] / 30  # Aproximação mensal
        
        return df
    except Exception as e:
        st.error(f"Erro ao carregar dados: {e}")
        return pd.DataFrame()

def show_logo():
    """Exibe o logo da empresa centralizado"""
    try:
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            st.image("Logo.png", width=250, use_column_width=False)
    except Exception as e:
        st.markdown(
            '<div style="text-align: center; font-size: 2rem; color: #FF6B35; font-weight: bold; margin: 1rem 0;">'
            '🥟 HORA DO PASTEL'
            '</div>', 
            unsafe_allow_html=True
        )

def calculate_kpis(df_filtered):
    """Calcula KPIs principais"""
    if df_filtered.empty:
        return {}
    
    total_faturamento = df_filtered['faturamento'].sum()
    total_pedidos = df_filtered['pedidos'].sum()
    ticket_medio = total_faturamento / total_pedidos if total_pedidos > 0 else 0
    num_lojas = df_filtered['loja'].nunique()
    faturamento_medio_loja = total_faturamento / num_lojas if num_lojas > 0 else 0
    
    return {
        'total_faturamento': total_faturamento,
        'total_pedidos': total_pedidos,
        'ticket_medio': ticket_medio,
        'num_lojas': num_lojas,
        'faturamento_medio_loja': faturamento_medio_loja
    }

def show_explanation(title, explanation):
    """Mostra caixa de explicação"""
    st.markdown(f"""
    <div class="explanation-box">
        <h4>💡 {title}</h4>
        <p>{explanation}</p>
    </div>
    """, unsafe_allow_html=True)

def show_insight(insight):
    """Mostra caixa de insight"""
    st.markdown(f"""
    <div class="insight-box">
        <h4>📊 Insight</h4>
        <p>{insight}</p>
    </div>
    """, unsafe_allow_html=True)

def analyze_seasonality(df):
    """Analisa sazonalidade dos dados"""
    monthly_revenue = df.groupby('mes')['faturamento'].mean().reset_index()
    monthly_revenue['mes_nome'] = monthly_revenue['mes'].map({
        1: 'Jan', 2: 'Fev', 3: 'Mar', 4: 'Abr', 5: 'Mai', 6: 'Jun',
        7: 'Jul', 8: 'Ago', 9: 'Set', 10: 'Out', 11: 'Nov', 12: 'Dez'
    })
    
    # Identifica picos e vales
    max_month = monthly_revenue.loc[monthly_revenue['faturamento'].idxmax()]
    min_month = monthly_revenue.loc[monthly_revenue['faturamento'].idxmin()]
    
    return monthly_revenue, max_month, min_month

def analyze_growth(df):
    """Analisa crescimento temporal"""
    # Crescimento anual
    yearly_growth = df.groupby('ano').agg({
        'faturamento': 'sum',
        'pedidos': 'sum'
    }).reset_index()
    
    yearly_growth['crescimento_fat'] = yearly_growth['faturamento'].pct_change() * 100
    yearly_growth['crescimento_ped'] = yearly_growth['pedidos'].pct_change() * 100
    
    return yearly_growth

def top_stores_analysis(df):
    """Análise das top lojas"""
    store_performance = df.groupby('loja').agg({
        'faturamento': 'sum',
        'pedidos': 'sum',
        'ticket': 'mean'
    }).reset_index()
    
    store_performance['ticket_calculado'] = store_performance['faturamento'] / store_performance['pedidos']
    store_performance = store_performance.sort_values('faturamento', ascending=False)
    
    return store_performance

def main():
    # Header
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        show_logo()
        st.markdown('<h1 class="main-header">Dashboard de Faturamento</h1>', unsafe_allow_html=True)
    
    # Carrega dados
    df = load_data()
    
    if df.empty:
        st.error("Não foi possível carregar os dados.")
        return
    
    # Sidebar - Filtros
    st.sidebar.header("🔍 Filtros")
    
    # Filtro de anos
    anos_disponiveis = sorted(df['ano'].unique())
    anos_selecionados = st.sidebar.multiselect(
        "Selecione os Anos:",
        anos_disponiveis,
        default=anos_disponiveis[-2:] if len(anos_disponiveis) >= 2 else anos_disponiveis
    )
    
    # Filtro de meses
    meses_nomes = {1: 'Janeiro', 2: 'Fevereiro', 3: 'Março', 4: 'Abril', 5: 'Maio', 6: 'Junho',
                   7: 'Julho', 8: 'Agosto', 9: 'Setembro', 10: 'Outubro', 11: 'Novembro', 12: 'Dezembro'}
    
    meses_disponiveis = sorted(df['mes'].unique())
    meses_selecionados = st.sidebar.multiselect(
        "Selecione os Meses:",
        meses_disponiveis,
        format_func=lambda x: meses_nomes.get(x, x),
        default=meses_disponiveis
    )
    
    # Filtro de lojas
    lojas_disponiveis = sorted(df['loja'].unique())
    lojas_selecionadas = st.sidebar.multiselect(
        "Selecione as Lojas:",
        lojas_disponiveis,
        default=lojas_disponiveis
    )
    
    # Aplica filtros
    df_filtered = df[
        (df['ano'].isin(anos_selecionados)) &
        (df['mes'].isin(meses_selecionados)) &
        (df['loja'].isin(lojas_selecionadas))
    ]
    
    if df_filtered.empty:
        st.warning("Nenhum dado encontrado com os filtros selecionados.")
        return
    
    # KPIs Principais
    st.header("📈 KPIs Principais")
    kpis = calculate_kpis(df_filtered)
    
    col1, col2, col3, col4, col5 = st.columns(5)
    
    with col1:
        st.metric(
            "Faturamento Total",
            f"R$ {kpis['total_faturamento']:,.2f}",
            help="Soma de todo o faturamento no período selecionado"
        )
    
    with col2:
        st.metric(
            "Total de Pedidos",
            f"{kpis['total_pedidos']:,.0f}",
            help="Quantidade total de pedidos realizados"
        )
    
    with col3:
        st.metric(
            "Ticket Médio",
            f"R$ {kpis['ticket_medio']:.2f}",
            help="Faturamento Total ÷ Total de Pedidos"
        )
    
    with col4:
        st.metric(
            "Nº de Lojas",
            f"{kpis['num_lojas']}",
            help="Número de lojas ativas no período"
        )
    
    with col5:
        st.metric(
            "Faturamento Médio/Loja",
            f"R$ {kpis['faturamento_medio_loja']:,.2f}",
            help="Faturamento Total ÷ Número de Lojas"
        )
    
    show_explanation(
        "Cálculo dos KPIs",
        "Os KPIs são calculados com base nos dados filtrados. O Ticket Médio é obtido dividindo o Faturamento Total pelo Total de Pedidos, representando o valor médio gasto por cliente. O Faturamento Médio por Loja indica a performance média das unidades no período."
    )
    
    # Análise Temporal
    st.header("📅 Análise Temporal")
    
    # Evolução mensal
    df_temporal = df_filtered.groupby(['ano', 'mes']).agg({
        'faturamento': 'sum',
        'pedidos': 'sum'
    }).reset_index()
    df_temporal['periodo'] = df_temporal['ano'].astype(str) + '-' + df_temporal['mes'].astype(str).str.zfill(2)
    df_temporal['ticket_medio'] = df_temporal['faturamento'] / df_temporal['pedidos']
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig_fat = px.line(
            df_temporal, 
            x='periodo', 
            y='faturamento',
            title='Evolução do Faturamento',
            labels={'faturamento': 'Faturamento (R$)', 'periodo': 'Período'},
            color_discrete_sequence=['#FF6B35']
        )
        fig_fat.update_layout(
            showlegend=False,
            yaxis_tickformat=',.0f',
            xaxis={'tickangle': 45}
        )
        fig_fat.update_traces(hovertemplate='<b>%{x}</b><br>Faturamento: R$ %{y:,.2f}<extra></extra>')
        st.plotly_chart(fig_fat, use_container_width=True)
    
    with col2:
        fig_ped = px.line(
            df_temporal, 
            x='periodo', 
            y='pedidos',
            title='Evolução dos Pedidos',
            labels={'pedidos': 'Quantidade de Pedidos', 'periodo': 'Período'},
            color_discrete_sequence=['#1f77b4']
        )
        fig_ped.update_layout(
            showlegend=False,
            xaxis={'tickangle': 45}
        )
        fig_ped.update_traces(hovertemplate='<b>%{x}</b><br>Pedidos: %{y:,.0f}<extra></extra>')
        st.plotly_chart(fig_ped, use_container_width=True)
    
    # Análise de Sazonalidade
    st.subheader("🌱 Análise de Sazonalidade")
    monthly_data, max_month, min_month = analyze_seasonality(df_filtered)
    
    fig_season = px.bar(
        monthly_data,
        x='mes_nome',
        y='faturamento',
        title='Faturamento Médio por Mês',
        labels={'faturamento': 'Faturamento Médio (R$)', 'mes_nome': 'Mês'},
        color='faturamento',
        color_continuous_scale='Viridis'
    )
    st.plotly_chart(fig_season, use_container_width=True)
    
    show_insight(
        f"O mês de **{max_month['mes_nome']}** apresenta o maior faturamento médio (R$ {max_month['faturamento']:,.2f}), "
        f"enquanto **{min_month['mes_nome']}** tem o menor (R$ {min_month['faturamento']:,.2f}). "
        f"Isso representa uma variação de {((max_month['faturamento'] - min_month['faturamento']) / min_month['faturamento'] * 100):.1f}% entre pico e vale."
    )
    
    # Análise de Crescimento
    st.subheader("📈 Análise de Crescimento")
    growth_data = analyze_growth(df_filtered)
    
    if len(growth_data) > 1:
        col1, col2 = st.columns(2)
        
        with col1:
            fig_growth_fat = px.bar(
                growth_data.dropna(),
                x='ano',
                y='crescimento_fat',
                title='Crescimento Anual do Faturamento (%)',
                labels={'crescimento_fat': 'Crescimento (%)', 'ano': 'Ano'},
                color='crescimento_fat',
                color_continuous_scale='RdYlGn'
            )
            st.plotly_chart(fig_growth_fat, use_container_width=True)
        
        with col2:
            fig_growth_ped = px.bar(
                growth_data.dropna(),
                x='ano',
                y='crescimento_ped',
                title='Crescimento Anual dos Pedidos (%)',
                labels={'crescimento_ped': 'Crescimento (%)', 'ano': 'Ano'},
                color='crescimento_ped',
                color_continuous_scale='RdYlGn'
            )
            st.plotly_chart(fig_growth_ped, use_container_width=True)
    
    show_explanation(
        "Análise de Crescimento",
        "O crescimento é calculado usando a fórmula: ((Valor Atual - Valor Anterior) / Valor Anterior) × 100. "
        "Valores positivos indicam crescimento, negativos indicam queda. Esta análise ajuda a identificar tendências de longo prazo."
    )
    
    # Performance por Loja
    st.header("🏪 Performance por Loja")
    
    store_data = top_stores_analysis(df_filtered)
    
    # Top 10 lojas por faturamento
    col1, col2 = st.columns(2)
    
    with col1:
        top_10_stores = store_data.head(10)
        fig_top_stores = px.bar(
            top_10_stores,
            x='faturamento',
            y='loja',
            orientation='h',
            title='Top 10 Lojas por Faturamento',
            labels={'faturamento': 'Faturamento (R$)', 'loja': 'Loja'},
            color='faturamento',
            color_continuous_scale='Oranges'
        )
        fig_top_stores.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_top_stores, use_container_width=True)
    
    with col2:
        # Ticket médio por loja (top 10)
        top_ticket = store_data.nlargest(10, 'ticket_calculado')
        fig_ticket = px.bar(
            top_ticket,
            x='ticket_calculado',
            y='loja',
            orientation='h',
            title='Top 10 Lojas por Ticket Médio',
            labels={'ticket_calculado': 'Ticket Médio (R$)', 'loja': 'Loja'},
            color='ticket_calculado',
            color_continuous_scale='Blues'
        )
        fig_ticket.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_ticket, use_container_width=True)
    
    # Análise de correlação
    st.subheader("🔗 Análise de Correlação")
    correlation_data = df_filtered.groupby('loja').agg({
        'faturamento': 'sum',
        'pedidos': 'sum',
        'ticket': 'mean'
    }).reset_index()
    
    fig_scatter = px.scatter(
        correlation_data,
        x='pedidos',
        y='faturamento',
        size='ticket',
        hover_name='loja',
        title='Relação entre Pedidos e Faturamento por Loja',
        labels={'pedidos': 'Total de Pedidos', 'faturamento': 'Faturamento Total (R$)', 'ticket': 'Ticket Médio'},
        color='ticket',
        color_continuous_scale='Viridis'
    )
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    show_explanation(
        "Análise de Correlação",
        "Este gráfico mostra a relação entre volume de pedidos e faturamento. O tamanho das bolhas representa o ticket médio. "
        "Lojas no canto superior direito têm alta performance em ambas as métricas. Lojas com bolhas grandes têm tickets médios elevados."
    )
    
    # Mapa de Calor - Performance Temporal por Loja
    st.header("🗺️ Mapa de Calor - Performance Temporal")
    
    # Prepara dados para heatmap
    heatmap_data = df_filtered.pivot_table(
        values='faturamento',
        index='loja',
        columns='periodo',
        aggfunc='sum',
        fill_value=0
    )
    
    # Seleciona top 15 lojas para melhor visualização
    top_stores_list = store_data.head(15)['loja'].tolist()
    heatmap_filtered = heatmap_data.loc[heatmap_data.index.isin(top_stores_list)]
    
    fig_heatmap = px.imshow(
        heatmap_filtered.values,
        x=heatmap_filtered.columns,
        y=heatmap_filtered.index,
        title='Mapa de Calor: Faturamento por Loja e Período',
        labels={'x': 'Período', 'y': 'Loja', 'color': 'Faturamento (R$)'},
        color_continuous_scale='Oranges'
    )
    fig_heatmap.update_layout(height=600)
    st.plotly_chart(fig_heatmap, use_container_width=True)
    
    show_explanation(
        "Mapa de Calor",
        "Cores mais escuras indicam maior faturamento. Este mapa permite identificar padrões sazonais específicos por loja "
        "e períodos de alta/baixa performance. Útil para planejamento de campanhas e alocação de recursos."
    )
    
    # Análise Comparativa
    st.header("⚖️ Análise Comparativa")
    
    # Comparação entre anos
    if len(anos_selecionados) >= 2:
        comparison_data = df_filtered.groupby(['ano', 'mes']).agg({
            'faturamento': 'sum',
            'pedidos': 'sum'
        }).reset_index()
        
        fig_comparison = px.line(
            comparison_data,
            x='mes',
            y='faturamento',
            color='ano',
            title='Comparação de Faturamento por Mês entre Anos',
            labels={'faturamento': 'Faturamento (R$)', 'mes': 'Mês', 'ano': 'Ano'},
            markers=True
        )
        fig_comparison.update_xaxes(tickmode='linear', tick0=1, dtick=1)
        st.plotly_chart(fig_comparison, use_container_width=True)
        
        # Cálculo de variação
        if len(anos_selecionados) == 2:
            ano_atual = max(anos_selecionados)
            ano_anterior = min(anos_selecionados)
            
            fat_atual = df_filtered[df_filtered['ano'] == ano_atual]['faturamento'].sum()
            fat_anterior = df_filtered[df_filtered['ano'] == ano_anterior]['faturamento'].sum()
            
            if fat_anterior > 0:
                variacao = ((fat_atual - fat_anterior) / fat_anterior) * 100
                cor_variacao = "🟢" if variacao > 0 else "🔴"
                
                show_insight(
                    f"{cor_variacao} Comparando {ano_anterior} vs {ano_atual}: "
                    f"{'Crescimento' if variacao > 0 else 'Queda'} de {abs(variacao):.1f}% no faturamento "
                    f"(R$ {fat_anterior:,.2f} → R$ {fat_atual:,.2f})"
                )
    
    # Distribuição de Faturamento
    st.subheader("📊 Distribuição de Faturamento")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Gráfico de pizza - top lojas
        top_5_stores = store_data.head(5)
        outros_faturamento = store_data.iloc[5:]['faturamento'].sum() if len(store_data) > 5 else 0
        
        pie_data = top_5_stores[['loja', 'faturamento']].copy()
        if outros_faturamento > 0:
            pie_data = pd.concat([pie_data, pd.DataFrame({'loja': ['Outras'], 'faturamento': [outros_faturamento]})], ignore_index=True)
        
        fig_pie = px.pie(
            pie_data,
            values='faturamento',
            names='loja',
            title='Distribuição do Faturamento - Top 5 Lojas',
            color_discrete_sequence=px.colors.qualitative.Set3
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Distribuição do ticket médio
        fig_hist = px.histogram(
            store_data,
            x='ticket_calculado',
            nbins=20,
            title='Distribuição do Ticket Médio por Loja',
            labels={'ticket_calculado': 'Ticket Médio (R$)', 'count': 'Quantidade de Lojas'},
            color_discrete_sequence=['#FF6B35']
        )
        st.plotly_chart(fig_hist, use_container_width=True)
    
    # Ranking de Performance
    st.header("🏆 Ranking de Performance")
    
    # Cria métricas de performance
    store_ranking = store_data.copy()
    store_ranking['rank_faturamento'] = store_ranking['faturamento'].rank(ascending=False)
    store_ranking['rank_pedidos'] = store_ranking['pedidos'].rank(ascending=False)
    store_ranking['rank_ticket'] = store_ranking['ticket_calculado'].rank(ascending=False)
    store_ranking['score_geral'] = (
        store_ranking['rank_faturamento'] + 
        store_ranking['rank_pedidos'] + 
        store_ranking['rank_ticket']
    ) / 3
    store_ranking = store_ranking.sort_values('score_geral')
    
    # Mostra tabela de ranking
    ranking_display = store_ranking[['loja', 'faturamento', 'pedidos', 'ticket_calculado']].head(10)
    ranking_display.columns = ['Loja', 'Faturamento (R$)', 'Pedidos', 'Ticket Médio (R$)']
    ranking_display['Faturamento (R$)'] = ranking_display['Faturamento (R$)'].apply(lambda x: f"R$ {x:,.2f}")
    ranking_display['Pedidos'] = ranking_display['Pedidos'].apply(lambda x: f"{x:,.0f}")
    ranking_display['Ticket Médio (R$)'] = ranking_display['Ticket Médio (R$)'].apply(lambda x: f"R$ {x:.2f}")
    
    st.dataframe(ranking_display, use_container_width=True, hide_index=True)
    
    show_explanation(
        "Ranking de Performance",
        "O ranking é calculado considerando três dimensões: faturamento total, quantidade de pedidos e ticket médio. "
        "Cada loja recebe um score baseado na média dos rankings individuais. Lojas no topo equilibram bem volume e valor."
    )
    
    # Análise de Tendências
    st.header("📈 Análise de Tendências")
    
    # Tendência de crescimento por loja
    store_trends = []
    for loja in df_filtered['loja'].unique():
        loja_data = df_filtered[df_filtered['loja'] == loja].copy()
        if len(loja_data) >= 3:  # Mínimo para calcular tendência
            loja_data = loja_data.sort_values('periodo')
            loja_data['periodo_num'] = range(len(loja_data))
            
            # Regressão linear simples
            x = loja_data['periodo_num'].values
            y = loja_data['faturamento'].values
            
            if len(x) > 1 and np.std(x) > 0:
                coef = np.polyfit(x, y, 1)
                tendencia = coef[0]  # Coeficiente angular
                
                store_trends.append({
                    'loja': loja,
                    'tendencia': tendencia,
                    'faturamento_medio': loja_data['faturamento'].mean(),
                    'periodos': len(loja_data)
                })
    
    if store_trends:
        trends_df = pd.DataFrame(store_trends)
        trends_df['tendencia_normalizada'] = trends_df['tendencia'] / trends_df['faturamento_medio'] * 100
        trends_df = trends_df.sort_values('tendencia_normalizada', ascending=False)
        
        # Gráfico de tendências
        fig_trends = px.bar(
            trends_df.head(15),
            x='tendencia_normalizada',
            y='loja',
            orientation='h',
            title='Tendência de Crescimento por Loja (% normalizado)',
            labels={'tendencia_normalizada': 'Tendência de Crescimento (%)', 'loja': 'Loja'},
            color='tendencia_normalizada',
            color_continuous_scale='RdYlGn'
        )
        fig_trends.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_trends, use_container_width=True)
        
        # Insights sobre tendências
        crescimento_lojas = trends_df[trends_df['tendencia_normalizada'] > 0]
        declinio_lojas = trends_df[trends_df['tendencia_normalizada'] < 0]
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric(
                "Lojas em Crescimento",
                len(crescimento_lojas),
                f"{len(crescimento_lojas)/len(trends_df)*100:.1f}% do total"
            )
        
        with col2:
            st.metric(
                "Lojas em Declínio",
                len(declinio_lojas),
                f"{len(declinio_lojas)/len(trends_df)*100:.1f}% do total"
            )
    
    show_explanation(
        "Análise de Tendências",
        "A tendência é calculada usando regressão linear sobre os períodos. O valor é normalizado pelo faturamento médio "
        "para permitir comparação justa entre lojas de diferentes tamanhos. Valores positivos indicam crescimento consistente."
    )
    
    # Análise de Eficiência
    st.header("⚡ Análise de Eficiência")
    
    # Eficiência = Faturamento / Pedidos (Ticket Médio)
    efficiency_data = store_data.copy()
    efficiency_data['eficiencia_score'] = (
        (efficiency_data['faturamento'] / efficiency_data['faturamento'].max()) * 0.4 +
        (efficiency_data['pedidos'] / efficiency_data['pedidos'].max()) * 0.3 +
        (efficiency_data['ticket_calculado'] / efficiency_data['ticket_calculado'].max()) * 0.3
    ) * 100
    
    efficiency_data = efficiency_data.sort_values('eficiencia_score', ascending=False)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        fig_efficiency = px.bar(
            efficiency_data.head(15),
            x='eficiencia_score',
            y='loja',
            orientation='h',
            title='Score de Eficiência por Loja',
            labels={'eficiencia_score': 'Score de Eficiência (0-100)', 'loja': 'Loja'},
            color='eficiencia_score',
            color_continuous_scale='Viridis'
        )
        fig_efficiency.update_layout(yaxis={'categoryorder': 'total ascending'})
        st.plotly_chart(fig_efficiency, use_container_width=True)
    
    with col2:
        st.subheader("Top 5 Mais Eficientes")
        for i, row in efficiency_data.head(5).iterrows():
            st.markdown(f"""
            **{i+1}. {row['loja']}**  
            Score: {row['eficiencia_score']:.1f}  
            Faturamento: R$ {row['faturamento']:,.0f}  
            Ticket: R$ {row['ticket_calculado']:.2f}
            """)
    
    show_explanation(
        "Score de Eficiência",
        "O score combina três métricas: Faturamento (peso 40%), Volume de Pedidos (peso 30%) e Ticket Médio (peso 30%). "
        "Este indicador equilibra volume e valor, identificando lojas que maximizam receita com boa experiência do cliente."
    )
    
    # Insights Finais e Recomendações
    st.header("💡 Insights e Recomendações")
    
    # Calcula insights automáticos
    total_lojas = len(store_data)
    loja_top_faturamento = store_data.iloc[0]
    loja_top_ticket = store_data.nlargest(1, 'ticket_calculado').iloc[0]
    ticket_medio_geral = store_data['ticket_calculado'].mean()
    
    insights = [
        f"🎯 **Loja Líder**: {loja_top_faturamento['loja']} domina com R$ {loja_top_faturamento['faturamento']:,.2f} em faturamento",
        f"💰 **Maior Ticket**: {loja_top_ticket['loja']} tem o maior ticket médio (R$ {loja_top_ticket['ticket_calculado']:.2f})",
        f"📊 **Ticket Médio da Rede**: R$ {ticket_medio_geral:.2f}",
        f"🏪 **Rede**: {total_lojas} lojas ativas no período analisado"
    ]
    
    if len(growth_data.dropna()) > 0:
        ultimo_crescimento = growth_data.dropna().iloc[-1]['crescimento_fat']
        if ultimo_crescimento > 0:
            insights.append(f"📈 **Crescimento Positivo**: {ultimo_crescimento:.1f}% no último período analisado")
        else:
            insights.append(f"📉 **Atenção**: Queda de {abs(ultimo_crescimento):.1f}% no último período analisado")
    
    for insight in insights:
        st.markdown(insight)
    
    # Recomendações estratégicas
    st.subheader("🎯 Recomendações Estratégicas")
    
    recomendacoes = []
    
    # Análise de lojas com baixa performance
    low_performance = store_data[store_data['faturamento'] < store_data['faturamento'].quantile(0.25)]
    if not low_performance.empty:
        recomendacoes.append(
            f"📍 **Otimização**: {len(low_performance)} lojas estão no quartil inferior de faturamento. "
            f"Considere ações de marketing local ou revisão operacional."
        )
    
    # Análise de ticket médio
    low_ticket = store_data[store_data['ticket_calculado'] < ticket_medio_geral * 0.9]
    if not low_ticket.empty:
        recomendacoes.append(
            f"🍽️ **Upselling**: {len(low_ticket)} lojas têm ticket médio abaixo da média da rede. "
            f"Implemente estratégias de upselling e cross-selling."
        )
    
    # Análise sazonal
    if not monthly_data.empty:
        meses_baixos = monthly_data[monthly_data['faturamento'] < monthly_data['faturamento'].mean() * 0.9]
        if not meses_baixos.empty:
            meses_nomes_baixos = [meses_nomes[m] for m in meses_baixos['mes'].tolist()]
            recomendacoes.append(
                f"📅 **Sazonalidade**: Meses de baixa performance ({', '.join(meses_nomes_baixos)}). "
                f"Desenvolva campanhas promocionais específicas para estes períodos."
            )
    
    for rec in recomendacoes:
        st.markdown(rec)
    
    # Métricas Avançadas
    st.header("🧮 Métricas Avançadas")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Coeficiente de variação
        cv_faturamento = (store_data['faturamento'].std() / store_data['faturamento'].mean()) * 100
        st.metric(
            "Coeficiente de Variação",
            f"{cv_faturamento:.1f}%",
            help="Medida de dispersão relativa do faturamento entre lojas. Valores altos indicam grande variabilidade."
        )
    
    with col2:
        # Índice de concentração (Gini simplificado)
        faturamentos_ordenados = np.sort(store_data['faturamento'].values)
        n = len(faturamentos_ordenados)
        cumsum = np.cumsum(faturamentos_ordenados)
        gini = (2 * np.sum((np.arange(1, n+1) * faturamentos_ordenados))) / (n * cumsum[-1]) - (n+1)/n
        
        st.metric(
            "Índice de Concentração",
            f"{gini:.3f}",
            help="Varia de 0 (distribuição uniforme) a 1 (máxima concentração). Indica se o faturamento está concentrado em poucas lojas."
        )
    
    with col3:
        # ROI médio (simplificado como faturamento/pedidos)
        roi_medio = store_data['ticket_calculado'].mean()
        st.metric(
            "ROI Médio Estimado",
            f"R$ {roi_medio:.2f}",
            help="Retorno médio por transação (ticket médio). Indicador de eficiência comercial."
        )
    
    show_explanation(
        "Métricas Avançadas",
        "**Coeficiente de Variação**: Mede a dispersão dos dados em relação à média. "
        "**Índice de Concentração**: Baseado no coeficiente de Gini, indica se poucas lojas concentram a maior parte do faturamento. "
        "**ROI Estimado**: Representa o retorno médio por transação, fundamental para avaliar eficiência comercial."
    )
    
    # Dados Detalhados
    with st.expander("📋 Ver Dados Detalhados"):
        st.subheader("Resumo por Loja")
        detailed_data = store_data.copy()
        detailed_data['Faturamento'] = detailed_data['faturamento'].apply(lambda x: f"R$ {x:,.2f}")
        detailed_data['Pedidos'] = detailed_data['pedidos'].apply(lambda x: f"{x:,.0f}")
        detailed_data['Ticket Médio'] = detailed_data['ticket_calculado'].apply(lambda x: f"R$ {x:.2f}")
        
        st.dataframe(
            detailed_data[['loja', 'Faturamento', 'Pedidos', 'Ticket Médio']].rename(columns={'loja': 'Loja'}),
            use_container_width=True,
            hide_index=True
        )
        
        st.subheader("Dados Brutos Filtrados")
        st.dataframe(df_filtered, use_container_width=True, hide_index=True)

if __name__ == "__main__":
    main()
