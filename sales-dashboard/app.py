import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import seaborn as sns
from scipy import stats
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import statsmodels.api as sm
from statsmodels.tsa.seasonal import seasonal_decompose
from datetime import datetime, timedelta
import warnings
import os

warnings.filterwarnings('ignore')

# Configuração da página
st.set_page_config(
    page_title="Analytics Dashboard - Faturamento",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS customizado para melhorar a aparência
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .insight-box {
        background: #f8f9ff;
        padding: 1rem;
        border-left: 4px solid #667eea;
        border-radius: 5px;
        margin: 1rem 0;
    }
    .alert-box {
        background: #fff3cd;
        border: 1px solid #ffeaa7;
        padding: 1rem;
        border-radius: 5px;
        color: #856404;
    }
    .success-box {
        background: #d1edff;
        border: 1px solid #74b9ff;
        padding: 1rem;
        border-radius: 5px;
        color: #0984e3;
    }
</style>
""", unsafe_allow_html=True)

# Função para carregar e processar dados
@st.cache_data
def load_and_process_data():
    """Carrega e processa os dados com análises avançadas"""
    try:
        df = pd.read_csv('public/Faturamento.csv')
        
        # Converter tipos
        df['faturamento'] = pd.to_numeric(df['faturamento'], errors='coerce')
        df['pedidos'] = pd.to_numeric(df['pedidos'], errors='coerce')
        df['ticket'] = pd.to_numeric(df['ticket'], errors='coerce')
        df['data'] = pd.to_datetime(df['periodo'], format='%Y-%m', errors='coerce')
        
        # Remover dados inválidos
        df = df.dropna(subset=['faturamento', 'pedidos', 'ticket'])
        
        # Calcular métricas derivadas
        df['receita_por_pedido'] = df['faturamento'] / df['pedidos']
        df['eficiencia_vendas'] = df['pedidos'] / df['faturamento'] * 1000  # Pedidos por R$ 1000
        df['trimestre'] = df['data'].dt.quarter
        df['semestre'] = df['data'].dt.to_period('6M')
        
        # Análise de crescimento
        df = df.sort_values(['loja', 'data'])
        df['crescimento_mensal'] = df.groupby('loja')['faturamento'].pct_change() * 100
        df['media_movel_3m'] = df.groupby('loja')['faturamento'].rolling(window=3, min_periods=1).mean().reset_index(0, drop=True)
        
        # Classificação de performance
        df['performance_score'] = (
            (df['faturamento'] / df['faturamento'].max()) * 0.4 +
            (df['pedidos'] / df['pedidos'].max()) * 0.3 +
            (df['ticket'] / df['ticket'].max()) * 0.3
        ) * 100
        
        return df
    except Exception as e:
        st.error(f"Erro ao carregar dados: {e}")
        return pd.DataFrame()

# Função para análise de cluster
@st.cache_data
def perform_cluster_analysis(df):
    """Realiza análise de clustering das lojas"""
    # Preparar dados para clustering
    features = ['faturamento', 'pedidos', 'ticket']
    loja_metrics = df.groupby('loja')[features].agg(['mean', 'sum', 'std']).round(2)
    loja_metrics.columns = ['_'.join(col).strip() for col in loja_metrics.columns]
    
    # Normalizar dados
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(loja_metrics)
    
    # K-means clustering
    kmeans = KMeans(n_clusters=4, random_state=42)
    clusters = kmeans.fit_predict(scaled_data)
    
    loja_metrics['cluster'] = clusters
    loja_metrics['loja'] = loja_metrics.index
    
    # Definir nomes dos clusters
    cluster_names = {
        0: "🏆 High Performers",
        1: "🎯 Steady Growth", 
        2: "🚀 Rising Stars",
        3: "⚠️ Attention Needed"
    }
    
    loja_metrics['cluster_name'] = loja_metrics['cluster'].map(cluster_names)
    
    return loja_metrics

# Função para previsão simples
@st.cache_data
def simple_forecast(df, loja_selecionada, meses_previsao=6):
    """Faz previsão simples usando regressão linear"""
    loja_data = df[df['loja'] == loja_selecionada].copy()
    loja_data = loja_data.sort_values('data')
    
    if len(loja_data) < 3:
        return None, None
    
    # Preparar dados para regressão
    loja_data['mes_numero'] = range(len(loja_data))
    X = loja_data[['mes_numero']]
    y = loja_data['faturamento']
    
    # Treinar modelo
    model = LinearRegression()
    model.fit(X, y)
    
    # Fazer previsões
    futuro_X = np.array(range(len(loja_data), len(loja_data) + meses_previsao)).reshape(-1, 1)
    previsoes = model.predict(futuro_X)
    
    # Calcular R²
    r2 = r2_score(y, model.predict(X))
    
    return previsoes, r2

# Função para exibir logo
def show_logo():
    """Exibe o logo se existir"""
    if os.path.exists('logo.png'):
        st.image('logo.png', width=180)

# Função para calcular KPIs interativos
def display_interactive_kpis(df_filtered):
    """Exibe KPIs com tooltips interativos"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_faturamento = df_filtered['faturamento'].sum()
        crescimento = df_filtered.groupby('ano')['faturamento'].sum().pct_change().iloc[-1] * 100 if len(df_filtered['ano'].unique()) > 1 else 0
        
        st.metric(
            "💰 Faturamento Total",
            f"R$ {total_faturamento:,.0f}".replace(',', '.'),
            f"{crescimento:+.1f}%" if not np.isnan(crescimento) else "N/A",
            help=f"""
            **Análise do Faturamento:**
            • Representa 100% da receita no período selecionado
            • Média mensal: R$ {total_faturamento/max(1, len(df_filtered)):,.0f}
            • Desvio padrão: R$ {df_filtered['faturamento'].std():,.0f}
            • Coeficiente de variação: {(df_filtered['faturamento'].std()/df_filtered['faturamento'].mean()*100):.1f}%
            """
        )
    
    with col2:
        total_pedidos = df_filtered['pedidos'].sum()
        pedidos_por_mes = total_pedidos / max(1, len(df_filtered))
        
        st.metric(
            "🛒 Total de Pedidos",
            f"{total_pedidos:,.0f}".replace(',', '.'),
            f"{pedidos_por_mes:.0f} p/mês",
            help=f"""
            **Análise de Pedidos:**
            • Volume total de transações no período
            • Média por registro: {pedidos_por_mes:.1f} pedidos
            • Máximo em um período: {df_filtered['pedidos'].max():,.0f}
            • Concentração: {(df_filtered['pedidos'].std()/df_filtered['pedidos'].mean()*100):.1f}% variação
            """
        )
    
    with col3:
        ticket_medio = df_filtered['ticket'].mean()
        ticket_mediana = df_filtered['ticket'].median()
        
        st.metric(
            "🎯 Ticket Médio",
            f"R$ {ticket_medio:.2f}".replace('.', ','),
            f"Mediana: R$ {ticket_mediana:.2f}".replace('.', ','),
            help=f"""
            **Análise do Ticket Médio:**
            • Valor médio por pedido no período
            • Mediana: R$ {ticket_mediana:.2f} (mais estável que média)
            • Amplitude: R$ {df_filtered['ticket'].max() - df_filtered['ticket'].min():.2f}
            • 75% dos tickets são até: R$ {df_filtered['ticket'].quantile(0.75):.2f}
            """
        )
    
    with col4:
        eficiencia_media = df_filtered['eficiencia_vendas'].mean()
        roi_vendas = (total_faturamento / total_pedidos) if total_pedidos > 0 else 0
        
        st.metric(
            "⚡ Eficiência de Vendas",
            f"{eficiencia_media:.2f}",
            f"ROI: {roi_vendas:.1f}x",
            help=f"""
            **Análise de Eficiência:**
            • Pedidos por R$ 1.000 de faturamento
            • ROI: Retorno por pedido processado
            • Benchmark interno: {df_filtered['eficiencia_vendas'].median():.2f}
            • Lojas mais eficientes têm menor índice (mais faturamento por pedido)
            """
        )

# Função para análise de sazonalidade
def analyze_seasonality(df):
    """Analisa padrões sazonais nos dados"""
    sazonalidade = df.groupby('mes').agg({
        'faturamento': ['sum', 'mean', 'count'],
        'pedidos': 'sum',
        'ticket': 'mean'
    }).round(2)
    
    sazonalidade.columns = ['fat_total', 'fat_medio', 'registros', 'pedidos_total', 'ticket_medio']
    sazonalidade['mes'] = sazonalidade.index
    
    # Identificar padrões
    melhor_mes = sazonalidade['fat_total'].idxmax()
    pior_mes = sazonalidade['fat_total'].idxmin()
    
    return sazonalidade, melhor_mes, pior_mes

# Função para detecção de anomalias
def detect_anomalies(df):
    """Detecta anomalias nos dados usando Z-score"""
    anomalies = []
    
    for loja in df['loja'].unique():
        loja_data = df[df['loja'] == loja].copy()
        if len(loja_data) > 3:
            z_scores = np.abs(stats.zscore(loja_data['faturamento']))
            outliers = loja_data[z_scores > 2.5]  # Z-score > 2.5 indica anomalia
            
            for idx, row in outliers.iterrows():
                anomalies.append({
                    'loja': loja,
                    'periodo': row['periodo'],
                    'faturamento': row['faturamento'],
                    'z_score': z_scores[loja_data.index.get_loc(idx)],
                    'tipo': 'Alto' if row['faturamento'] > loja_data['faturamento'].mean() else 'Baixo'
                })
    
    return pd.DataFrame(anomalies)

# Função para calcular Gini
def calculate_gini(x):
    """Calcula o coeficiente de Gini"""
    sorted_x = np.sort(x)
    n = len(x)
    cumsum = np.cumsum(sorted_x)
    return (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n

# Função principal
def main():
    # Header com logo e título
    col1, col2 = st.columns([1, 5])
    
    with col1:
        show_logo()
    
    with col2:
        st.title("📊 Analytics Dashboard - Faturamento")
        st.markdown("### 🚀 Análise Inteligente de Performance de Vendas")
        st.markdown("*Dashboard avançado com insights baseados em dados e machine learning*")
    
    # Carregar dados
    with st.spinner("🔄 Carregando e processando dados..."):
        df = load_and_process_data()
    
    if df.empty:
        st.error("❌ Não foi possível carregar os dados do arquivo CSV.")
        return
    
    # Sidebar com filtros avançados
    st.sidebar.header("🎛️ Controles Avançados")
    
    # Filtros principais
    anos_disponiveis = sorted(df['ano'].unique())
    ano_selecionado = st.sidebar.selectbox(
        "📅 Ano de Análise:",
        options=['Todos os Anos'] + [str(ano) for ano in anos_disponiveis],
        index=0
    )
    
    lojas_disponiveis = sorted(df['loja'].unique())
    lojas_selecionadas = st.sidebar.multiselect(
        "🏪 Lojas para Análise:",
        options=lojas_disponiveis,
        default=lojas_disponiveis[:8],
        help="Selecione até 8 lojas para melhor visualização"
    )
    
    # Filtros avançados
    st.sidebar.subheader("🔍 Filtros Avançados")
    
    ticket_range = st.sidebar.slider(
        "💳 Faixa de Ticket Médio:",
        min_value=float(df['ticket'].min()),
        max_value=float(df['ticket'].max()),
        value=(float(df['ticket'].min()), float(df['ticket'].max())),
        step=1.0,
        help="Filtre por faixa de ticket médio para análise de segmentação"
    )
    
    faturamento_min = st.sidebar.number_input(
        "💰 Faturamento Mínimo:",
        min_value=0,
        value=0,
        help="Filtre lojas com faturamento mínimo"
    )
    
    # Aplicar filtros
    df_filtered = df.copy()
    
    if ano_selecionado != 'Todos os Anos':
        df_filtered = df_filtered[df_filtered['ano'] == int(ano_selecionado)]
    
    if lojas_selecionadas:
        df_filtered = df_filtered[df_filtered['loja'].isin(lojas_selecionadas)]
    
    df_filtered = df_filtered[
        (df_filtered['ticket'] >= ticket_range[0]) & 
        (df_filtered['ticket'] <= ticket_range[1]) &
        (df_filtered['faturamento'] >= faturamento_min)
    ]
    
    # KPIs Interativos
    st.header("📈 KPIs Inteligentes")
    display_interactive_kpis(df_filtered)
    
    # Análise de clusters
    st.header("🎯 Segmentação Inteligente de Lojas")
    
    with st.spinner("🤖 Executando análise de clustering..."):
        cluster_data = perform_cluster_analysis(df)
    
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Gráfico de clusters
        fig_cluster = px.scatter(
            cluster_data,
            x='faturamento_sum',
            y='ticket_mean',
            color='cluster_name',
            size='pedidos_sum',
            hover_name='loja',
            title="🔬 Segmentação de Lojas por Performance",
            labels={
                'faturamento_sum': 'Faturamento Total (R$)',
                'ticket_mean': 'Ticket Médio (R$)',
                'pedidos_sum': 'Total de Pedidos'
            }
        )
        fig_cluster.update_layout(height=500)
        st.plotly_chart(fig_cluster, use_container_width=True)
    
    with col2:
        st.subheader("📊 Perfil dos Clusters")
        for cluster_name in cluster_data['cluster_name'].unique():
            cluster_lojas = cluster_data[cluster_data['cluster_name'] == cluster_name]
            avg_fat = cluster_lojas['faturamento_sum'].mean()
            avg_ticket = cluster_lojas['ticket_mean'].mean()
            
            st.markdown(f"""
            <div class="insight-box">
                <strong>{cluster_name}</strong><br>
                📍 {len(cluster_lojas)} lojas<br>
                💰 Faturamento médio: R$ {avg_fat:,.0f}<br>
                🎯 Ticket médio: R$ {avg_ticket:.2f}
            </div>
            """, unsafe_allow_html=True)
    
    # Análise temporal avançada
    st.header("⏰ Análise Temporal Inteligente")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Decomposição sazonal
        st.subheader("📈 Tendência e Sazonalidade")
        df_temporal = df_filtered.groupby('data')['faturamento'].sum().reset_index()
        df_temporal = df_temporal.sort_values('data')
        
        if len(df_temporal) >= 12:
            # Decomposição sazonal
            ts_data = df_temporal.set_index('data')['faturamento']
            decomposition = seasonal_decompose(ts_data, model='additive', period=12)
            
            fig_decomp = make_subplots(
                rows=3, cols=1,
                subplot_titles=['📊 Série Original', '📈 Tendência', '🔄 Sazonalidade'],
                vertical_spacing=0.1
            )
            
            fig_decomp.add_trace(
                go.Scatter(x=decomposition.observed.index, y=decomposition.observed.values, 
                          name="Original", line=dict(color='blue')), row=1, col=1
            )
            fig_decomp.add_trace(
                go.Scatter(x=decomposition.trend.index, y=decomposition.trend.values, 
                          name="Tendência", line=dict(color='red')), row=2, col=1
            )
            fig_decomp.add_trace(
                go.Scatter(x=decomposition.seasonal.index, y=decomposition.seasonal.values, 
                          name="Sazonalidade", line=dict(color='green')), row=3, col=1
            )
            
            fig_decomp.update_layout(height=600, showlegend=False)
            st.plotly_chart(fig_decomp, use_container_width=True)
        else:
            # Gráfico simples se não há dados suficientes
            fig_simple = px.line(
                df_temporal, x='data', y='faturamento',
                title="Evolução Temporal do Faturamento"
            )
            st.plotly_chart(fig_simple, use_container_width=True)
    
    with col2:
        # Análise de crescimento e previsão
        st.subheader("🔮 Previsão e Crescimento")
        
        loja_para_previsao = st.selectbox(
            "Selecione uma loja para previsão:",
            options=df_filtered['loja'].unique(),
            help="Escolha uma loja para ver a previsão de faturamento"
        )
        
        if loja_para_previsao:
            previsoes, r2 = simple_forecast(df, loja_para_previsao)
            
            if previsoes is not None:
                st.markdown(f"""
                <div class="success-box">
                    <strong>📊 Previsão para {loja_para_previsao}</strong><br>
                    🎯 Confiabilidade do modelo: {r2:.1%}<br>
                    📈 Próximos 6 meses: R$ {previsoes.mean():,.0f}/mês<br>
                    📊 Tendência: {"↗️ Crescimento" if previsoes[-1] > previsoes[0] else "↘️ Declínio"}
                </div>
                """, unsafe_allow_html=True)
                
                # Gráfico de previsão
                loja_historico = df[df['loja'] == loja_para_previsao].copy()
                loja_historico = loja_historico.sort_values('data')
                
                fig_prev = go.Figure()
                
                # Dados históricos
                fig_prev.add_trace(go.Scatter(
                    x=loja_historico['data'],
                    y=loja_historico['faturamento'],
                    mode='lines+markers',
                    name='Histórico',
                    line=dict(color='blue')
                ))
                
                # Previsões
                futuras_datas = pd.date_range(
                    start=loja_historico['data'].max() + pd.DateOffset(months=1),
                    periods=6,
                    freq='M'
                )
                
                fig_prev.add_trace(go.Scatter(
                    x=futuras_datas,
                    y=previsoes,
                    mode='lines+markers',
                    name='Previsão',
                    line=dict(color='red', dash='dash')
                ))
                
                fig_prev.update_layout(
                    title=f"📈 Previsão para {loja_para_previsao}",
                    height=300
                )
                st.plotly_chart(fig_prev, use_container_width=True)
    
    # Análise de performance comparativa
    st.header("🏆 Análise de Performance Comparativa")
    
    # Heatmap de performance
    col1, col2 = st.columns([3, 1])
    
    with col1:
        st.subheader("🌡️ Mapa de Calor - Performance por Loja/Mês")
        
        pivot_data = df_filtered.pivot_table(
            values='performance_score',
            index='loja',
            columns='mes',
            aggfunc='mean'
        ).fillna(0)
        
        fig_heatmap = px.imshow(
            pivot_data.values,
            labels=dict(x="Mês", y="Loja", color="Score"),
            x=[f"Mês {i}" for i in pivot_data.columns],
            y=pivot_data.index,
            aspect="auto",
            color_continuous_scale="RdYlGn"
        )
        fig_heatmap.update_layout(height=400)
        st.plotly_chart(fig_heatmap, use_container_width=True)
    
    with col2:
        st.subheader("🎖️ Rankings")
        
        # Top performers
        performance_ranking = df_filtered.groupby('loja')['performance_score'].mean().sort_values(ascending=False)
        
        st.markdown("**🏅 Top 5 Performance:**")
        for i, (loja, score) in enumerate(performance_ranking.head().items(), 1):
            emoji = "🥇" if i == 1 else "🥈" if i == 2 else "🥉" if i == 3 else "⭐"
            st.markdown(f"{emoji} **{loja}**: {score:.1f}pts")
        
        st.markdown("**⚠️ Atenção Necessária:**")
        for loja, score in performance_ranking.tail(3).items():
            st.markdown(f"🔸 **{loja}**: {score:.1f}pts")
    
    # Insights e alertas automáticos
    st.header("🧠 Insights Automáticos")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("💡 Descobertas")
        
        # Análise de concentração
        concentracao = df_filtered.groupby('loja')['faturamento'].sum()
        pareto_80 = concentracao.sort_values(ascending=False).cumsum() / concentracao.sum()
        lojas_80_pct = len(pareto_80[pareto_80 <= 0.8])
        
        st.markdown(f"""
        <div class="insight-box">
            📊 <strong>Princípio de Pareto</strong><br>
            {lojas_80_pct} lojas ({(lojas_80_pct/len(concentracao)*100):.1f}%) 
            representam 80% do faturamento total
        </div>
        """, unsafe_allow_html=True)
        
        # Análise de volatilidade
        volatilidade_media = df_filtered.groupby('loja')['crescimento_mensal'].std().mean()
        st.markdown(f"""
        <div class="insight-box">
            📈 <strong>Volatilidade Média</strong><br>
            {volatilidade_media:.1f}% de variação mensal típica
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.subheader("⚠️ Alertas")
        
        # Detectar anomalias
        anomalias = detect_anomalies(df_filtered)
        
        if not anomalias.empty:
            anomalias_recentes = anomalias[anomalias['tipo'] == 'Baixo'].head(3)
            for _, anomalia in anomalias_recentes.iterrows():
                st.markdown(f"""
                <div class="alert-box">
                    🚨 <strong>{anomalia['loja']}</strong><br>
                    Performance baixa em {anomalia['periodo']}<br>
                    Faturamento: R$ {anomalia['faturamento']:,.0f}
                </div>
                """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="success-box">
                ✅ <strong>Sem Anomalias</strong><br>
                Todas as lojas dentro do padrão esperado
            </div>
            """, unsafe_allow_html=True)
    
    with col3:
        st.subheader("🎯 Recomendações")
        
        # Gerar recomendações baseadas em dados
        top_performer = df_filtered.groupby('loja')['performance_score'].mean().idxmax()
        ticket_alto = df_filtered.groupby('loja')['ticket'].mean().idxmax()
        crescimento_forte = df_filtered.groupby('loja')['crescimento_mensal'].mean().idxmax()
        
        st.markdown(f"""
        <div class="insight-box">
            🌟 <strong>Benchmarking</strong><br>
            • Estudar modelo de <strong>{top_performer}</strong><br>
            • Analisar estratégia de <strong>{ticket_alto}</strong><br>
            • Replicar crescimento de <strong>{crescimento_forte}</strong>
        </div>
        """, unsafe_allow_html=True)
    
    # Análise de correlações e insights estatísticos
    st.header("📊 Análise Estatística Avançada")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🔗 Matriz de Correlações")
        
        corr_cols = ['faturamento', 'pedidos', 'ticket', 'performance_score']
        correlation_matrix = df_filtered[corr_cols].corr()
        
        fig_corr = px.imshow(
            correlation_matrix,
            text_auto=True,
            aspect="auto",
            color_continuous_scale="RdBu",
            title="Correlações entre Métricas Principais"
        )
        st.plotly_chart(fig_corr, use_container_width=True)
        
        # Insights de correlação
        correlacao_fat_pedidos = correlation_matrix.loc['faturamento', 'pedidos']
        correlacao_fat_ticket = correlation_matrix.loc['faturamento', 'ticket']
        
        st.markdown(f"""
        **🔍 Interpretação:**
        - Faturamento vs Pedidos: {correlacao_fat_pedidos:.2f} {'(forte)' if abs(correlacao_fat_pedidos) > 0.7 else '(moderada)' if abs(correlacao_fat_pedidos) > 0.3 else '(fraca)'}
        - Faturamento vs Ticket: {correlacao_fat_ticket:.2f} {'(forte)' if abs(correlacao_fat_ticket) > 0.7 else '(moderada)' if abs(correlacao_fat_ticket) > 0.3 else '(fraca)'}
        """)
    
    with col2:
        st.subheader("📈 Distribuição de Métricas")
        
        metrica_analise = st.selectbox(
            "Escolha uma métrica para análise:",
            options=['faturamento', 'pedidos', 'ticket', 'performance_score'],
            format_func=lambda x: {
                'faturamento': '💰 Faturamento',
                'pedidos': '🛒 Pedidos', 
                'ticket': '🎯 Ticket Médio',
                'performance_score': '⭐ Score de Performance'
            }[x]
        )
        
        fig_dist = px.histogram(
            df_filtered,
            x=metrica_analise,
            nbins=30,
            title=f"Distribuição: {metrica_analise.title()}",
            marginal="box"
        )
        fig_dist.update_layout(height=400)
        st.plotly_chart(fig_dist, use_container_width=True)
        
        # Estatísticas descritivas
        stats_desc = df_filtered[metrica_analise].describe()
        st.markdown(f"""
        **📊 Estatísticas:**
        - Média: {stats_desc['mean']:,.2f}
        - Mediana: {stats_desc['50%']:,.2f}
        - Desvio padrão: {stats_desc['std']:,.2f}
        - Assimetria: {df_filtered[metrica_analise].skew():.2f}
        """)
    
    # Análise de sazonalidade
    st.header("🗓️ Inteligência Sazonal")
    
    sazonalidade, melhor_mes, pior_mes = analyze_seasonality(df_filtered)
    
    col1, col2 = st.columns([3, 1])
    
    with col1:
        # Gráfico radar de sazonalidade
        fig_radar = go.Figure()
        
        fig_radar.add_trace(go.Scatterpolar(
            r=sazonalidade['fat_total'],
            theta=[f"Mês {i}" for i in sazonalidade.index],
            fill='toself',
            name='Faturamento Mensal',
            line_color='rgb(67, 125, 191)'
        ))
        
        fig_radar.update_layout(
            polar=dict(
                radialaxis=dict(visible=True, range=[0, sazonalidade['fat_total'].max()])
            ),
            title="🎯 Radar de Performance Sazonal",
            height=400
        )
        st.plotly_chart(fig_radar, use_container_width=True)
    
    with col2:
        st.subheader("📅 Padrões Sazonais")
        
        variacao_sazonal = (sazonalidade['fat_total'].max() - sazonalidade['fat_total'].min()) / sazonalidade['fat_total'].mean() * 100
        
        st.markdown(f"""
        <div class="insight-box">
            🏆 <strong>Melhor mês:</strong> {melhor_mes}<br>
            📉 <strong>Pior mês:</strong> {pior_mes}<br>
            📊 <strong>Variação sazonal:</strong> {variacao_sazonal:.1f}%
        </div>
        """, unsafe_allow_html=True)
        
        # Recomendações sazonais
        if variacao_sazonal > 20:
            st.markdown("""
            <div class="alert-box">
                ⚠️ <strong>Alta variação sazonal detectada!</strong><br>
                Considere estratégias para suavizar a sazonalidade
            </div>
            """, unsafe_allow_html=True)
    
    # Dashboard de comparação
    st.header("⚖️ Análise Comparativa Inteligente")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Eficiência Operacional")
        
        # Calcular eficiência por loja
        eficiencia_loja = df_filtered.groupby('loja').agg({
            'faturamento': 'sum',
            'pedidos': 'sum',
            'ticket': 'mean'
        })
        eficiencia_loja['faturamento_por_pedido'] = eficiencia_loja['faturamento'] / eficiencia_loja['pedidos']
        eficiencia_loja['eficiencia_score'] = (
            eficiencia_loja['faturamento_por_pedido'] / eficiencia_loja['faturamento_por_pedido'].max() * 100
        )
        
        fig_eficiencia = px.scatter(
            eficiencia_loja.reset_index(),
            x='pedidos',
            y='faturamento',
            size='ticket',
            color='eficiencia_score',
            hover_name='loja',
            title="💡 Matriz de Eficiência: Volume vs Receita",
            color_continuous_scale="Viridis"
        )
        fig_eficiencia.update_layout(height=400)
        st.plotly_chart(fig_eficiencia, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Oportunidades de Crescimento")
        
        # Identificar oportunidades
        benchmark_ticket = df_filtered['ticket'].quantile(0.75)
        benchmark_pedidos = df_filtered['pedidos'].quantile(0.75)
        
        oportunidades = df_filtered.groupby('loja').agg({
            'ticket': 'mean',
            'pedidos': 'mean',
            'faturamento': 'sum'
        })
        
        oportunidades['potencial_ticket'] = np.where(
            oportunidades['ticket'] < benchmark_ticket, 
            (benchmark_ticket - oportunidades['ticket']) * oportunidades['pedidos'],
            0
        )
        
        oportunidades_top = oportunidades.nlargest(5, 'potencial_ticket')
        
        fig_oportunidade = px.bar(
            oportunidades_top.reset_index(),
            x='loja',
            y='potencial_ticket',
            title="💎 Top 5 Oportunidades de Aumento de Ticket",
            color='potencial_ticket',
            color_continuous_scale="Blues"
        )
        fig_oportunidade.update_xaxes(tickangle=45)
        fig_oportunidade.update_layout(height=400)
        st.plotly_chart(fig_oportunidade, use_container_width=True)
    
    # Tabela inteligente com insights
    st.header("📋 Relatório Executivo Inteligente")
    
    # Preparar dados para tabela executiva
    executive_summary = df_filtered.groupby('loja').agg({
        'faturamento': ['sum', 'mean', 'std'],
        'pedidos': ['sum', 'mean'],
        'ticket': ['mean', 'std'],
        'crescimento_mensal': 'mean',
        'performance_score': 'mean'
    }).round(2)
    
    executive_summary.columns = [
        'Fat_Total', 'Fat_Medio', 'Fat_DP', 'Ped_Total', 'Ped_Medio',
        'Ticket_Medio', 'Ticket_DP', 'Crescimento', 'Score'
    ]
    
    # Adicionar classificações
    executive_summary['Classe_Faturamento'] = pd.cut(
        executive_summary['Fat_Total'], 
        bins=3, 
        labels=['🔴 Baixo', '🟡 Médio', '🟢 Alto']
    )
    
    executive_summary['Estabilidade'] = np.where(
        executive_summary['Fat_DP'] / executive_summary['Fat_Medio'] < 0.3,
        '🟢 Estável', '🟡 Volátil'
    )
    
    # Reformatar para exibição
    executive_display = executive_summary.reset_index()
    executive_display['Fat_Total'] = executive_display['Fat_Total'].apply(lambda x: f"R$ {x:,.0f}")
    executive_display['Ticket_Medio'] = executive_display['Ticket_Medio'].apply(lambda x: f"R$ {x:.2f}")
    executive_display['Crescimento'] = executive_display['Crescimento'].apply(lambda x: f"{x:+.1f}%")
    executive_display['Score'] = executive_display['Score'].apply(lambda x: f"{x:.1f}")
    
    # Selecionar colunas para exibição
    display_cols = ['loja', 'Fat_Total', 'Ped_Total', 'Ticket_Medio', 'Crescimento', 'Score', 'Classe_Faturamento', 'Estabilidade']
    final_display = executive_display[display_cols]
    final_display.columns = ['🏪 Loja', '💰 Faturamento', '🛒 Pedidos', '🎯 Ticket', '📈 Crescimento', '⭐ Score', '📊 Classe', '📈 Estabilidade']
    
    st.dataframe(
        final_display.sort_values('⭐ Score', ascending=False),
        use_container_width=True,
        hide_index=True
    )
    
    # Insights finais e recomendações estratégicas
    st.header("🎯 Recomendações Estratégicas")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Ações de Crescimento")
        
        # Identificar lojas com potencial
        lojas_potencial = df_filtered.groupby('loja').agg({
            'crescimento_mensal': 'mean',
            'ticket': 'mean',
            'faturamento': 'sum'
        })
        
        # Lojas com crescimento positivo mas faturamento baixo
        potencial_crescimento = lojas_potencial[
            (lojas_potencial['crescimento_mensal'] > 0) & 
            (lojas_potencial['faturamento'] < lojas_potencial['faturamento'].median())
        ].sort_values('crescimento_mensal', ascending=False)
        
        if not potencial_crescimento.empty:
            st.markdown("**🚀 Lojas com Alto Potencial:**")
            for loja in potencial_crescimento.head(3).index:
                crescimento = potencial_crescimento.loc[loja, 'crescimento_mensal']
                st.markdown(f"• **{loja}**: {crescimento:+.1f}% crescimento mensal")
        
        # Análise de ticket médio
        ticket_benchmark = df_filtered['ticket'].quantile(0.75)
        lojas_baixo_ticket = df_filtered.groupby('loja')['ticket'].mean()
        lojas_melhorar_ticket = lojas_baixo_ticket[lojas_baixo_ticket < ticket_benchmark].sort_values()
        
        if not lojas_melhorar_ticket.empty:
            st.markdown("**💡 Foco em Ticket Médio:**")
            for loja in lojas_melhorar_ticket.head(3).index:
                ticket_atual = lojas_melhorar_ticket[loja]
                potencial = ticket_benchmark - ticket_atual
                st.markdown(f"• **{loja}**: +R$ {potencial:.2f} de potencial")
    
    with col2:
        st.subheader("⚠️ Ações Corretivas")
        
        # Lojas com declínio
        lojas_declinio = df_filtered.groupby('loja')['crescimento_mensal'].mean()
        lojas_atencao = lojas_declinio[lojas_declinio < -5].sort_values()
        
        if not lojas_atencao.empty:
            st.markdown("**🔴 Lojas Requerendo Atenção:**")
            for loja in lojas_atencao.head(3).index:
                declinio = lojas_atencao[loja]
                st.markdown(f"• **{loja}**: {declinio:.1f}% declínio mensal")
        
        # Análise de eficiência
        eficiencia_baixa = df_filtered.groupby('loja')['eficiencia_vendas'].mean()
        ineficientes = eficiencia_baixa[eficiencia_baixa > eficiencia_baixa.quantile(0.8)].sort_values(ascending=False)
        
        if not ineficientes.empty:
            st.markdown("**🎯 Melhorar Eficiência:**")
            for loja in ineficientes.head(3).index:
                efic = ineficientes[loja]
                st.markdown(f"• **{loja}**: {efic:.1f} pedidos/R$1k (otimizar)")
    
    # Footer com estatísticas gerais
    st.markdown("---")
    st.markdown("### 📊 Resumo Executivo da Análise")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        cagr = (df_filtered.groupby('ano')['faturamento'].sum().iloc[-1] / df_filtered.groupby('ano')['faturamento'].sum().iloc[0]) ** (1/len(df_filtered['ano'].unique())) - 1
        st.metric("📈 CAGR", f"{cagr:.1%}", help="Taxa de Crescimento Anual Composta")
    
    with col2:
        volatilidade = df_filtered['faturamento'].std() / df_filtered['faturamento'].mean()
        st.metric("📊 Volatilidade", f"{volatilidade:.1%}", help="Coeficiente de Variação do Faturamento")
    
    with col3:
        gini_faturamento = calculate_gini(df_filtered.groupby('loja')['faturamento'].sum().values)
        st.metric("⚖️ Concentração", f"{gini_faturamento:.3f}", help="Índice de Gini (0=igualitário, 1=concentrado)")
    
    with col4:
        pred_accuracy = "Alta" if len(df_filtered) > 24 else "Média" if len(df_filtered) > 12 else "Baixa"
        st.metric("🎯 Confiabilidade", pred_accuracy, help="Qualidade das previsões baseada no volume de dados")

if __name__ == "__main__":
    main()
