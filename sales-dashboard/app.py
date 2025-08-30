import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os

# Configuração da página
st.set_page_config(
    page_title="Dashboard de Faturamento",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Função para carregar dados
@st.cache_data
def load_data():
    """Carrega e processa os dados do CSV"""
    try:
        # Ler o CSV com separador padrão (vírgula)
        df = pd.read_csv('public/Faturamento.csv')
        
        # Converter colunas numéricas
        df['faturamento'] = pd.to_numeric(df['faturamento'], errors='coerce')
        df['pedidos'] = pd.to_numeric(df['pedidos'], errors='coerce')
        df['ticket'] = pd.to_numeric(df['ticket'], errors='coerce')
        
        # Criar coluna de data
        df['data'] = pd.to_datetime(df['periodo'], format='%Y-%m', errors='coerce')
        
        # Remover linhas com dados inválidos
        df = df.dropna(subset=['faturamento', 'pedidos', 'ticket'])
        
        return df
    except Exception as e:
        st.error(f"Erro ao carregar dados: {e}")
        return pd.DataFrame()

# Função para exibir logo
def show_logo():
    """Exibe o logo se existir"""
    if os.path.exists('logo.png'):
        st.image('logo.png', width=200)

# Função principal
def main():
    # Header
    col1, col2 = st.columns([1, 4])
    
    with col1:
        show_logo()
    
    with col2:
        st.title("📊 Dashboard de Faturamento")
        st.markdown("### Análise de vendas por loja e período")
    
    # Carregar dados
    df = load_data()
    
    if df.empty:
        st.error("Não foi possível carregar os dados do arquivo CSV.")
        return
    
    # Sidebar - Filtros
    st.sidebar.header("🔍 Filtros")
    
    # Filtro de período
    anos_disponiveis = sorted(df['ano'].unique())
    ano_selecionado = st.sidebar.selectbox(
        "Selecione o Ano:",
        options=['Todos'] + anos_disponiveis,
        index=0
    )
    
    # Filtro de lojas
    lojas_disponiveis = sorted(df['loja'].unique())
    lojas_selecionadas = st.sidebar.multiselect(
        "Selecione as Lojas:",
        options=lojas_disponiveis,
        default=lojas_disponiveis[:5]  # Primeiras 5 lojas por padrão
    )
    
    # Aplicar filtros
    df_filtered = df.copy()
    
    if ano_selecionado != 'Todos':
        df_filtered = df_filtered[df_filtered['ano'] == ano_selecionado]
    
    if lojas_selecionadas:
        df_filtered = df_filtered[df_filtered['loja'].isin(lojas_selecionadas)]
    
    # Métricas principais
    st.header("📈 Métricas Principais")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_faturamento = df_filtered['faturamento'].sum()
        st.metric(
            "Faturamento Total",
            f"R$ {total_faturamento:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.')
        )
    
    with col2:
        total_pedidos = df_filtered['pedidos'].sum()
        st.metric(
            "Total de Pedidos",
            f"{total_pedidos:,.0f}".replace(',', '.')
        )
    
    with col3:
        ticket_medio = df_filtered['ticket'].mean()
        st.metric(
            "Ticket Médio",
            f"R$ {ticket_medio:.2f}".replace('.', ',')
        )
    
    with col4:
        total_lojas = df_filtered['loja'].nunique()
        st.metric(
            "Lojas Ativas",
            total_lojas
        )
    
    # Gráficos
    col1, col2 = st.columns(2)
    
    # Gráfico 1: Faturamento por período
    with col1:
        st.subheader("📅 Faturamento por Período")
        df_tempo = df_filtered.groupby('periodo')['faturamento'].sum().reset_index()
        df_tempo = df_tempo.sort_values('periodo')
        
        fig_tempo = px.line(
            df_tempo, 
            x='periodo', 
            y='faturamento',
            title="Evolução do Faturamento",
            labels={'faturamento': 'Faturamento (R$)', 'periodo': 'Período'}
        )
        fig_tempo.update_layout(height=400)
        st.plotly_chart(fig_tempo, use_container_width=True)
    
    # Gráfico 2: Top lojas por faturamento
    with col2:
        st.subheader("🏪 Top Lojas por Faturamento")
        df_lojas = df_filtered.groupby('loja')['faturamento'].sum().reset_index()
        df_lojas = df_lojas.sort_values('faturamento', ascending=False).head(10)
        
        fig_lojas = px.bar(
            df_lojas,
            x='faturamento',
            y='loja',
            orientation='h',
            title="Top 10 Lojas",
            labels={'faturamento': 'Faturamento (R$)', 'loja': 'Loja'}
        )
        fig_lojas.update_layout(height=400)
        st.plotly_chart(fig_lojas, use_container_width=True)
    
    # Gráfico 3: Comparação de métricas por loja
    st.subheader("📊 Análise Detalhada por Loja")
    
    df_resumo = df_filtered.groupby('loja').agg({
        'faturamento': 'sum',
        'pedidos': 'sum',
        'ticket': 'mean'
    }).reset_index()
    
    # Gráfico de dispersão: Faturamento vs Pedidos
    fig_scatter = px.scatter(
        df_resumo,
        x='pedidos',
        y='faturamento',
        size='ticket',
        hover_name='loja',
        title="Faturamento vs Pedidos (tamanho = ticket médio)",
        labels={
            'pedidos': 'Total de Pedidos',
            'faturamento': 'Faturamento Total (R$)',
            'ticket': 'Ticket Médio (R$)'
        }
    )
    fig_scatter.update_layout(height=500)
    st.plotly_chart(fig_scatter, use_container_width=True)
    
    # Tabela de dados
    st.subheader("📋 Dados Detalhados")
    
    # Opções de visualização
    col1, col2 = st.columns([3, 1])
    
    with col2:
        mostrar_resumo = st.checkbox("Mostrar Resumo por Loja", value=True)
    
    if mostrar_resumo:
        # Tabela resumo
        df_table = df_resumo.copy()
        df_table['faturamento'] = df_table['faturamento'].apply(
            lambda x: f"R$ {x:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.')
        )
        df_table['pedidos'] = df_table['pedidos'].apply(
            lambda x: f"{x:,.0f}".replace(',', '.')
        )
        df_table['ticket'] = df_table['ticket'].apply(
            lambda x: f"R$ {x:.2f}".replace('.', ',')
        )
        df_table.columns = ['Loja', 'Faturamento Total', 'Total de Pedidos', 'Ticket Médio']
        df_table = df_table.sort_values('Faturamento Total', ascending=False)
        
        st.dataframe(df_table, use_container_width=True, hide_index=True)
    else:
        # Tabela completa
        df_display = df_filtered.copy()
        df_display['faturamento'] = df_display['faturamento'].apply(
            lambda x: f"R$ {x:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.')
        )
        df_display['pedidos'] = df_display['pedidos'].apply(
            lambda x: f"{x:,.0f}".replace(',', '.')
        )
        df_display['ticket'] = df_display['ticket'].apply(
            lambda x: f"R$ {x:.2f}".replace('.', ',')
        )
        df_display = df_display[['periodo', 'loja', 'faturamento', 'pedidos', 'ticket']]
        df_display.columns = ['Período', 'Loja', 'Faturamento', 'Pedidos', 'Ticket Médio']
        
        st.dataframe(df_display, use_container_width=True, hide_index=True)
    
    # Análises adicionais
    st.subheader("🔍 Análises Adicionais")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Crescimento por ano
        if len(df_filtered['ano'].unique()) > 1:
            st.write("**Crescimento por Ano:**")
            df_anos = df_filtered.groupby('ano')['faturamento'].sum().reset_index()
            for i in range(1, len(df_anos)):
                ano_atual = df_anos.iloc[i]['ano']
                ano_anterior = df_anos.iloc[i-1]['ano']
                crescimento = ((df_anos.iloc[i]['faturamento'] - df_anos.iloc[i-1]['faturamento']) / df_anos.iloc[i-1]['faturamento']) * 100
                st.write(f"- {ano_anterior} → {ano_atual}: {crescimento:+.1f}%")
    
    with col2:
        # Top 3 lojas
        st.write("**Top 3 Lojas por Faturamento:**")
        top_lojas = df_filtered.groupby('loja')['faturamento'].sum().nlargest(3)
        for i, (loja, faturamento) in enumerate(top_lojas.items(), 1):
            st.write(f"{i}. {loja}: R$ {faturamento:,.2f}".replace(',', 'X').replace('.', ',').replace('X', '.'))

if __name__ == "__main__":
    main()
