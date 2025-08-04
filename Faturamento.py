import streamlit as st
import pandas as pd
import re


import os

def load_raw_data(file_path: str) -> str:
    """Lê e retorna o conteúdo do arquivo de dados.
    Linhas iniciadas com '#' são ignoradas como comentários.
    """
    if not os.path.isfile(file_path):
        return ""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = []
        for line in f:
            if line.strip().startswith('#') or not line.strip():
                continue
            lines.append(line.rstrip('\n'))
        return '\n'.join(lines)


def parse_data(text: str) -> pd.DataFrame:
    """Converte o texto bruto em um DataFrame.

    Cada linha do texto contém as colunas: Data, Status, Loja,
    Faturamento (R$), Pedidos e Ticket Médio. O texto utiliza
    separadores de espaço e vírgula para os valores monetários.
    Esta função usa expressões regulares para extrair as colunas e
    converte valores monetários para float e pedidos para int.
    """
    pattern = re.compile(
        r'^(?P<Data>\S+/\d{4})\s+'
        r'(?P<Status>\w+)\s+'
        r'(?P<Loja>.*?)\s+'
        r'R\$\s*(?P<Faturamento>[\d\.\,]+)\s+'
        r'(?P<Pedidos>[\d\.]+)\s+'
        r'R\$\s*(?P<TicketMedio>[\d\.\,]+)$'
    )
    rows = []
    for raw_line in text.strip().splitlines():
        # Substitui tabulações por espaços para unificar a separação entre colunas
        line = raw_line.replace('\t', ' ').strip()
        if not line:
            continue
        match = pattern.match(line)
        if not match:
            # Ignora linhas que não correspondem ao padrão (por exemplo cabeçalhos)
            continue
        row = match.groupdict()
        # Converte campos numéricos removendo separadores de milhar e
        # substituindo vírgula por ponto para valores monetários.
        faturamento = row['Faturamento'].replace('.', '').replace(',', '.')
        pedidos = row['Pedidos'].replace('.', '')
        ticket = row['TicketMedio'].replace('.', '').replace(',', '.')
        rows.append({
            'Data': row['Data'],
            'Status': row['Status'],
            'Loja': row['Loja'],
            'Faturamento': float(faturamento),
            'Pedidos': int(pedidos),
            'Ticket Médio': float(ticket),
        })
    df = pd.DataFrame(rows)
    # Converte a coluna de Data para um tipo de data utilizando os nomes de mês em português.
    month_map = {
        'jan': 'Jan', 'fev': 'Feb', 'mar': 'Mar', 'abr': 'Apr',
        'mai': 'May', 'jun': 'Jun', 'jul': 'Jul', 'ago': 'Aug',
        'set': 'Sep', 'out': 'Oct', 'nov': 'Nov', 'dez': 'Dec'
    }
    def parse_date(pt_date: str) -> pd.Timestamp:
        mes, ano = pt_date.split('/')
        mes = mes.lower()
        mes_en = month_map.get(mes, mes)
        return pd.to_datetime(f"{mes_en}/{ano}")
    df['Data'] = df['Data'].apply(parse_date)
    # Cria coluna de categoria para separar lojas ativas e inativas
    df['Categoria'] = df['Status'].map({'On': 'lojas_ativas', 'Off': 'lojas_inativas'})
    # Cria coluna com mês/ano (mm/aaaa) para facilitar filtragem pelo usuário
    df['MesAno'] = df['Data'].dt.strftime('%m/%Y')
    return df


def main():
    st.set_page_config(page_title="Dashboard de Vendas", layout="wide")
    st.title("Dashboard de Vendas por Loja")
    # Processa o texto para gerar o DataFrame
    raw_text = load_raw_data('data.txt')
    df = parse_data(raw_text)

    # Barra lateral com filtros
    st.sidebar.header("Filtros")
    # Opções de lojas
    lojas = df['Loja'].unique().tolist()
    lojas_selecionadas = st.sidebar.multiselect(
        "Selecione as lojas:", lojas, default=lojas
    )
    # Opções de categorias (ativas/inativas)
    categorias = df['Categoria'].unique().tolist()
    categorias_selecionadas = st.sidebar.multiselect(
        "Selecione o tipo de loja:", categorias, default=categorias
    )
    # Opções de meses/anos para o filtro de datas.  Utilizamos um slider
    # com duas alças que retorna uma tupla (inicio, fim).
    datas_unicas = [d.strftime('%m/%Y') for d in sorted(df['Data'].unique())]
    data_range = st.sidebar.select_slider(
        "Selecione o intervalo de datas (mm/aaaa):",
        options=datas_unicas,
        value=(datas_unicas[0], datas_unicas[-1])
    )
    data_inicio_str, data_fim_str = data_range
    # Converte as strings selecionadas para datas reais (primeiro dia do mês)
    def str_to_date(mes_ano: str):
        return pd.to_datetime(f"01/{mes_ano}", dayfirst=True)
    data_inicio_dt = str_to_date(data_inicio_str)
    data_fim_dt = str_to_date(data_fim_str)
    # Garante que a data final seja maior ou igual à data inicial
    if data_fim_dt < data_inicio_dt:
        data_inicio_dt, data_fim_dt = data_fim_dt, data_inicio_dt
    # Aplica filtros
    df_filtrado = df[
        (df['Data'] >= data_inicio_dt) &
        (df['Data'] <= data_fim_dt) &
        (df['Loja'].isin(lojas_selecionadas)) &
        (df['Categoria'].isin(categorias_selecionadas))
    ]

    # Métricas resumidas
    total_faturamento = df_filtrado['Faturamento'].sum()
    total_pedidos = df_filtrado['Pedidos'].sum()
    ticket_medio = (total_faturamento / total_pedidos) if total_pedidos else 0

    # Funções de formatação para valores no padrão brasileiro
    def format_currency_br(value: float) -> str:
        formatted = f"{value:,.2f}"
        # Troca separadores: ponto vira vírgula e vírgula vira ponto
        formatted = formatted.replace(',', 'X').replace('.', ',').replace('X', '.')
        return f"R$ {formatted}"

    def format_int_br(value: int) -> str:
        formatted = f"{value:,}"
        return formatted.replace(',', '.')

    col1, col2, col3 = st.columns(3)
    col1.metric("Faturamento Total", format_currency_br(total_faturamento))
    col2.metric("Total de Pedidos", format_int_br(total_pedidos))
    col3.metric("Ticket Médio", format_currency_br(ticket_medio))

    # Mostra a tabela de dados filtrados
    st.subheader("Dados filtrados")
    st.dataframe(df_filtrado.sort_values(by='Data'))

    # Gráfico de faturamento ao longo do tempo
    import altair as alt
    faturamento_mensal = df_filtrado.groupby('Data')['Faturamento'].sum().reset_index()
    chart1 = alt.Chart(faturamento_mensal).mark_line(point=True).encode(
        x=alt.X('Data:T', title='Data'),
        y=alt.Y('Faturamento:Q', title='Faturamento (R$)')
    ).properties(
        title='Faturamento ao longo do tempo'
    )
    st.altair_chart(chart1, use_container_width=True)

    # Gráfico de faturamento por loja
    faturamento_loja = df_filtrado.groupby('Loja')['Faturamento'].sum().reset_index()
    chart2 = alt.Chart(faturamento_loja).mark_bar().encode(
        x=alt.X('Faturamento:Q', title='Faturamento (R$)'),
        y=alt.Y('Loja:N', sort='-x', title='Loja')
    ).properties(
        title='Faturamento por loja'
    )
    st.altair_chart(chart2, use_container_width=True)

    # Gráfico do ticket médio ao longo do tempo: soma do faturamento dividido pelo número de pedidos em cada mês
    if not df_filtrado.empty:
        ticket_medio_mensal = (
            df_filtrado.groupby('Data').apply(
                lambda x: x['Faturamento'].sum() / x['Pedidos'].sum() if x['Pedidos'].sum() > 0 else 0
            ).reset_index(name='TicketMedio')
        )
        chart3 = alt.Chart(ticket_medio_mensal).mark_line(point=True).encode(
            x=alt.X('Data:T', title='Data'),
            y=alt.Y('TicketMedio:Q', title='Ticket Médio (R$)')
        ).properties(
            title='Ticket médio ao longo do tempo'
        )
        st.altair_chart(chart3, use_container_width=True)


if __name__ == "__main__":
    main()