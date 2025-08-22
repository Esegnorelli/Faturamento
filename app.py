"""
Aplicativo Streamlit para analisar dados de faturamento, pedidos e ticket médio.

O aplicativo carrega os dados de ``Faturamento.csv`` presentes no diretório atual,
converte colunas monetárias para valores numéricos e exibe um painel interativo
com filtros de ano, mês e loja. As métricas de faturamento total, número de
pedidos e ticket médio são calculadas dinamicamente. Gráficos de barras
mostram a evolução do faturamento e pedidos ao longo do tempo e por loja.
Além disso, são destacados insights úteis, como a loja com maior faturamento,
a loja com mais pedidos e a loja com maior ticket médio no período filtrado.

Para executar o aplicativo localmente, instale as dependências (pandas e
streamlit) e execute:

```
streamlit run app.py
```

O arquivo ``Logo.jpg`` incluído no diretório será exibido no cabeçalho da
aplicação.
"""

import locale
from pathlib import Path

import pandas as pd
import streamlit as st


def carregar_dados(caminho_csv: str) -> pd.DataFrame:
    """Carrega o CSV de faturamento e converte valores monetários.

    Args:
        caminho_csv: Caminho para o arquivo CSV.

    Returns:
        DataFrame com colunas adicionais ``Faturamento_Num`` e
        ``Ticket_Médio_Num`` convertidas para ``float``.
    """
    df = pd.read_csv(caminho_csv)

    def texto_para_float(valor: str) -> float:
        """Converte valores do tipo 'R$ 1.234,56' para float.

        Ignora espaços e símbolos monetários, removendo pontos de
        separação de milhares e trocando vírgula por ponto para o separador
        decimal.
        """
        if pd.isna(valor):
            return 0.0
        # Garantir que é string
        texto = str(valor).strip()
        texto = texto.replace("R$", "").replace(".", "").replace(",", ".")
        try:
            return float(texto)
        except ValueError:
            return 0.0

    # Aplicar conversão
    df["Faturamento_Num"] = df["Faturamento"].apply(texto_para_float)
    df["Ticket_Médio_Num"] = df["Ticket_Médio"].apply(texto_para_float)
    return df


@st.cache_data(show_spinner=False)
def obter_dados() -> pd.DataFrame:
    """Envolve ``carregar_dados`` em cache para evitar recarregamento.

    Utiliza o arquivo 'Faturamento.csv' localizado no mesmo diretório
    deste script.
    """
    caminho = Path(__file__).with_name("Faturamento.csv")
    return carregar_dados(str(caminho))


def formatar_moeda(valor: float) -> str:
    """Formata um número como valor monetário em reais.

    Usa a localidade brasileira para formatação (caso disponível),
    caso contrário implementa manualmente.
    """
    try:
        locale.setlocale(locale.LC_ALL, "pt_BR.UTF-8")
        return locale.currency(valor, grouping=True)
    except locale.Error:
        # Fallback manual: usa separador de milhar como ponto e decimal como vírgula
        inteiro, decimal = f"{valor:,.2f}".split(".")
        inteiro = inteiro.replace(",", ".")  # inverter separadores temporários
        return f"R$ {inteiro},{decimal}"


def criar_interface():
    """Constrói a interface Streamlit com filtros, métricas e gráficos."""
    st.set_page_config(page_title="Dashboard de Faturamento", layout="wide")

    # Exibir logo se existir
    logo_path = Path(__file__).with_name("Logo.jpg")
    if logo_path.exists():
        st.image(str(logo_path), width=200)

    st.title("Dashboard de Faturamento")

    df = obter_dados()

    # Filtros interativos
    anos = sorted(df["Ano"].unique())
    meses = sorted(df["Mes"].unique())
    lojas = sorted(df["Loja"].unique())

    st.sidebar.header("Filtros")
    anos_selecionados = st.sidebar.multiselect(
        "Ano", options=anos, default=anos, format_func=lambda x: str(x)
    )
    meses_selecionados = st.sidebar.multiselect(
        "Mês", options=meses, default=meses,
        format_func=lambda m: f"{m:02d}"
    )
    lojas_selecionadas = st.sidebar.multiselect(
        "Loja", options=lojas, default=lojas
    )

    # Filtrando dados conforme seleção
    df_filtrado = df[
        df["Ano"].isin(anos_selecionados) &
        df["Mes"].isin(meses_selecionados) &
        df["Loja"].isin(lojas_selecionadas)
    ]

    if df_filtrado.empty:
        st.warning("Nenhum dado encontrado para os filtros selecionados.")
        return

    # Métricas agregadas
    faturamento_total = df_filtrado["Faturamento_Num"].sum()
    pedidos_total = int(df_filtrado["Pedidos"].sum())
    ticket_medio = faturamento_total / pedidos_total if pedidos_total else 0

    # Exibição de métricas
    col1, col2, col3 = st.columns(3)
    col1.metric(
        "Faturamento Total", formatar_moeda(faturamento_total),
        help="Soma de todo o faturamento para o período filtrado."
    )
    col2.metric(
        "Total de Pedidos", f"{pedidos_total:,}",
        help="Número total de pedidos para o período filtrado."
    )
    col3.metric(
        "Ticket Médio", formatar_moeda(ticket_medio),
        help="Média de faturamento por pedido."
    )

    # Agrupar por ano/mês para gráficos temporais
    df_temp = (
        df_filtrado
        .groupby(["Ano", "Mes"])
        .agg({"Faturamento_Num": "sum", "Pedidos": "sum"})
        .reset_index()
    )
    df_temp["AnoMes"] = df_temp["Ano"].astype(str) + "-" + df_temp["Mes"].astype(int).astype(str).str.zfill(2)

    st.subheader("Faturamento e Pedidos por Período")
    # Layout com duas colunas para os gráficos temporais
    col4, col5 = st.columns(2)
    with col4:
        st.write("### Faturamento por Mês")
        st.bar_chart(
            data=df_temp.set_index("AnoMes")["Faturamento_Num"],
            use_container_width=True
        )
    with col5:
        st.write("### Pedidos por Mês")
        st.bar_chart(
            data=df_temp.set_index("AnoMes")["Pedidos"],
            use_container_width=True
        )

    # Agrupar por loja
    df_loja = (
        df_filtrado
        .groupby("Loja")
        .agg({"Faturamento_Num": "sum", "Pedidos": "sum"})
        .reset_index()
    )
    df_loja["Ticket_Medio"] = df_loja.apply(
        lambda x: x["Faturamento_Num"] / x["Pedidos"] if x["Pedidos"] else 0,
        axis=1
    )

    st.subheader("Faturamento por Loja")
    # Ordenar por faturamento antes de plotar
    df_loja_sorted = df_loja.sort_values("Faturamento_Num", ascending=False)
    st.bar_chart(
        data=df_loja_sorted.set_index("Loja")["Faturamento_Num"],
        use_container_width=True
    )

    st.subheader("Insights")
    # Loja com maior faturamento
    top_faturamento = df_loja_sorted.iloc[0]
    st.markdown(
        f"**Loja com maior faturamento:** {top_faturamento['Loja']} com "+
        f"{formatar_moeda(top_faturamento['Faturamento_Num'])}"
    )
    # Loja com mais pedidos
    top_pedidos = df_loja.sort_values("Pedidos", ascending=False).iloc[0]
    st.markdown(
        f"**Loja com maior número de pedidos:** {top_pedidos['Loja']} com "
        f"{int(top_pedidos['Pedidos']):,} pedidos"
    )
    # Loja com maior ticket médio
    top_ticket = df_loja.sort_values("Ticket_Medio", ascending=False).iloc[0]
    st.markdown(
        f"**Loja com maior ticket médio:** {top_ticket['Loja']} com "
        f"{formatar_moeda(top_ticket['Ticket_Medio'])}"
    )

    # Exibir tabela de dados filtrados caso o usuário deseje
    with st.expander("Ver dados filtrados"):
        st.dataframe(
            df_filtrado[[
                "Ano", "Mes", "Loja", "Faturamento", "Pedidos", "Ticket_Médio"
            ]].reset_index(drop=True),
            use_container_width=True
        )


if __name__ == "__main__":
    criar_interface()