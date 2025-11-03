from __future__ import annotations
from pathlib import Path
from typing import List, Optional, Tuple, Dict
import numpy as np
import pandas as pd
import plotly.express as px
import streamlit as st

# =====================================================================
# CONSTANTES
# =====================================================================
MESES = {
    1: "Janeiro", 2: "Fevereiro", 3: "Março", 4: "Abril",
    5: "Maio", 6: "Junho", 7: "Julho", 8: "Agosto",
    9: "Setembro", 10: "Outubro", 11: "Novembro", 12: "Dezembro"
}

COLUNAS_REQUERIDAS = {"Data", "Loja"}
QUEDA_ALERTA_THRESHOLD = -30  # Percentual para alertas de queda
TOP_N = 10  # Número de itens em rankings

# =====================================================================
# FUNÇÕES AUXILIARES
# =====================================================================

@st.cache_data
def carregar_dados(caminho: Path) -> pd.DataFrame:
    """Carrega e faz parse do CSV com tratamento de diferentes separadores."""
    if not caminho.exists():
        raise FileNotFoundError(f"Arquivo {caminho} não encontrado.")
    
    try:
        return pd.read_csv(caminho, encoding="utf-8", sep=";")
    except Exception:
        return pd.read_csv(caminho, encoding="utf-8", sep=",")


def normalizar_colunas(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza nomes de colunas para padrão esperado."""
    mapeamento = {}
    for col in df.columns:
        col_lower = str(col).strip().lower()
        if col_lower == "data":
            mapeamento[col] = "Data"
        elif col_lower == "loja":
            mapeamento[col] = "Loja"
        elif col_lower in ("faturamento", "receita", "valor"):
            mapeamento[col] = "Faturamento"
        elif col_lower in ("pedidos", "qtd_pedidos", "qtd"):
            mapeamento[col] = "Pedidos"
        elif col_lower in ("ticket médio", "ticket medio", "ticket_medio", "ticket-medio"):
            mapeamento[col] = "Ticket Médio"
    
    return df.rename(columns=mapeamento)


def limpar_valor_monetario(serie: pd.Series) -> pd.Series:
    """Remove formatação monetária brasileira e converte para float."""
    if pd.api.types.is_string_dtype(serie):
        return (
            serie.astype(str)
            .str.replace("R$", "", regex=False)
            .str.replace(" ", "", regex=False)
            .str.replace(".", "", regex=False)
            .str.replace(",", ".")
        )
    return serie


def processar_dados(df: pd.DataFrame) -> pd.DataFrame:
    """Aplica limpeza e transformações nos dados."""
    # Validação de colunas
    if not COLUNAS_REQUERIDAS.issubset(set(df.columns)):
        raise ValueError(f"CSV precisa ter ao menos: {', '.join(COLUNAS_REQUERIDAS)}")
    
    # Conversão de tipos
    df["Data"] = pd.to_datetime(df["Data"], errors="coerce", dayfirst=True)
    df = df.dropna(subset=["Data"]).copy()
    df["Loja"] = df["Loja"].astype(str).str.strip().str.title()
    
    # Limpeza de valores monetários
    df["Faturamento"] = pd.to_numeric(
        limpar_valor_monetario(df.get("Faturamento", pd.Series([0]))),
        errors="coerce"
    ).fillna(0.0)
    
    df["Pedidos"] = pd.to_numeric(
        limpar_valor_monetario(df.get("Pedidos", pd.Series([0]))),
        errors="coerce"
    ).fillna(0).astype(int)
    
    if "Ticket Médio" in df.columns:
        df["Ticket Médio"] = pd.to_numeric(
            limpar_valor_monetario(df["Ticket Médio"]),
            errors="coerce"
        )
    else:
        df["Ticket Médio"] = np.nan
    
    # Colunas derivadas
    df["Ano"] = df["Data"].dt.year
    df["Mes"] = df["Data"].dt.month
    df["MesNome"] = df["Mes"].map(MESES)
    df["Mes/Ano"] = df["Data"].dt.to_period("M").dt.to_timestamp()
    
    return df.sort_values(["Data", "Loja"]).reset_index(drop=True)


def calcular_variacao_mom(
    df: pd.DataFrame,
    meses: List[pd.Timestamp],
    coluna: str
) -> Optional[float]:
    """Calcula variação Month-over-Month (MoM) em percentual."""
    if len(meses) < 2:
        return None
    
    mes_atual, mes_anterior = meses[-1], meses[-2]
    valor_atual = df[df["Mes/Ano"] == mes_atual][coluna].sum()
    valor_anterior = df[df["Mes/Ano"] == mes_anterior][coluna].sum()
    
    if valor_anterior == 0:
        return None
    
    return (valor_atual - valor_anterior) / valor_anterior * 100


def calcular_variacao_ticket_mom(
    df: pd.DataFrame,
    meses: List[pd.Timestamp]
) -> Optional[float]:
    """Calcula variação MoM do ticket médio."""
    if len(meses) < 2:
        return None
    
    mes_atual, mes_anterior = meses[-1], meses[-2]
    
    # Calcula tickets médios
    atual = df[df["Mes/Ano"] == mes_atual]
    anterior = df[df["Mes/Ano"] == mes_anterior]
    
    ticket_atual = atual["Faturamento"].sum() / atual["Pedidos"].sum() if atual["Pedidos"].sum() > 0 else np.nan
    ticket_anterior = anterior["Faturamento"].sum() / anterior["Pedidos"].sum() if anterior["Pedidos"].sum() > 0 else np.nan
    
    if pd.notna(ticket_anterior) and ticket_anterior > 0:
        return (ticket_atual - ticket_anterior) / ticket_anterior * 100
    
    return None


def formatar_moeda(valor: float) -> str:
    """Formata valor para padrão monetário brasileiro."""
    return f"R$ {valor:,.2f}".replace(",", "X").replace(".", ",").replace("X", ".")


def formatar_inteiro(valor: float) -> str:
    """Formata inteiro com separador de milhares."""
    return f"{int(valor):,}".replace(",", ".")


def formatar_percentual(valor: Optional[float]) -> str:
    """Formata percentual com 1 casa decimal."""
    return f"{valor:.1f}%".replace(".", ",") if valor is not None else "—"


def gerar_alertas(
    mensal: pd.DataFrame,
    dados_filtrados: pd.DataFrame,
    meses: List[pd.Timestamp]
) -> List[Tuple[str, str]]:
    """
    Gera lista de alertas com base em quedas, recordes e lojas sem faturamento.
    Retorna lista de tuplas (tipo, mensagem) onde tipo é 'success', 'warning' ou 'error'.
    """
    alertas = []
    
    # Alertas de quedas MoM
    if len(meses) >= 2:
        mes_atual, mes_anterior = meses[-1], meses[-2]
        
        atual = mensal[mensal["Mes/Ano"] == mes_atual].set_index("Loja")
        anterior = mensal[mensal["Mes/Ano"] == mes_anterior].set_index("Loja")
        
        comparacao = atual[["Faturamento"]].join(
            anterior[["Faturamento"]],
            lsuffix="_atual",
            rsuffix="_anterior"
        )
        comparacao["var_%"] = (
            (comparacao["Faturamento_atual"] - comparacao["Faturamento_anterior"]) /
            comparacao["Faturamento_anterior"].replace({0: np.nan}) * 100
        )
        
        quedas = comparacao.dropna().sort_values("var_%")
        
        for loja, row in quedas.iterrows():
            if row["var_%"] <= QUEDA_ALERTA_THRESHOLD:
                alertas.append((
                    "warning",
                    f"⚠️ {loja}: queda de {row['var_%']:.1f}% vs mês anterior."
                ))
    
    # Alertas de lojas sem faturamento
    lojas_zeradas = dados_filtrados.groupby("Loja")["Faturamento"].sum()
    for loja in lojas_zeradas[lojas_zeradas == 0].index:
        alertas.append(("error", f"🛑 {loja}: sem faturamento neste recorte."))
    
    # Alertas de recordes
    for loja, grupo in mensal.groupby("Loja"):
        grupo = grupo.sort_values("Mes/Ano")
        if len(grupo) >= 2:
            fat_ultimo = grupo.iloc[-1]["Faturamento"]
            if fat_ultimo == grupo["Faturamento"].max() and fat_ultimo > 0:
                alertas.append((
                    "success",
                    f"✅ {loja}: maior faturamento mensal do histórico no recorte."
                ))
    
    return alertas


# =====================================================================
# CONFIGURAÇÃO DA PÁGINA
# =====================================================================
st.set_page_config(
    page_title="Faturamento — Insights",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Faturamento por Loja — Insights")
st.caption(
    "Interface minimalista focada em decisões: KPIs com contexto, "
    "rankings, alertas, sazonalidade e tabela."
)

# =====================================================================
# CARREGAMENTO E PROCESSAMENTO DOS DADOS
# =====================================================================
try:
    caminho_csv = (
        Path(__file__).parent / "Faturamento.csv"
        if "__file__" in globals()
        else Path("Faturamento.csv")
    )
    
    dados_raw = carregar_dados(caminho_csv)
    dados_raw = normalizar_colunas(dados_raw)
    dados = processar_dados(dados_raw)
    
except FileNotFoundError as e:
    st.error(f"❌ {e}")
    st.stop()
except ValueError as e:
    st.error(f"❌ {e}")
    st.stop()
except Exception as e:
    st.error(f"❌ Erro ao processar dados: {e}")
    st.stop()

# =====================================================================
# SIDEBAR: FILTROS
# =====================================================================
st.sidebar.header("🎚️ Filtros")

anos_disponiveis = sorted(dados["Ano"].unique().tolist())
meses_disponiveis = sorted(dados["Mes"].unique().tolist())
lojas_disponiveis = sorted(dados["Loja"].dropna().unique().tolist())

anos_selecionados = st.sidebar.multiselect(
    "Ano",
    options=anos_disponiveis,
    default=anos_disponiveis
)

mes_opcoes = ["Todos"] + [f"{m:02d} - {MESES[m]}" for m in meses_disponiveis]
mes_selecionado = st.sidebar.selectbox("Mês", mes_opcoes, index=0)

lojas_selecionadas = st.sidebar.multiselect(
    "Lojas (manual)",
    options=lojas_disponiveis,
    default=lojas_disponiveis
)

# Aplicar filtros
dados_filtrados = dados[dados["Ano"].isin(anos_selecionados)].copy()

if mes_selecionado != "Todos":
    mes_numero = int(mes_selecionado.split(" - ")[0])
    dados_filtrados = dados_filtrados[dados_filtrados["Mes"] == mes_numero]

if lojas_selecionadas:
    dados_filtrados = dados_filtrados[dados_filtrados["Loja"].isin(lojas_selecionadas)]

if dados_filtrados.empty:
    st.warning("⚠️ Sem dados para este recorte. Ajuste os filtros.")
    st.stop()

# =====================================================================
# AGREGAÇÕES E CÁLCULOS
# =====================================================================
# Agregação mensal
dados_mensal = (
    dados_filtrados
    .groupby(["Loja", "Mes/Ano"], as_index=False)
    .agg({"Faturamento": "sum", "Pedidos": "sum"})
    .sort_values(["Loja", "Mes/Ano"])
)
dados_mensal["Ticket Médio"] = (
    dados_mensal["Faturamento"] / dados_mensal["Pedidos"].replace(0, np.nan)
)

# KPIs gerais
faturamento_total = float(dados_filtrados["Faturamento"].sum())
pedidos_total = int(dados_filtrados["Pedidos"].sum())
ticket_medio_geral = faturamento_total / pedidos_total if pedidos_total > 0 else 0.0

# Cálculo de variações MoM
meses_unicos = sorted(dados_mensal["Mes/Ano"].unique())
variacao_fat = calcular_variacao_mom(dados_mensal, meses_unicos, "Faturamento")
variacao_ped = calcular_variacao_mom(dados_mensal, meses_unicos, "Pedidos")
variacao_ticket = calcular_variacao_ticket_mom(dados_mensal, meses_unicos)

# =====================================================================
# KPIS PRINCIPAIS
# =====================================================================
col1, col2, col3, col4 = st.columns(4)

with col1:
    lojas_ativas = (
        dados_mensal[dados_mensal["Mes/Ano"] == meses_unicos[-1]]["Loja"].nunique()
        if meses_unicos else 0
    )
    st.metric("🏪 Lojas ativas no recorte", lojas_ativas)

with col2:
    st.metric(
        "💸 Faturamento (R$)",
        formatar_moeda(faturamento_total),
        delta=formatar_percentual(variacao_fat) if variacao_fat is not None else None
    )

with col3:
    st.metric(
        "🧾 Pedidos",
        formatar_inteiro(pedidos_total),
        delta=formatar_percentual(variacao_ped) if variacao_ped is not None else None
    )

with col4:
    st.metric(
        "🎟️ Ticket médio (R$)",
        formatar_moeda(ticket_medio_geral),
        delta=formatar_percentual(variacao_ticket) if variacao_ticket is not None else None
    )

st.caption(
    "ℹ️ **Os deltas (%) mostram a variação mês contra mês (MoM) do mesmo recorte**: "
    "comparamos o mês atual com o mês imediatamente anterior."
)

# =====================================================================
# ABAS
# =====================================================================
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Gráficos",
    "🏆 Rankings",
    "🚨 Alertas",
    "🗓️ Sazonalidade",
    "📋 Tabela"
])

# --- ABA 1: GRÁFICOS ---
with tab1:
    col_g1, col_g2 = st.columns(2)
    
    with col_g1:
        fig_fat = px.line(
            dados_mensal,
            x="Mes/Ano",
            y="Faturamento",
            color="Loja",
            markers=True,
            title="Faturamento mensal por loja"
        )
        fig_fat.update_layout(hovermode="x unified", yaxis_tickprefix="R$ ")
        st.plotly_chart(fig_fat, use_container_width=True)
    
    with col_g2:
        fig_ped = px.line(
            dados_mensal,
            x="Mes/Ano",
            y="Pedidos",
            color="Loja",
            markers=True,
            title="Pedidos mensais por loja"
        )
        fig_ped.update_layout(hovermode="x unified")
        st.plotly_chart(fig_ped, use_container_width=True)
    
    fig_ticket = px.line(
        dados_mensal,
        x="Mes/Ano",
        y="Ticket Médio",
        color="Loja",
        markers=True,
        title="Ticket médio mensal (R$)"
    )
    fig_ticket.update_layout(yaxis_tickprefix="R$ ")
    st.plotly_chart(fig_ticket, use_container_width=True)

# --- ABA 2: RANKINGS ---
with tab2:
    st.subheader("Top 10 — Faturamento no recorte")
    top_faturamento = (
        dados_filtrados
        .groupby("Loja")["Faturamento"]
        .sum()
        .sort_values(ascending=False)
        .head(TOP_N)
        .reset_index()
    )
    
    if not top_faturamento.empty:
        fig_top = px.bar(
            top_faturamento,
            x="Loja",
            y="Faturamento",
            title=f"Top {TOP_N} por Faturamento (R$)"
        )
        fig_top.update_layout(yaxis_tickprefix="R$ ")
        st.plotly_chart(fig_top, use_container_width=True)
    
    st.subheader("Maiores crescimentos (MoM)")
    if len(meses_unicos) >= 2:
        mes_atual, mes_anterior = meses_unicos[-1], meses_unicos[-2]
        
        atual = dados_mensal[dados_mensal["Mes/Ano"] == mes_atual].set_index("Loja")
        anterior = dados_mensal[dados_mensal["Mes/Ano"] == mes_anterior].set_index("Loja")
        
        comparacao = atual[["Faturamento"]].join(
            anterior[["Faturamento"]],
            lsuffix="_atual",
            rsuffix="_anterior"
        )
        comparacao["MoM (%)"] = (
            (comparacao["Faturamento_atual"] - comparacao["Faturamento_anterior"]) /
            comparacao["Faturamento_anterior"].replace({0: np.nan}) * 100
        )
        
        crescimentos = (
            comparacao
            .dropna(subset=["MoM (%)"])
            .sort_values("MoM (%)", ascending=False)
            .head(TOP_N)
            .reset_index()
        )
        
        fig_cresc = px.bar(
            crescimentos,
            x="Loja",
            y="MoM (%)",
            title=f"Top {TOP_N} Crescimentos MoM (%)"
        )
        st.plotly_chart(fig_cresc, use_container_width=True)
    else:
        st.info("Precisa de pelo menos 2 meses para comparar.")

# --- ABA 3: ALERTAS ---
with tab3:
    st.subheader("Alertas do período")
    
    alertas = gerar_alertas(dados_mensal, dados_filtrados, meses_unicos)
    
    # Exibir gráfico de quedas se houver dados suficientes
    if len(meses_unicos) >= 2:
        mes_atual, mes_anterior = meses_unicos[-1], meses_unicos[-2]
        
        atual = dados_mensal[dados_mensal["Mes/Ano"] == mes_atual].set_index("Loja")
        anterior = dados_mensal[dados_mensal["Mes/Ano"] == mes_anterior].set_index("Loja")
        
        comparacao = atual[["Faturamento"]].join(
            anterior[["Faturamento"]],
            lsuffix="_atual",
            rsuffix="_anterior"
        )
        comparacao["var_%"] = (
            (comparacao["Faturamento_atual"] - comparacao["Faturamento_anterior"]) /
            comparacao["Faturamento_anterior"].replace({0: np.nan}) * 100
        )
        
        quedas = (
            comparacao
            .dropna()
            .sort_values("var_%")
            .head(TOP_N)
            .reset_index()
        )
        
        fig_quedas = px.bar(
            quedas,
            x="Loja",
            y="var_%",
            title="Maiores quedas (MoM %)"
        )
        st.plotly_chart(fig_quedas, use_container_width=True)
    
    # Exibir alertas
    if alertas:
        for tipo, mensagem in alertas:
            if tipo == "success":
                st.success(mensagem)
            elif tipo == "error":
                st.error(mensagem)
            else:
                st.warning(mensagem)
    else:
        st.success("✅ Nenhum alerta relevante encontrado.")

# --- ABA 4: SAZONALIDADE ---
with tab4:
    st.subheader("Faturamento por mês do ano (histórico)")
    
    sazonalidade = (
        dados
        .groupby(dados["Data"].dt.month)["Faturamento"]
        .sum()
        .reset_index()
    )
    sazonalidade.columns = ["Mes", "Faturamento"]
    sazonalidade["Mês"] = sazonalidade["Mes"].map(lambda m: f"{m:02d} - {MESES[m]}")
    
    fig_saz = px.bar(
        sazonalidade,
        x="Mês",
        y="Faturamento",
        title="Sazonalidade — Histórico"
    )
    fig_saz.update_layout(yaxis_tickprefix="R$ ")
    st.plotly_chart(fig_saz, use_container_width=True)
    
    st.subheader("Heatmap — Loja x Mês/Ano (recorte)")
    pivot = dados_mensal.pivot_table(
        index="Loja",
        columns="Mes/Ano",
        values="Faturamento",
        aggfunc="sum",
        fill_value=0.0
    )
    
    if not pivot.empty:
        fig_heat = px.imshow(
            pivot,
            aspect="auto",
            labels=dict(x="Mês/Ano", y="Loja", color="R$"),
            title="Heatmap de Faturamento"
        )
        st.plotly_chart(fig_heat, use_container_width=True)

# --- ABA 5: TABELA ---
with tab5:
    st.subheader("Transacional do recorte")
    
    colunas_exibir = [
        c for c in [
            "Data", "Loja", "Faturamento", "Pedidos", "Ticket Médio",
            "Ano", "Mes", "MesNome", "Mes/Ano"
        ]
        if c in dados_filtrados.columns
    ]
    
    st.dataframe(
        dados_filtrados[colunas_exibir],
        use_container_width=True,
        hide_index=True
    )
    
    st.subheader("Agregado mensal por loja (recorte)")
    st.dataframe(dados_mensal, use_container_width=True, hide_index=True)

# =====================================================================
# INSIGHTS E DOCUMENTAÇÃO
# =====================================================================
st.markdown("""
### 💡 Insights rápidos
- **Concentre-se nas variações MoM**: os deltas (%) nos KPIs comparam o mês atual do recorte com o mês anterior do recorte.
- **Crescimentos/Quedas**: verifique a aba **🏆 Rankings** (Top MoM) e **🚨 Alertas** (quedas relevantes ≤ −30%).
- **Sazonalidade**: use **🗓️ Sazonalidade** para planejar estoque e campanhas em meses historicamente fortes/fracos.
""")

with st.expander("🧮 Como calculamos & objetivo do dashboard"):
    st.markdown("""
**Objetivo**: oferecer uma visão **clara e mínima** do desempenho por loja, destacando **o que mudou** e **onde agir** (quedas, picos e sazonalidade).

**Fórmulas (resumo)**
- **Faturamento total (recorte)** = soma de `Faturamento` dos registros filtrados.
- **Pedidos totais (recorte)** = soma de `Pedidos` dos registros filtrados.
- **Ticket médio (recorte)** = `Faturamento total / Pedidos totais`.
- **MoM (%)** (para cada KPI) = `(valor_mês_atual − valor_mês_anterior) / valor_mês_anterior × 100`.
  - Para **Ticket médio**, primeiro calculamos `Faturamento_mês / Pedidos_mês` em cada mês e depois aplicamos a fórmula de MoM.

**Melhorias da versão 2.0**
- ✅ Código modular com funções reutilizáveis
- ✅ Cache de dados para melhor performance
- ✅ Tratamento de erros aprimorado
- ✅ Constantes bem definidas
- ✅ Type hints e documentação
- ✅ Código mais limpo e manutenível
""")

st.caption("Minimalista por padrão; aprofunde-se conforme a necessidade nas abas acima.")