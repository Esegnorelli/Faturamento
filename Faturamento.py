from __future__ import annotations
from pathlib import Path
from typing import List, Optional
import io
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

"""
Hora do Pastel — Dashboard Clean
---------------------------------
streamlit run dashboard.py
"""

# ============================
# CONFIGURAÇÃO E ESTILOS
# ============================
st.set_page_config(
    page_title="Hora do Pastel",
    page_icon="🥟",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# CSS customizado para layout limpo
st.markdown("""
<style>
    /* Remover padding extra */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 1rem;
    }
    
    /* Estilo dos cards de métricas */
    [data-testid="stMetricValue"] {
        font-size: 2rem;
        font-weight: 600;
    }
    
    [data-testid="stMetricDelta"] {
        font-size: 0.9rem;
    }
    
    /* Tabs mais clean */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        padding: 0.5rem 1rem;
        font-weight: 500;
    }
    
    /* Menos espaçamento entre elementos */
    .element-container {
        margin-bottom: 0.5rem;
    }
    
    /* Ocultar índices de tabelas */
    .stDataFrame {
        font-size: 0.9rem;
    }
    
    /* Botões mais discretos */
    .stDownloadButton button {
        border: 1px solid #ddd;
        background: white;
        color: #333;
    }
</style>
""", unsafe_allow_html=True)

# ============================
# CONSTANTES
# ============================
MESES = {
    1: "Jan", 2: "Fev", 3: "Mar", 4: "Abr",
    5: "Mai", 6: "Jun", 7: "Jul", 8: "Ago",
    9: "Set", 10: "Out", 11: "Nov", 12: "Dez"
}

COLUNAS_REQUERIDAS = {"Data", "Loja"}
QUEDA_ALERTA = -20
TOP_N = 8

# ============================
# FUNÇÕES DE CARREGAMENTO
# ============================
@st.cache_data(show_spinner=False)
def _ler_csv_bytes(arquivo: bytes) -> pd.DataFrame:
    head = arquivo[:4096].decode("utf-8", errors="ignore")
    sep = ";" if ";" in head else ","
    return pd.read_csv(io.BytesIO(arquivo), sep=sep, encoding="utf-8")

@st.cache_data(show_spinner=False)
def _ler_csv_caminho(caminho: Path) -> pd.DataFrame:
    try:
        return pd.read_csv(caminho, encoding="utf-8-sig", sep=";")
    except:
        return pd.read_csv(caminho, encoding="utf-8-sig", sep=",")

# ============================
# PROCESSAMENTO
# ============================
def _normalizar_colunas(df: pd.DataFrame) -> pd.DataFrame:
    mapa = {}
    for c in df.columns:
        k = str(c).strip().lower()
        if k == "data": mapa[c] = "Data"
        elif k == "loja": mapa[c] = "Loja"
        elif k in ("faturamento", "receita", "valor"): mapa[c] = "Faturamento"
        elif k in ("pedidos", "qtd_pedidos", "qtd"): mapa[c] = "Pedidos"
        elif k in ("ticket médio", "ticket medio", "ticket_medio"): mapa[c] = "Ticket Médio"
    return df.rename(columns=mapa)

def _limpar_valor(serie: pd.Series) -> pd.Series:
    s = serie.astype(str)
    return (s.str.replace("R$", "", regex=False)
             .str.replace(" ", "", regex=False)
             .str.replace(".", "", regex=False)
             .str.replace(",", "."))

def _parse_data(col: pd.Series) -> pd.Series:
    s = col.astype(str).str.strip()
    m_mmyyyy = s.str.match(r"^\d{1,2}/\d{4}$")
    m_ddmmyyyy = s.str.match(r"^\d{1,2}/\d{1,2}/\d{4}$")
    
    dt1 = pd.to_datetime(s.where(m_ddmmyyyy), dayfirst=True, errors="coerce")
    dt2 = pd.to_datetime("01/" + s.where(m_mmyyyy), format="%d/%m/%Y", errors="coerce")
    dt3 = pd.to_datetime(s, dayfirst=True, errors="coerce")
    return dt1.fillna(dt2).fillna(dt3)

def processar_dados(df: pd.DataFrame) -> pd.DataFrame:
    if not COLUNAS_REQUERIDAS.issubset(df.columns):
        raise ValueError(f"CSV precisa ter: {', '.join(COLUNAS_REQUERIDAS)}")
    
    df = df.copy()
    df["Data"] = _parse_data(df["Data"])
    df = df.dropna(subset=["Data"])
    
    df["Loja"] = df["Loja"].astype(str).str.strip().str.title()
    
    if "Faturamento" in df.columns:
        df["Faturamento"] = pd.to_numeric(_limpar_valor(df["Faturamento"]), errors="coerce").fillna(0.0)
    else:
        df["Faturamento"] = 0.0
    
    if "Pedidos" in df.columns:
        df["Pedidos"] = pd.to_numeric(_limpar_valor(df["Pedidos"]), errors="coerce").fillna(0).astype(int)
    else:
        df["Pedidos"] = 0
    
    df["Ano"] = df["Data"].dt.year
    df["Mes"] = df["Data"].dt.month
    df["MesNome"] = df["Mes"].map(MESES)
    df["Mes/Ano"] = df["Data"].dt.to_period("M").dt.to_timestamp()
    
    return df.sort_values(["Data", "Loja"]).reset_index(drop=True)

# ============================
# CÁLCULOS
# ============================
def _variacao_mom(df: pd.DataFrame, meses: List, coluna: str) -> Optional[float]:
    if len(meses) < 2:
        return None
    a, b = meses[-1], meses[-2]
    va = df[df["Mes/Ano"] == a][coluna].sum()
    vb = df[df["Mes/Ano"] == b][coluna].sum()
    return (va - vb) / vb * 100 if vb != 0 else None

def _fmt_moeda(x: float) -> str:
    return f"R$ {x:,.0f}".replace(",", ".")

def _fmt_int(x: float) -> str:
    return f"{int(x):,}".replace(",", ".")

def _fmt_pct(x: Optional[float]) -> str:
    return f"{x:+.1f}%".replace(".", ",") if x is not None else ""

# ============================
# HEADER LIMPO
# ============================
col_logo, col_title = st.columns([1, 11])
with col_logo:
    st.markdown("# 🥟")
with col_title:
    st.markdown("### Hora do Pastel")

# ============================
# CONTROLES COMPACTOS
# ============================
with st.expander("⚙️ Filtros e Configurações", expanded=False):
    c1, c2, c3 = st.columns([2, 2, 1])
    
    with c1:
        up = st.file_uploader("Upload CSV", type=["csv"], label_visibility="collapsed")
    
    with c2:
        st.caption("Formato: Data (mm/aaaa), Loja, Faturamento, Pedidos")
    
    with c3:
        if st.button("🔄 Limpar", use_container_width=True):
            st.rerun()

# ============================
# CARREGAMENTO
# ============================
try:
    if up is not None:
        _raw = _ler_csv_bytes(up.read())
    else:
        caminho = Path("Faturamento.csv")
        _raw = _ler_csv_caminho(caminho)
    
    _raw = _normalizar_colunas(_raw)
    dados = processar_dados(_raw)
except Exception as e:
    st.error(f"❌ Erro ao carregar dados: {e}")
    st.stop()

# ============================
# FILTROS INLINE
# ============================
c1, c2, c3, c4 = st.columns([3, 2, 3, 2])

with c1:
    anos = sorted(dados["Ano"].unique())
    anos_sel = st.multiselect(
        "Período",
        anos,
        default=anos,
        key="anos",
        label_visibility="collapsed",
        placeholder="Selecione os anos"
    )

with c2:
    meses = sorted(dados["Mes"].unique())
    mes_opts = ["Todos"] + [MESES[m] for m in meses]
    mes_sel = st.selectbox(
        "Mês",
        mes_opts,
        index=0,
        label_visibility="collapsed"
    )

with c3:
    lojas = sorted(dados["Loja"].unique())
    lojas_sel = st.multiselect(
        "Lojas",
        lojas,
        default=lojas,
        key="lojas",
        label_visibility="collapsed",
        placeholder="Selecione as lojas"
    )

with c4:
    todos_filtros = st.button("Selecionar tudo", use_container_width=True)
    if todos_filtros:
        st.rerun()

# Aplicar filtros
f = dados[dados["Ano"].isin(anos_sel)] if anos_sel else dados.copy()
if mes_sel != "Todos":
    mes_num = [k for k, v in MESES.items() if v == mes_sel][0]
    f = f[f["Mes"] == mes_num]
if lojas_sel:
    f = f[f["Loja"].isin(lojas_sel)]

if f.empty:
    st.warning("⚠️ Nenhum dado encontrado com os filtros aplicados")
    st.stop()

st.divider()

# ============================
# AGREGAÇÕES
# ============================
mensal = (
    f.groupby(["Loja", "Mes/Ano"], as_index=False)
     .agg({"Faturamento": "sum", "Pedidos": "sum"})
)
mensal["Ticket Médio"] = mensal["Faturamento"] / mensal["Pedidos"].replace(0, np.nan)

fat_total = f["Faturamento"].sum()
ped_total = f["Pedidos"].sum()
ticket_geral = fat_total / ped_total if ped_total > 0 else 0

meses_unicos = sorted(mensal["Mes/Ano"].unique())
var_fat = _variacao_mom(mensal, meses_unicos, "Faturamento")
var_ped = _variacao_mom(mensal, meses_unicos, "Pedidos")

# ============================
# KPIs PRINCIPAIS
# ============================
c1, c2, c3, c4 = st.columns(4)

with c1:
    lojas_ativas = mensal[mensal["Mes/Ano"] == meses_unicos[-1]]["Loja"].nunique() if meses_unicos else 0
    st.metric("Lojas Ativas", lojas_ativas, help="Lojas com movimento no último mês do período")

with c2:
    st.metric(
        "Faturamento",
        _fmt_moeda(fat_total),
        _fmt_pct(var_fat),
        help="Total de faturamento no período selecionado\nDelta: variação vs mês anterior"
    )

with c3:
    st.metric(
        "Pedidos",
        _fmt_int(ped_total),
        _fmt_pct(var_ped),
        help="Total de pedidos no período\nDelta: variação vs mês anterior"
    )

with c4:
    st.metric(
        "Ticket Médio",
        _fmt_moeda(ticket_geral),
        help="Faturamento ÷ Pedidos no período"
    )

st.divider()

# ============================
# TABS CLEAN
# ============================
tab1, tab2, tab3, tab4 = st.tabs(["📊 Visão Geral", "🏆 Rankings", "📈 Análises", "📥 Dados"])

# --- VISÃO GERAL ---
with tab1:
    # Gráfico principal grande
    fig_fat = px.line(
        mensal,
        x="Mes/Ano",
        y="Faturamento",
        color="Loja",
        markers=True,
        title="Evolução do Faturamento"
    )
    fig_fat.update_layout(
        hovermode="x unified",
        yaxis_tickprefix="R$ ",
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=-0.3),
        height=400,
        margin=dict(t=40, b=80)
    )
    st.plotly_chart(fig_fat, use_container_width=True)
    
    # Dois gráficos menores lado a lado
    c1, c2 = st.columns(2)
    
    with c1:
        fig_ped = px.bar(
            mensal.groupby("Mes/Ano")["Pedidos"].sum().reset_index(),
            x="Mes/Ano",
            y="Pedidos",
            title="Total de Pedidos"
        )
        fig_ped.update_layout(showlegend=False, height=300, margin=dict(t=40, b=40))
        st.plotly_chart(fig_ped, use_container_width=True)
    
    with c2:
        ticket_mensal = mensal.groupby("Mes/Ano").agg({
            "Faturamento": "sum",
            "Pedidos": "sum"
        })
        ticket_mensal["Ticket"] = ticket_mensal["Faturamento"] / ticket_mensal["Pedidos"]
        
        fig_tk = px.line(
            ticket_mensal.reset_index(),
            x="Mes/Ano",
            y="Ticket",
            title="Ticket Médio",
            markers=True
        )
        fig_tk.update_layout(showlegend=False, yaxis_tickprefix="R$ ", height=300, margin=dict(t=40, b=40))
        st.plotly_chart(fig_tk, use_container_width=True)

# --- RANKINGS ---
with tab2:
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown("##### 🥇 Top Lojas por Faturamento")
        top_fat = (
            f.groupby("Loja")["Faturamento"]
            .sum()
            .sort_values(ascending=False)
            .head(TOP_N)
            .reset_index()
        )
        
        fig_top = px.bar(
            top_fat,
            y="Loja",
            x="Faturamento",
            orientation="h",
            text="Faturamento"
        )
        fig_top.update_traces(texttemplate='R$ %{text:,.0f}', textposition='outside')
        fig_top.update_layout(showlegend=False, height=400, xaxis_title="", yaxis_title="")
        st.plotly_chart(fig_top, use_container_width=True)
    
    with c2:
        st.markdown("##### 📦 Top Lojas por Pedidos")
        top_ped = (
            f.groupby("Loja")["Pedidos"]
            .sum()
            .sort_values(ascending=False)
            .head(TOP_N)
            .reset_index()
        )
        
        fig_ped = px.bar(
            top_ped,
            y="Loja",
            x="Pedidos",
            orientation="h",
            text="Pedidos"
        )
        fig_ped.update_traces(texttemplate='%{text:,}', textposition='outside')
        fig_ped.update_layout(showlegend=False, height=400, xaxis_title="", yaxis_title="")
        st.plotly_chart(fig_ped, use_container_width=True)
    
    # Crescimento MoM
    if len(meses_unicos) >= 2:
        st.markdown("##### 📈 Variação MoM (Mês sobre Mês)")
        
        a, b = meses_unicos[-1], meses_unicos[-2]
        cur = mensal[mensal["Mes/Ano"] == a].set_index("Loja")
        prv = mensal[mensal["Mes/Ano"] == b].set_index("Loja")
        
        comp = cur[["Faturamento"]].join(prv[["Faturamento"]], lsuffix="_atual", rsuffix="_anterior")
        comp["Variação (%)"] = (
            (comp["Faturamento_atual"] - comp["Faturamento_anterior"]) /
            comp["Faturamento_anterior"].replace({0: np.nan}) * 100
        )
        comp = comp.dropna().sort_values("Variação (%)", ascending=True).reset_index()
        
        # Colorir por crescimento/queda
        comp["Cor"] = comp["Variação (%)"].apply(lambda x: "Crescimento" if x > 0 else "Queda")
        
        fig_mom = px.bar(
            comp,
            y="Loja",
            x="Variação (%)",
            orientation="h",
            color="Cor",
            color_discrete_map={"Crescimento": "#10b981", "Queda": "#ef4444"},
            text="Variação (%)"
        )
        fig_mom.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        fig_mom.update_layout(showlegend=False, height=500, xaxis_title="", yaxis_title="")
        st.plotly_chart(fig_mom, use_container_width=True)

# --- ANÁLISES ---
with tab3:
    c1, c2 = st.columns(2)
    
    with c1:
        st.markdown("##### 📅 Sazonalidade")
        saz = dados.groupby(dados["Data"].dt.month)["Faturamento"].sum().reset_index()
        saz["Mês"] = saz["Data"].map(MESES)
        
        fig_saz = px.bar(saz, x="Mês", y="Faturamento")
        fig_saz.update_layout(showlegend=False, yaxis_tickprefix="R$ ", height=350)
        st.plotly_chart(fig_saz, use_container_width=True)
    
    with c2:
        st.markdown("##### 🎯 Distribuição de Ticket")
        tickets = mensal["Ticket Médio"].dropna()
        
        fig_dist = px.histogram(
            tickets,
            nbins=20,
            labels={"value": "Ticket Médio", "count": "Frequência"}
        )
        fig_dist.update_layout(showlegend=False, height=350)
        st.plotly_chart(fig_dist, use_container_width=True)
    
    # Heatmap compacto
    st.markdown("##### 🗺️ Mapa de Calor - Faturamento por Loja e Período")
    pv = mensal.pivot_table(
        index="Loja",
        columns="Mes/Ano",
        values="Faturamento",
        aggfunc="sum",
        fill_value=0
    )
    
    fig_heat = px.imshow(
        pv,
        aspect="auto",
        color_continuous_scale="Blues",
        labels=dict(color="Faturamento")
    )
    fig_heat.update_layout(height=400)
    st.plotly_chart(fig_heat, use_container_width=True)
    
    # Alertas em cards compactos
    if len(meses_unicos) >= 2:
        st.markdown("##### ⚠️ Alertas")
        
        a, b = meses_unicos[-1], meses_unicos[-2]
        cur = mensal[mensal["Mes/Ano"] == a].set_index("Loja")
        prv = mensal[mensal["Mes/Ano"] == b].set_index("Loja")
        
        comp = cur[["Faturamento"]].join(prv[["Faturamento"]], lsuffix="_atual", rsuffix="_anterior")
        comp["var"] = (
            (comp["Faturamento_atual"] - comp["Faturamento_anterior"]) /
            comp["Faturamento_anterior"].replace({0: np.nan}) * 100
        )
        
        quedas = comp[comp["var"] < QUEDA_ALERTA].dropna().sort_values("var")
        
        if not quedas.empty:
            for loja, row in quedas.iterrows():
                st.warning(f"⚠️ **{loja}**: queda de {row['var']:.1f}% vs mês anterior")
        else:
            st.success("✅ Nenhum alerta de queda significativa")

# --- DADOS ---
with tab4:
    c1, c2 = st.columns([3, 1])
    
    with c1:
        st.markdown("##### 📋 Dados Agregados por Loja e Período")
    
    with c2:
        buf = io.StringIO()
        mensal.to_csv(buf, index=False)
        st.download_button(
            "⬇️ Exportar CSV",
            buf.getvalue(),
            "hora_do_pastel_dados.csv",
            "text/csv",
            use_container_width=True
        )
    
    # Formatar para exibição
    mensal_display = mensal.copy()
    mensal_display["Faturamento"] = mensal_display["Faturamento"].apply(_fmt_moeda)
    mensal_display["Pedidos"] = mensal_display["Pedidos"].apply(lambda x: f"{x:,}".replace(",", "."))
    mensal_display["Ticket Médio"] = mensal_display["Ticket Médio"].apply(lambda x: _fmt_moeda(x) if pd.notna(x) else "—")
    mensal_display["Mes/Ano"] = mensal_display["Mes/Ano"].dt.strftime("%m/%Y")
    
    st.dataframe(
        mensal_display,
        use_container_width=True,
        hide_index=True,
        height=400
    )
    
    # Resumo estatístico compacto
    with st.expander("📊 Estatísticas do Período"):
        stats = f"""
        - **Faturamento Total**: {_fmt_moeda(fat_total)}
        - **Pedidos Total**: {_fmt_int(ped_total)}
        - **Ticket Médio**: {_fmt_moeda(ticket_geral)}
        - **Período**: {f['Data'].min().strftime('%m/%Y')} a {f['Data'].max().strftime('%m/%Y')}
        - **Lojas no Período**: {f['Loja'].nunique()}
        """
        st.markdown(stats)