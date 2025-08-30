# Dashboard Hora do Pastel 🥟

Dashboard interativo para análise de faturamento da rede Hora do Pastel, desenvolvido com Streamlit.

## 📋 Funcionalidades

### KPIs Principais
- **Faturamento Total**: Soma de todo o faturamento no período
- **Total de Pedidos**: Quantidade total de pedidos realizados  
- **Ticket Médio**: Valor médio por pedido (Faturamento ÷ Pedidos)
- **Número de Lojas**: Quantidade de lojas ativas
- **Faturamento Médio por Loja**: Performance média das unidades

### Análises Disponíveis

#### 📅 Análise Temporal
- Evolução mensal do faturamento e pedidos
- Análise de sazonalidade com identificação de picos e vales
- Crescimento anual com percentuais de variação

#### 🏪 Performance por Loja
- Ranking das top 10 lojas por faturamento
- Top 10 lojas por ticket médio
- Análise de correlação entre pedidos e faturamento

#### 🗺️ Visualizações Avançadas
- Mapa de calor temporal por loja
- Gráficos de tendência e crescimento
- Distribuição de faturamento (pizza e histograma)

#### ⚖️ Análise Comparativa
- Comparação entre anos selecionados
- Análise de variação percentual
- Insights automáticos sobre performance

#### 🏆 Ranking e Eficiência
- Score de performance combinando múltiplas métricas
- Análise de tendências por loja usando regressão linear
- Score de eficiência ponderado

#### 💡 Insights Inteligentes
- Recomendações estratégicas automáticas
- Identificação de lojas em crescimento/declínio
- Sugestões para otimização e upselling

#### 🧮 Métricas Avançadas
- Coeficiente de variação
- Índice de concentração (Gini)
- ROI médio estimado

### 🔍 Filtros Dinâmicos
- **Anos**: Seleção múltipla de anos
- **Meses**: Filtro por meses específicos
- **Lojas**: Escolha de lojas para análise

## 🚀 Como Executar

### Pré-requisitos
- Python 3.7 ou superior
- pip (gerenciador de pacotes Python)

### Instalação

1. **Clone ou baixe os arquivos do projeto**
   ```bash
   # Certifique-se de ter os arquivos:
   # - app.py
   # - Faturamento.csv
   # - Logo.png
   # - requirements.txt
   ```

2. **Instale as dependências**
   ```bash
   pip install -r requirements.txt
   ```

3. **Execute o dashboard**
   ```bash
   streamlit run app.py
   ```

4. **Acesse o dashboard**
   - O Streamlit abrirá automaticamente no navegador
   - URL padrão: http://localhost:8501

## 📊 Como Usar o Dashboard

### Interface Principal
1. **Sidebar (Barra Lateral)**: Use os filtros para selecionar período e lojas
2. **KPIs**: Visualize métricas principais no topo da página
3. **Análises**: Navegue pelas diferentes seções de análise
4. **Explicações**: Cada seção possui explicações detalhadas dos cálculos

### Interpretação das Métricas

#### Ticket Médio
- **Cálculo**: Faturamento Total ÷ Total de Pedidos
- **Interpretação**: Valor médio gasto por cliente
- **Uso**: Indicador de eficiência comercial e potencial de upselling

#### Score de Eficiência
- **Faturamento**: 40% do peso total
- **Volume de Pedidos**: 30% do peso total  
- **Ticket Médio**: 30% do peso total
- **Interpretação**: Score de 0-100 que equilibra volume e valor

#### Análise de Tendências
- **Método**: Regressão linear sobre períodos temporais
- **Normalização**: Tendência ÷ Faturamento Médio × 100
- **Interpretação**: % de crescimento/declínio consistente

#### Índice de Concentração (Gini)
- **Range**: 0 (distribuição uniforme) a 1 (máxima concentração)
- **Interpretação**: Indica se poucas lojas concentram o faturamento
- **Uso**: Planejamento de expansão e balanceamento da rede

## 🎯 Insights e Recomendações Automáticas

O dashboard gera automaticamente:

### Insights de Performance
- Identificação da loja líder em faturamento
- Destaque da loja com maior ticket médio
- Análise de crescimento/declínio por período

### Recomendações Estratégicas
- **Otimização**: Lojas no quartil inferior de performance
- **Upselling**: Lojas com ticket médio abaixo da média
- **Sazonalidade**: Períodos de baixa performance para campanhas

## 📁 Estrutura de Dados

### Formato do CSV
O arquivo `Faturamento.csv` deve conter:
- `mes`: Mês (1-12)
- `ano`: Ano (YYYY)
- `loja`: Nome da loja
- `faturamento`: Valor do faturamento
- `pedidos`: Quantidade de pedidos
- `ticket`: Ticket médio
- `periodo`: Período no formato YYYY-MM

### Separador
- Utiliza "|" como separador de colunas

## 🔧 Personalização

### Cores e Temas
- Cor principal: #FF6B35 (laranja Hora do Pastel)
- Esquemas de cores: Viridis, Oranges, Blues, RdYlGn
- CSS customizado para caixas de insight e explicação

### Adição de Novas Métricas
Para adicionar novas análises:
1. Crie a função de cálculo
2. Adicione a visualização usando Plotly
3. Inclua explicações usando `show_explanation()`
4. Adicione insights usando `show_insight()`

## 📞 Suporte

Para dúvidas sobre:
- **Interpretação de métricas**: Consulte as explicações em cada seção
- **Funcionalidades**: Use os tooltips (?) nos elementos
- **Dados detalhados**: Expanda a seção "Ver Dados Detalhados"

## 📈 Próximos Passos Sugeridos

1. **Análise Geográfica**: Adicionar mapas por região
2. **Previsão**: Implementar modelos preditivos
3. **Alertas**: Sistema de notificações para anomalias
4. **Benchmarking**: Comparação com metas e objetivos
5. **Drill-down**: Análise detalhada por categoria de produto

---

**Desenvolvido para a Rede Hora do Pastel** 🥟  
*Dashboard de Business Intelligence para análise de performance*
