# 🥟 Dashboard de Faturamento - Hora do Pastel

Dashboard interativo para visualização e análise dos dados de faturamento da rede Hora do Pastel.

## 📊 Funcionalidades

- **Métricas Principais**: Visualização de faturamento total, total de pedidos, ticket médio e número de lojas ativas
- **Filtros Dinâmicos**: Filtragem por ano, mês e loja específica
- **Gráficos Interativos**:
  - Faturamento por loja (gráfico de barras)
  - Evolução mensal do faturamento (gráfico de linha)
  - Distribuição de pedidos por loja (gráfico de rosca)
  - Ticket médio por loja (gráfico de barras)
- **Tabela Detalhada**: Dados completos com funcionalidade de pesquisa
- **Design Responsivo**: Otimizado para desktop e dispositivos móveis

## 🚀 Como usar

### Método 1: Servidor Local (Recomendado)
1. **Iniciar servidor**: Execute `python3 servidor.py` no terminal
2. **Acessar dashboard**: O navegador abrirá automaticamente ou acesse http://localhost:8000
3. **Visualizar dados**: Os dados são carregados automaticamente

### Método 2: Arquivo Local
1. **Abrir o dashboard**: Abra o arquivo `index.html` diretamente no navegador
2. **Nota**: Alguns navegadores podem bloquear arquivos locais

### Interagindo com o Dashboard
1. **Aplicar filtros**: Use os filtros no topo para refinar a análise por período ou loja
2. **Interagir com gráficos**: Passe o mouse sobre os gráficos para ver detalhes
3. **Pesquisar na tabela**: Use a caixa de pesquisa para encontrar registros específicos

## 📁 Estrutura do Projeto

```
Faturamento/
├── index.html          # Página principal do dashboard
├── style.css           # Estilos e design responsivo
├── script.js           # Lógica JavaScript e funcionalidades
├── data.js             # Dados em formato JavaScript
├── Faturamento.csv     # Dados de faturamento (original)
├── logo.png           # Logo da Hora do Pastel
├── servidor.py         # Servidor HTTP local
└── README.md          # Esta documentação
```

## 📈 Dados

O dashboard utiliza dados do arquivo `Faturamento.csv` com as seguintes colunas:
- **mes**: Mês do registro (1-12)
- **ano**: Ano do registro
- **loja**: Nome da loja
- **faturamento**: Valor do faturamento em reais
- **pedidos**: Número total de pedidos
- **ticket**: Ticket médio por pedido
- **periodo**: Período no formato YYYY-MM

## 🎨 Design

- **Cores principais**: Gradiente laranja (#ff6b35 a #ff8c42) inspirado na identidade visual da marca
- **Tipografia**: Segoe UI para melhor legibilidade
- **Layout**: Grid responsivo com cards e gráficos organizados
- **Animações**: Transições suaves e animações nos números

## 🛠️ Tecnologias

- **HTML5**: Estrutura semântica
- **CSS3**: Estilização avançada com Flexbox e Grid
- **JavaScript ES6+**: Funcionalidades interativas
- **Chart.js**: Biblioteca para gráficos
- **Papa Parse**: Parser de CSV
- **Design Responsivo**: Media queries para adaptabilidade

## 📱 Responsividade

O dashboard é totalmente responsivo e se adapta a diferentes tamanhos de tela:
- **Desktop** (>768px): Layout completo com gráficos lado a lado
- **Tablet** (768px): Layout adaptado com elementos empilhados
- **Mobile** (<480px): Interface otimizada para toque

## 🔄 Atualizações de Dados

Para atualizar os dados do dashboard:
1. Substitua o arquivo `Faturamento.csv` pelos novos dados
2. Mantenha a mesma estrutura de colunas
3. Recarregue a página no navegador

## 📊 Métricas Calculadas

- **Faturamento Total**: Soma de todos os valores de faturamento filtrados
- **Total de Pedidos**: Soma de todos os pedidos filtrados
- **Ticket Médio**: Faturamento total dividido pelo total de pedidos
- **Lojas Ativas**: Número único de lojas nos dados filtrados

## 🎯 Funcionalidades Extras

- **Pesquisa na Tabela**: Campo de busca para filtrar registros
- **Animações**: Números animados ao carregar
- **Hover Effects**: Efeitos visuais em cards e gráficos
- **Ordenação Automática**: Dados ordenados por período mais recente

## 📞 Suporte

Este dashboard foi desenvolvido especificamente para a análise de dados da rede Hora do Pastel. Para sugestões ou melhorias, entre em contato.

---

**© 2024 Hora do Pastel - Dashboard de Faturamento**
