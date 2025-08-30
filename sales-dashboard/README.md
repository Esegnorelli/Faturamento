# Dashboard de Faturamento - Streamlit

Dashboard interativo para visualização de dados de faturamento, desenvolvido com Streamlit e Python.

## 🚀 Funcionalidades

- **Métricas principais**: Visualização de KPIs como faturamento total, pedidos e ticket médio
- **Gráficos interativos**: Análise temporal e comparativa com Plotly
- **Filtros dinâmicos**: Filtragem por loja e ano
- **Interface responsiva**: Layout adaptável desenvolvido com Streamlit
- **Análise de tendências**: Acompanhamento de crescimento e performance

## 📊 Dados

O dashboard utiliza dados de:
- Faturamento por loja e período
- Quantidade de pedidos
- Ticket médio
- Dados temporais (mês/ano)

## 🛠️ Tecnologias

- **Framework**: Streamlit
- **Análise de Dados**: Pandas
- **Visualização**: Plotly Express
- **Linguagem**: Python 3.10+

## 📁 Estrutura do Projeto

```
sales-dashboard/
├── app.py                    # Aplicação principal Streamlit
├── requirements.txt          # Dependências Python
├── logo.png                 # Logo da empresa
├── public/
│   └── Faturamento.csv      # Dados de faturamento
├── PROJETO_COMPLETO.md      # Documentação do projeto
└── README.md               # Este arquivo
```

## 🚀 Como Usar

### Instalação

1. **Clone ou baixe o projeto**
2. **Instale as dependências**:
   ```bash
   pip install -r requirements.txt
   ```

### Execução

3. **Execute a aplicação**:
   ```bash
   streamlit run app.py
   ```
4. **Acesse** o dashboard no navegador (geralmente `http://localhost:8501`)

## 📈 Funcionalidades do Dashboard

### Métricas Principais
- Faturamento total consolidado
- Número total de pedidos
- Ticket médio calculado
- Quantidade de lojas ativas

### Gráficos Interativos
1. **Evolução Temporal**: Linha do tempo mostrando tendências de faturamento
2. **Top Lojas**: Ranking horizontal das lojas por faturamento
3. **Análise Detalhada**: Gráfico de dispersão relacionando faturamento, pedidos e ticket médio

### Filtros
- **Por Ano**: Visualização de dados específicos de um ano ou todos os anos
- **Por Loja**: Seleção múltipla de lojas específicas

### Tabelas Interativas
- **Resumo por Loja**: Consolidação de métricas por loja
- **Dados Completos**: Visualização detalhada de todos os registros

## 📊 Análises Disponíveis

- **Crescimento Anual**: Comparação percentual ano a ano
- **Top 3 Lojas**: Ranking das melhores performances
- **Performance Individual**: Métricas detalhadas por loja
- **Correlações**: Análise da relação entre pedidos, faturamento e ticket médio

## 🔄 Atualizando Dados

Para atualizar os dados:
1. Substitua o arquivo `public/Faturamento.csv`
2. Mantenha o formato: `numero|mes,ano,loja,faturamento,pedidos,ticket,periodo`
3. Reinicie a aplicação Streamlit

## 🎨 Personalização

O Streamlit oferece várias opções de personalização:
- **Tema**: Configure em `.streamlit/config.toml`
- **Layout**: Modifique o código em `app.py`
- **Cores dos Gráficos**: Ajuste as configurações do Plotly

## 🐳 Deploy

Para fazer deploy da aplicação:

### Streamlit Cloud
1. Faça push do código para GitHub
2. Conecte o repositório no [Streamlit Cloud](https://streamlit.io/cloud)
3. Configure o arquivo principal como `app.py`

### Docker
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

## 📋 Requisitos

- Python 3.10+
- Streamlit 1.29.0+
- Pandas 2.1.4+
- Plotly 5.17.0+

## 🤝 Contribuição

Para contribuir:
1. Faça um fork do projeto
2. Crie uma branch para sua feature
3. Commit suas mudanças
4. Abra um Pull Request

## 📄 Licença

Este projeto está sob licença MIT.

---

<div align="center">
  <p>
    Desenvolvido com ❤️ para análise de dados de faturamento<br/>
    <strong>Python • Streamlit • Insights</strong>
  </p>
</div>
