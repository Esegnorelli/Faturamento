// Variáveis globais
let rawData = [];
let filteredData = [];
let charts = {};

// Configurações dos gráficos
Chart.defaults.font.family = "'Segoe UI', Tahoma, Geneva, Verdana, sans-serif";
Chart.defaults.color = '#666';

// Inicialização
document.addEventListener('DOMContentLoaded', function() {
    loadData();
});

// Carregamento dos dados diretamente do arquivo data.js
function loadData() {
    try {
        // Usar dados do arquivo data.js
        rawData = faturamentoData.filter(row => row.mes && row.ano && row.loja);
        filteredData = [...rawData];
        
        console.log(`Dados carregados: ${rawData.length} registros`);
        
        populateFilters();
        updateDashboard();
        createCharts();
        populateTable();
        
        // Event listeners para filtros
        document.getElementById('yearFilter').addEventListener('change', applyFilters);
        document.getElementById('monthFilter').addEventListener('change', applyFilters);
        document.getElementById('storeFilter').addEventListener('change', applyFilters);
        
        // Animar métricas após carregar
        setTimeout(animateMetrics, 500);
        
    } catch (error) {
        console.error('Erro ao carregar dados:', error);
        alert('Erro ao carregar dados. Verifique se o arquivo data.js está disponível.');
    }
}

// Popular filtros
function populateFilters() {
    const years = [...new Set(rawData.map(row => row.ano))].sort();
    const stores = [...new Set(rawData.map(row => row.loja))].sort();
    
    const yearFilter = document.getElementById('yearFilter');
    const storeFilter = document.getElementById('storeFilter');
    
    // Limpar filtros existentes
    yearFilter.innerHTML = '<option value="">Todos os anos</option>';
    storeFilter.innerHTML = '<option value="">Todas as lojas</option>';
    
    // Adicionar anos
    years.forEach(year => {
        const option = document.createElement('option');
        option.value = year;
        option.textContent = year;
        yearFilter.appendChild(option);
    });
    
    // Adicionar lojas
    stores.forEach(store => {
        const option = document.createElement('option');
        option.value = store;
        option.textContent = store;
        storeFilter.appendChild(option);
    });
}

// Aplicar filtros
function applyFilters() {
    const yearFilter = document.getElementById('yearFilter').value;
    const monthFilter = document.getElementById('monthFilter').value;
    const storeFilter = document.getElementById('storeFilter').value;
    
    filteredData = rawData.filter(row => {
        let matches = true;
        
        if (yearFilter && row.ano != yearFilter) matches = false;
        if (monthFilter && row.mes != monthFilter) matches = false;
        if (storeFilter && row.loja !== storeFilter) matches = false;
        
        return matches;
    });
    
    updateDashboard();
    updateCharts();
    populateTable();
}

// Atualizar métricas do dashboard
function updateDashboard() {
    const totalRevenue = filteredData.reduce((sum, row) => sum + (row.faturamento || 0), 0);
    const totalOrders = filteredData.reduce((sum, row) => sum + (row.pedidos || 0), 0);
    const avgTicket = totalOrders > 0 ? totalRevenue / totalOrders : 0;
    const activeStores = new Set(filteredData.map(row => row.loja)).size;
    
    document.getElementById('totalRevenue').textContent = formatCurrency(totalRevenue);
    document.getElementById('totalOrders').textContent = formatNumber(totalOrders);
    document.getElementById('avgTicket').textContent = formatCurrency(avgTicket);
    document.getElementById('activeStores').textContent = activeStores;
}

// Criar gráficos
function createCharts() {
    createRevenueByStoreChart();
    createMonthlyEvolutionChart();
    createOrdersByStoreChart();
    createAvgTicketChart();
}

// Atualizar gráficos
function updateCharts() {
    Object.values(charts).forEach(chart => chart.destroy());
    createCharts();
}

// Gráfico de faturamento por loja
function createRevenueByStoreChart() {
    const storeRevenue = {};
    
    filteredData.forEach(row => {
        if (!storeRevenue[row.loja]) {
            storeRevenue[row.loja] = 0;
        }
        storeRevenue[row.loja] += row.faturamento || 0;
    });
    
    const sortedStores = Object.entries(storeRevenue)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 10); // Top 10 lojas
    
    const ctx = document.getElementById('revenueByStoreChart').getContext('2d');
    charts.revenueByStore = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: sortedStores.map(([store]) => store),
            datasets: [{
                label: 'Faturamento (R$)',
                data: sortedStores.map(([, revenue]) => revenue),
                backgroundColor: '#ff6b35',
                borderColor: '#ff4500',
                borderWidth: 1,
                borderRadius: 5
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: {
                        callback: function(value) {
                            return formatCurrency(value);
                        }
                    }
                },
                x: {
                    ticks: {
                        maxRotation: 45
                    }
                }
            }
        }
    });
}

// Gráfico de evolução mensal
function createMonthlyEvolutionChart() {
    const monthlyData = {};
    
    filteredData.forEach(row => {
        const period = `${row.mes}/${row.ano}`;
        if (!monthlyData[period]) {
            monthlyData[period] = 0;
        }
        monthlyData[period] += row.faturamento || 0;
    });
    
    const sortedPeriods = Object.entries(monthlyData)
        .sort((a, b) => {
            const [monthA, yearA] = a[0].split('/').map(Number);
            const [monthB, yearB] = b[0].split('/').map(Number);
            return (yearA - yearB) || (monthA - monthB);
        });
    
    const ctx = document.getElementById('monthlyEvolutionChart').getContext('2d');
    charts.monthlyEvolution = new Chart(ctx, {
        type: 'line',
        data: {
            labels: sortedPeriods.map(([period]) => period),
            datasets: [{
                label: 'Faturamento Mensal (R$)',
                data: sortedPeriods.map(([, revenue]) => revenue),
                borderColor: '#ff6b35',
                backgroundColor: 'rgba(255, 107, 53, 0.1)',
                borderWidth: 3,
                fill: true,
                tension: 0.4,
                pointBackgroundColor: '#ff6b35',
                pointBorderColor: '#fff',
                pointBorderWidth: 2,
                pointRadius: 6
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: {
                        callback: function(value) {
                            return formatCurrency(value);
                        }
                    }
                }
            }
        }
    });
}

// Gráfico de pedidos por loja
function createOrdersByStoreChart() {
    const storeOrders = {};
    
    filteredData.forEach(row => {
        if (!storeOrders[row.loja]) {
            storeOrders[row.loja] = 0;
        }
        storeOrders[row.loja] += row.pedidos || 0;
    });
    
    const sortedStores = Object.entries(storeOrders)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 10); // Top 10 lojas
    
    const ctx = document.getElementById('ordersByStoreChart').getContext('2d');
    charts.ordersByStore = new Chart(ctx, {
        type: 'doughnut',
        data: {
            labels: sortedStores.map(([store]) => store),
            datasets: [{
                data: sortedStores.map(([, orders]) => orders),
                backgroundColor: [
                    '#ff6b35', '#ff8c42', '#ffa726', '#ffb74d', 
                    '#ffcc80', '#ffe0b2', '#ff7043', '#ff5722',
                    '#f4511e', '#d84315'
                ],
                borderWidth: 2,
                borderColor: '#fff'
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    position: 'bottom',
                    labels: {
                        padding: 20,
                        usePointStyle: true
                    }
                }
            }
        }
    });
}

// Gráfico de ticket médio por loja
function createAvgTicketChart() {
    const storeTickets = {};
    const storeCounts = {};
    
    filteredData.forEach(row => {
        if (!storeTickets[row.loja]) {
            storeTickets[row.loja] = 0;
            storeCounts[row.loja] = 0;
        }
        if (row.ticket) {
            storeTickets[row.loja] += row.ticket;
            storeCounts[row.loja]++;
        }
    });
    
    const avgTickets = {};
    Object.keys(storeTickets).forEach(store => {
        avgTickets[store] = storeCounts[store] > 0 ? storeTickets[store] / storeCounts[store] : 0;
    });
    
    const sortedStores = Object.entries(avgTickets)
        .sort((a, b) => b[1] - a[1])
        .slice(0, 10); // Top 10 lojas
    
    const ctx = document.getElementById('avgTicketChart').getContext('2d');
    charts.avgTicket = new Chart(ctx, {
        type: 'bar',
        data: {
            labels: sortedStores.map(([store]) => store),
            datasets: [{
                label: 'Ticket Médio (R$)',
                data: sortedStores.map(([, avgTicket]) => avgTicket),
                backgroundColor: '#4caf50',
                borderColor: '#388e3c',
                borderWidth: 1,
                borderRadius: 5
            }]
        },
        options: {
            responsive: true,
            plugins: {
                legend: {
                    display: false
                }
            },
            scales: {
                y: {
                    beginAtZero: true,
                    ticks: {
                        callback: function(value) {
                            return formatCurrency(value);
                        }
                    }
                },
                x: {
                    ticks: {
                        maxRotation: 45
                    }
                }
            }
        }
    });
}

// Popular tabela
function populateTable() {
    const tableBody = document.getElementById('tableBody');
    tableBody.innerHTML = '';
    
    // Ordenar dados por período (mais recente primeiro)
    const sortedData = [...filteredData].sort((a, b) => {
        if (a.ano !== b.ano) return b.ano - a.ano;
        if (a.mes !== b.mes) return b.mes - a.mes;
        return a.loja.localeCompare(b.loja);
    });
    
    sortedData.forEach(row => {
        const tr = document.createElement('tr');
        tr.innerHTML = `
            <td>${formatPeriod(row.mes, row.ano)}</td>
            <td>${row.loja}</td>
            <td>${formatCurrency(row.faturamento)}</td>
            <td>${formatNumber(row.pedidos)}</td>
            <td>${formatCurrency(row.ticket)}</td>
        `;
        tableBody.appendChild(tr);
    });
}

// Funções utilitárias de formatação
function formatCurrency(value) {
    if (value === null || value === undefined || isNaN(value)) return 'R$ 0,00';
    return new Intl.NumberFormat('pt-BR', {
        style: 'currency',
        currency: 'BRL'
    }).format(value);
}

function formatNumber(value) {
    if (value === null || value === undefined || isNaN(value)) return '0';
    return new Intl.NumberFormat('pt-BR').format(value);
}

function formatPeriod(month, year) {
    const months = [
        'Jan', 'Fev', 'Mar', 'Abr', 'Mai', 'Jun',
        'Jul', 'Ago', 'Set', 'Out', 'Nov', 'Dez'
    ];
    return `${months[month - 1]}/${year}`;
}

// Função para gerar cores aleatórias para gráficos
function generateColors(count) {
    const colors = [];
    const baseColors = [
        '#ff6b35', '#ff8c42', '#ffa726', '#ffb74d', '#ffcc80',
        '#4caf50', '#66bb6a', '#81c784', '#a5d6a7', '#c8e6c9',
        '#2196f3', '#42a5f5', '#64b5f6', '#90caf9', '#bbdefb'
    ];
    
    for (let i = 0; i < count; i++) {
        colors.push(baseColors[i % baseColors.length]);
    }
    
    return colors;
}

// Função para animação dos números
function animateValue(element, start, end, duration) {
    const startTime = performance.now();
    const range = end - start;
    
    function updateValue(currentTime) {
        const elapsed = currentTime - startTime;
        const progress = Math.min(elapsed / duration, 1);
        
        const current = start + (range * easeOutQuart(progress));
        
        if (element.id.includes('Revenue') || element.id.includes('Ticket')) {
            element.textContent = formatCurrency(current);
        } else {
            element.textContent = formatNumber(Math.round(current));
        }
        
        if (progress < 1) {
            requestAnimationFrame(updateValue);
        }
    }
    
    requestAnimationFrame(updateValue);
}

// Função de easing para animação
function easeOutQuart(t) {
    return 1 - Math.pow(1 - t, 4);
}

// Função para adicionar animação aos cards de métricas
function animateMetrics() {
    const totalRevenue = filteredData.reduce((sum, row) => sum + (row.faturamento || 0), 0);
    const totalOrders = filteredData.reduce((sum, row) => sum + (row.pedidos || 0), 0);
    const avgTicket = totalOrders > 0 ? totalRevenue / totalOrders : 0;
    const activeStores = new Set(filteredData.map(row => row.loja)).size;
    
    animateValue(document.getElementById('totalRevenue'), 0, totalRevenue, 2000);
    animateValue(document.getElementById('totalOrders'), 0, totalOrders, 1500);
    animateValue(document.getElementById('avgTicket'), 0, avgTicket, 1800);
    animateValue(document.getElementById('activeStores'), 0, activeStores, 1000);
}

// Atualizar dashboard (versão sem animação para filtros)
function updateDashboard() {
    const totalRevenue = filteredData.reduce((sum, row) => sum + (row.faturamento || 0), 0);
    const totalOrders = filteredData.reduce((sum, row) => sum + (row.pedidos || 0), 0);
    const avgTicket = totalOrders > 0 ? totalRevenue / totalOrders : 0;
    const activeStores = new Set(filteredData.map(row => row.loja)).size;
    
    document.getElementById('totalRevenue').textContent = formatCurrency(totalRevenue);
    document.getElementById('totalOrders').textContent = formatNumber(totalOrders);
    document.getElementById('avgTicket').textContent = formatCurrency(avgTicket);
    document.getElementById('activeStores').textContent = activeStores;
}

// Função para exportar dados filtrados (funcionalidade extra)
function exportFilteredData() {
    // Criar CSV manualmente
    const headers = ['mes', 'ano', 'loja', 'faturamento', 'pedidos', 'ticket', 'periodo'];
    const csvContent = [
        headers.join(','),
        ...filteredData.map(row => [
            row.mes, row.ano, `"${row.loja}"`, row.faturamento, row.pedidos, row.ticket, row.periodo
        ].join(','))
    ].join('\n');
    
    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'faturamento_filtrado.csv';
    a.click();
    URL.revokeObjectURL(url);
}

// Adicionar funcionalidade de pesquisa na tabela (funcionalidade extra)
function addTableSearch() {
    const searchInput = document.createElement('input');
    searchInput.type = 'text';
    searchInput.placeholder = 'Pesquisar na tabela...';
    searchInput.style.cssText = `
        padding: 0.75rem;
        border: 2px solid #e0e0e0;
        border-radius: 8px;
        margin-bottom: 1rem;
        width: 100%;
        max-width: 300px;
        font-size: 1rem;
    `;
    
    const tableContainer = document.querySelector('.table-container');
    const tableWrapper = document.querySelector('.table-wrapper');
    tableContainer.insertBefore(searchInput, tableWrapper);
    
    searchInput.addEventListener('input', function() {
        const searchTerm = this.value.toLowerCase();
        const rows = document.querySelectorAll('#tableBody tr');
        
        rows.forEach(row => {
            const text = row.textContent.toLowerCase();
            row.style.display = text.includes(searchTerm) ? '' : 'none';
        });
    });
}

// Inicializar funcionalidades extras
setTimeout(() => {
    addTableSearch();
    animateMetrics();
}, 500);
