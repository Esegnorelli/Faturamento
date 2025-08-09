"""
Dashboard de Vendas (2019–2025)
================================

Este aplicativo Streamlit exibe uma visão geral das vendas de uma rede de lojas
com dados de faturamento, pedidos e ticket médio declarados entre 2019 e 2025.
O código embute o conjunto de dados original e oferece filtros interativos para
período, mês e lojas específicas. Além das métricas principais, como
faturamento total, número de pedidos e ticket médio real (calculado a partir
das vendas), o dashboard inclui diversos gráficos e tabelas que ajudam a
visualizar a performance no tempo e comparar lojas.

Melhorias em relação ao código inicial
--------------------------------------

* A função de carregamento (`load_data`) foi otimizada para processar as
  linhas com expressões regulares apenas uma vez e cachear os resultados.
* Foram adicionados novos gráficos que permitem comparar a performance das
  lojas em diferentes métricas: faturamento total, número de pedidos e ticket
  médio real. Assim, o usuário pode escolher qual métrica deseja analisar
  no ranking das lojas.
* Um gráfico de dispersão exibe a relação entre faturamento e ticket médio
  real para os filtros selecionados, possibilitando identificar lojas que se
  destacam por volume de vendas ou ticket médio.
* Foi incluída uma coluna que calcula a diferença entre o ticket médio real
  (calculado a partir de `Faturamento`/`Pedidos`) e o ticket médio declarado
  na base, permitindo ao usuário identificar inconsistências ou oportunidades
  de melhoria.
* A filtragem de lojas conta com um campo de busca por substring, o que
  facilita encontrar uma loja específica em listas extensas.
* Todos os valores monetários são formatados no padrão brasileiro para
  facilitar a leitura.

"""

# -*- coding: utf-8 -*-
import streamlit as st
import pandas as pd
import numpy as np
import re
from io import StringIO
from datetime import datetime
import altair as alt

# Configuração inicial da página
st.set_page_config(page_title="Dashboard Lojas • 2019–2025", layout="wide")

# ========================
# Dados brutos (texto)
# ========================
#
# Os dados foram incorporados diretamente no código para facilitar a execução
# offline. Caso você deseje utilizar dados externos, substitua a variável
# RAW_DATA por outra fonte (por exemplo, `st.file_uploader` ou uma URL).

RAW_DATA = r"""Data	Loja	Faturamento (R$)	Pedidos	Ticket Médio
jan/2022	Caxias do Sul	 R$ 113.573,38 	2839	 R$ 40,00 
jan/2022	Novo Hamburgo	 R$ 109.347,46 	2734	 R$ 40,00 
jan/2022	São Leopoldo	 R$ 86.398,50 	2160	 R$ 40,00 
jan/2022	Barra Shopping	 R$ 81.688,37 	2042	 R$ 40,00 
jan/2022	Campo Bom	 R$ 78.692,98 	1967	 R$ 40,00 
jan/2022	Sapiranga	 R$ 68.319,94 	1708	 R$ 40,00 
jan/2022	Montenegro	 R$ 66.613,24 	1665	 R$ 40,00 
jan/2022	Estância Velha	 R$ 56.168,43 	1404	 R$ 40,00 
jan/2022	Taquara	 R$ 44.794,83 	1120	 R$ 40,00 
jan/2022	Sapucaia do Sul (Vinicius)	 R$ 41.060,43 	1027	 R$ 40,00 
jan/2022	Nova Hartz	 R$ 27.738,19 	693	 R$ 40,00 
jan/2022	Parobé	 R$ 24.847,49 	621	 R$ 40,00 
fev/2022	Caxias do Sul	 R$ 122.454,15 	3061	 R$ 40,00 
fev/2022	Novo Hamburgo	 R$ 102.136,10 	2553	 R$ 40,00 
fev/2022	São Leopoldo	 R$ 86.518,36 	2163	 R$ 40,00 
fev/2022	Campo Bom	 R$ 67.834,05 	1696	 R$ 40,00 
fev/2022	Montenegro	 R$ 61.861,50 	1547	 R$ 40,00 
fev/2022	Barra Shopping	 R$ 57.842,73 	1446	 R$ 40,00 
fev/2022	Estância Velha	 R$ 54.014,92 	1350	 R$ 40,00 
fev/2022	Sapiranga	 R$ 50.509,72 	1263	 R$ 40,00 
fev/2022	Sapucaia do Sul (Vinicius)	 R$ 43.636,69 	1091	 R$ 40,00 
fev/2022	Taquara	 R$ 38.904,13 	973	 R$ 40,00 
fev/2022	Nova Hartz	 R$ 26.261,47 	657	 R$ 40,00 
fev/2022	Parobé	 R$ 11.423,74 	286	 R$ 40,00 
mar/2022	Caxias do Sul	 R$ 127.887,89 	3197	 R$ 40,00 
mar/2022	Novo Hamburgo	 R$ 97.693,73 	2442	 R$ 40,00 
mar/2022	São Leopoldo	 R$ 88.948,60 	2224	 R$ 40,00 
mar/2022	Barra Shopping	 R$ 74.019,52 	1850	 R$ 40,00 
mar/2022	Campo Bom	 R$ 70.078,83 	1752	 R$ 40,00 
mar/2022	Montenegro	 R$ 64.166,06 	1604	 R$ 40,00 
mar/2022	Sapucaia do Sul (Vinicius)	 R$ 53.045,75 	1326	 R$ 40,00 
mar/2022	Estância Velha	 R$ 49.705,28 	1243	 R$ 40,00 
mar/2022	Taquara	 R$ 44.695,67 	1117	 R$ 40,00 
mar/2022	Sapiranga	 R$ 43.262,25 	1082	 R$ 40,00 
mar/2022	Nova Hartz	 R$ 23.973,83 	599	 R$ 40,00 
mar/2022	Parobé	 R$ 20.954,39 	524	 R$ 40,00 
abr/2022	Caxias do Sul	 R$ 127.511,71 	3188	 R$ 40,00 
abr/2022	Novo Hamburgo	 R$ 87.438,82 	2186	 R$ 40,00 
abr/2022	São Leopoldo	 R$ 86.395,35 	2160	 R$ 40,00 
abr/2022	Campo Bom	 R$ 72.548,83 	1814	 R$ 40,00 
abr/2022	Sapiranga	 R$ 70.844,90 	1771	 R$ 40,00 
abr/2022	Barra Shopping	 R$ 70.409,25 	1760	 R$ 40,00 
abr/2022	Montenegro	 R$ 61.582,73 	1540	 R$ 40,00 
abr/2022	Sapucaia do Sul (Vinicius)	 R$ 51.947,18 	1299	 R$ 40,00 
abr/2022	Estância Velha	 R$ 49.985,10 	1250	 R$ 40,00 
abr/2022	Taquara	 R$ 39.036,29 	976	 R$ 40,00 
abr/2022	Parobé	 R$ 27.114,70 	678	 R$ 40,00 
abr/2022	Gravataí (Vinicius)	 R$ 1.692,97 	42	 R$ 40,00 
abr/2022	Passo d' Areia	 R$ 180,60 	5	 R$ 40,00 
mai/2022	Caxias do Sul	 R$ 135.340,16 	3384	 R$ 40,00 
mai/2022	Novo Hamburgo	 R$ 111.853,53 	2796	 R$ 40,00 
mai/2022	Barra Shopping	 R$ 86.642,40 	2166	 R$ 40,00 
mai/2022	São Leopoldo	 R$ 84.227,59 	2106	 R$ 40,00 
mai/2022	Montenegro	 R$ 69.828,35 	1746	 R$ 40,00 
mai/2022	Sapucaia do Sul (Vinicius)	 R$ 68.833,79 	1721	 R$ 40,00 
mai/2022	Campo Bom	 R$ 68.185,28 	1705	 R$ 40,00 
mai/2022	Sapiranga	 R$ 67.072,06 	1677	 R$ 40,00 
mai/2022	Estância Velha	 R$ 50.851,68 	1271	 R$ 40,00 
mai/2022	Passo d' Areia	 R$ 42.091,48 	1052	 R$ 40,00 
mai/2022	Parobé	 R$ 28.810,59 	720	 R$ 40,00 
mai/2022	Gravataí (Vinicius)	 R$ 27.093,40 	677	 R$ 40,00 
jun/2022	Caxias do Sul	 R$ 130.231,57 	3256	 R$ 40,00 
jun/2022	Novo Hamburgo	 R$ 106.592,51 	2665	 R$ 40,00 
jun/2022	São Leopoldo	 R$ 89.730,24 	2243	 R$ 40,00 
jun/2022	Barra Shopping	 R$ 84.036,62 	2101	 R$ 40,00 
jun/2022	Sapucaia do Sul (Vinicius)	 R$ 66.492,38 	1662	 R$ 40,00 
jun/2022	Montenegro	 R$ 66.385,92 	1660	 R$ 40,00 
jun/2022	Campo Bom	 R$ 63.918,09 	1598	 R$ 40,00 
jun/2022	Sapiranga	 R$ 60.678,24 	1517	 R$ 40,00 
jun/2022	Parobé	 R$ 48.445,97 	1211	 R$ 40,00 
jun/2022	Estância Velha	 R$ 46.494,03 	1162	 R$ 40,00 
jun/2022	Bento Gonçalves	 R$ 45.032,19 	1126	 R$ 40,00 
jun/2022	Passo d' Areia	 R$ 42.644,73 	1066	 R$ 39,99 
jun/2022	Gravataí (Vinicius)	 R$ 41.151,55 	1029	 R$ 40,00 
jul/2022	Caxias do Sul	 R$ 126.122,31 	3153	 R$ 40,00 
jul/2022	Barra Shopping	 R$ 94.378,55 	2359	 R$ 40,00 
jul/2022	Novo Hamburgo	 R$ 93.868,42 	2347	 R$ 40,00 
jul/2022	São Leopoldo	 R$ 93.484,16 	2337	 R$ 40,00 
jul/2022	Montenegro	 R$ 62.273,25 	1557	 R$ 40,00 
jul/2022	Campo Bom	 R$ 60.282,49 	1507	 R$ 40,00 
jul/2022	Sapiranga	 R$ 59.075,81 	1477	 R$ 40,00 
jul/2022	Bento Gonçalves	 R$ 55.899,24 	1397	 R$ 40,00 
jul/2022	Sapucaia do Sul (Vinicius)	 R$ 50.160,99 	1254	 R$ 40,00 
jul/2022	Parobé	 R$ 48.783,03 	1220	 R$ 40,00 
jul/2022	Passo d' Areia	 R$ 46.669,99 	1167	 R$ 40,00 
jul/2022	Estância Velha	 R$ 46.650,15 	1166	 R$ 40,00 
jul/2022	Gravataí (Vinicius)	 R$ 30.008,27 	750	 R$ 40,00 
ago/2022	Caxias do Sul	 R$ 116.023,31 	2901	 R$ 40,00 
ago/2022	Montenegro	 R$ 94.135,13 	2353	 R$ 40,00 
ago/2022	Novo Hamburgo	 R$ 94.135,13 	2353	 R$ 40,00 
ago/2022	São Leopoldo	 R$ 86.221,90 	2156	 R$ 40,00 
ago/2022	Barra Shopping	 R$ 83.047,57 	2076	 R$ 40,00 
ago/2022	Sapiranga	 R$ 52.366,66 	1309	 R$ 40,00 
ago/2022	Campo Bom	 R$ 52.123,38 	1303	 R$ 40,00 
ago/2022	Sapucaia do Sul (Vinicius)	 R$ 41.486,34 	1037	 R$ 40,00 
ago/2022	Parobé	 R$ 41.095,43 	1027	 R$ 40,00 
ago/2022	Bento Gonçalves	 R$ 40.071,78 	1002	 R$ 40,00 
ago/2022	Estância Velha	 R$ 38.714,55 	968	 R$ 40,00 
ago/2022	Passo d' Areia	 R$ 37.741,12 	944	 R$ 40,00 
ago/2022	Gravataí (Vinicius)	 R$ 28.416,10 	710	 R$ 40,00 
set/2022	Caxias do Sul	 R$ 123.235,02 	3081	 R$ 40,00 
set/2022	Novo Hamburgo	 R$ 92.538,74 	2313	 R$ 40,00 
set/2022	São Leopoldo	 R$ 85.438,97 	2136	 R$ 40,00 
set/2022	Barra Shopping	 R$ 83.890,10 	2097	 R$ 40,00 
set/2022	Montenegro	 R$ 64.800,87 	1620	 R$ 40,00 
set/2022	Shopping Total	 R$ 58.868,46 	1472	 R$ 40,00 
set/2022	Sapiranga	 R$ 54.144,32 	1354	 R$ 40,00 
set/2022	Estância Velha	 R$ 46.719,89 	1168	 R$ 40,00 
set/2022	Sapucaia do Sul (Vinicius)	 R$ 46.310,48 	1158	 R$ 40,00 
set/2022	Campo Bom	 R$ 45.891,05 	1147	 R$ 40,00 
set/2022	Gravataí (Vinicius)	 R$ 32.847,16 	821	 R$ 40,00 
set/2022	Bento Gonçalves	 R$ 31.500,99 	788	 R$ 40,00 
set/2022	Parobé	 R$ 11.096,31 	277	 R$ 40,00 
out/2022	Caxias do Sul	 R$ 129.679,96 	3242	 R$ 40,00 
out/2022	Novo Hamburgo	 R$ 102.027,07 	2551	 R$ 40,00 
out/2022	São Leopoldo	 R$ 82.395,81 	2060	 R$ 40,00 
out/2022	Barra Shopping	 R$ 78.738,54 	1968	 R$ 40,00 
out/2022	Shopping Total	 R$ 65.063,39 	1627	 R$ 40,00 
out/2022	Montenegro	 R$ 64.401,12 	1610	 R$ 40,00 
out/2022	Sapiranga	 R$ 60.550,62 	1514	 R$ 40,00 
out/2022	Campo Bom	 R$ 52.520,45 	1313	 R$ 40,00 
out/2022	Sapucaia do Sul (Vinicius)	 R$ 47.915,71 	1198	 R$ 40,00 
out/2022	Estância Velha	 R$ 47.000,79 	1175	 R$ 40,00 
out/2022	Bento Gonçalves	 R$ 45.406,58 	1135	 R$ 40,00 
out/2022	Gravataí (Vinicius)	 R$ 35.190,00 	880	 R$ 40,00 
nov/2022	Caxias do Sul	 R$ 133.514,87 	3338	 R$ 40,00 
nov/2022	Novo Hamburgo	 R$ 104.316,03 	2608	 R$ 40,00 
nov/2022	Barra Shopping	 R$ 84.529,68 	2113	 R$ 40,00 
nov/2022	São Leopoldo	 R$ 80.987,86 	2025	 R$ 40,00 
nov/2022	Shopping Total	 R$ 67.357,39 	1684	 R$ 40,00 
nov/2022	Sapiranga	 R$ 62.609,36 	1565	 R$ 40,00 
nov/2022	Montenegro	 R$ 56.334,00 	1408	 R$ 40,00 
nov/2022	Bento Gonçalves	 R$ 50.879,49 	1272	 R$ 40,00 
nov/2022	Campo Bom	 R$ 47.723,57 	1193	 R$ 40,00 
nov/2022	Sapucaia do Sul (Vinicius)	 R$ 47.287,04 	1182	 R$ 40,00 
nov/2022	Estância Velha	 R$ 42.640,67 	1066	 R$ 40,00 
nov/2022	Gravataí (Vinicius)	 R$ 36.978,01 	924	 R$ 40,00 
dez/2022	Caxias do Sul	 R$ 145.346,63 	3634	 R$ 40,00 
dez/2022	Barra Shopping	 R$ 114.359,18 	2859	 R$ 40,00 
dez/2022	Novo Hamburgo	 R$ 113.302,42 	2833	 R$ 40,00 
dez/2022	São Leopoldo	 R$ 83.431,88 	2086	 R$ 40,00 
dez/2022	Sapiranga	 R$ 68.956,81 	1724	 R$ 40,00 
dez/2022	Bento Gonçalves	 R$ 68.852,98 	1721	 R$ 40,00 
dez/2022	Shopping Total	 R$ 68.654,19 	1716	 R$ 40,00 
dez/2022	Montenegro	 R$ 66.423,65 	1661	 R$ 40,00 
dez/2022	Campo Bom	 R$ 60.028,88 	1501	 R$ 40,00 
dez/2022	Estância Velha	 R$ 47.261,17 	1182	 R$ 40,00 
dez/2022	Sapucaia do Sul (Vinicius)	 R$ 23.224,55 	581	 R$ 40,00 
dez/2022	Gravataí (Vinicius)	 R$ 19.684,10 	492	 R$ 40,00 
jan/2023	Caxias do Sul	 R$ 121.801,28 	3045	 R$ 40,00 
jan/2023	Barra Shopping	 R$ 90.203,42 	2255	 R$ 40,00 
jan/2023	Novo Hamburgo	 R$ 88.157,53 	2204	 R$ 40,00 
jan/2023	São Leopoldo	 R$ 69.874,45 	1747	 R$ 40,00 
jan/2023	Sapiranga	 R$ 65.163,84 	1629	 R$ 40,00 
jan/2023	Bento Gonçalves	 R$ 61.906,28 	1548	 R$ 40,00 
jan/2023	Montenegro	 R$ 55.479,93 	1387	 R$ 40,00 
jan/2023	Campo Bom	 R$ 52.482,61 	1312	 R$ 40,00 
jan/2023	Shopping Total	 R$ 52.314,19 	1308	 R$ 40,00 
jan/2023	Estância Velha	 R$ 46.351,82 	1159	 R$ 40,00 
fev/2023	Caxias do Sul	 R$ 115.396,20 	2885	 R$ 40,00 
fev/2023	Novo Hamburgo	 R$ 83.940,51 	2099	 R$ 40,00 
fev/2023	Barra Shopping	 R$ 79.548,77 	1989	 R$ 40,00 
fev/2023	São Leopoldo	 R$ 73.886,59 	1847	 R$ 40,00 
fev/2023	Bento Gonçalves	 R$ 62.326,83 	1558	 R$ 40,00 
fev/2023	Sapiranga	 R$ 57.426,75 	1436	 R$ 40,00 
fev/2023	Montenegro	 R$ 55.948,57 	1399	 R$ 40,00 
fev/2023	Shopping Total	 R$ 48.738,83 	1218	 R$ 40,00 
fev/2023	Campo Bom	 R$ 48.259,78 	1206	 R$ 40,00 
fev/2023	Estância Velha	 R$ 40.680,68 	1017	 R$ 40,00 
mar/2023	Caxias do Sul	 R$ 128.478,28 	3212	 R$ 40,00 
mar/2023	Novo Hamburgo	 R$ 88.832,21 	2221	 R$ 40,00 
mar/2023	Barra Shopping	 R$ 78.838,03 	1971	 R$ 40,00 
mar/2023	São Leopoldo	 R$ 78.824,79 	1971	 R$ 40,00 
mar/2023	Bento Gonçalves	 R$ 60.864,02 	1522	 R$ 40,00 
mar/2023	Shopping Total	 R$ 60.182,44 	1505	 R$ 40,00 
mar/2023	Sapiranga	 R$ 58.512,69 	1463	 R$ 40,00 
mar/2023	Montenegro	 R$ 52.020,93 	1301	 R$ 40,00 
mar/2023	Campo Bom	 R$ 51.589,66 	1290	 R$ 40,00 
mar/2023	Estância Velha	 R$ 46.230,41 	1156	 R$ 40,00 
mar/2023	Canoas	 R$ 22.604,87 	565	 R$ 40,00 
abr/2023	Caxias do Sul	 R$ 132.170,64 	3304	 R$ 40,00 
abr/2023	Novo Hamburgo	 R$ 94.378,13 	2359	 R$ 40,00 
abr/2023	Barra Shopping	 R$ 84.831,17 	2121	 R$ 40,00 
abr/2023	São Leopoldo	 R$ 74.857,51 	1871	 R$ 40,00 
abr/2023	Montenegro	 R$ 60.754,93 	1519	 R$ 40,00 
abr/2023	Shopping Total	 R$ 58.254,57 	1456	 R$ 40,00 
abr/2023	Sapiranga	 R$ 58.055,34 	1451	 R$ 40,00 
abr/2023	Bento Gonçalves	 R$ 55.433,79 	1386	 R$ 40,00 
abr/2023	Campo Bom	 R$ 49.530,31 	1238	 R$ 40,00 
abr/2023	Estância Velha	 R$ 42.372,51 	1059	 R$ 40,00 
abr/2023	Canoas	 R$ 39.736,40 	993	 R$ 40,00 
mai/2023	Caxias do Sul	 R$ 150.441,83 	3761	 R$ 40,00 
mai/2023	Novo Hamburgo	 R$ 118.109,36 	2953	 R$ 40,00 
mai/2023	Barra Shopping	 R$ 85.362,47 	2134	 R$ 40,00 
mai/2023	São Leopoldo	 R$ 76.057,43 	1901	 R$ 40,00 
mai/2023	Bento Gonçalves	 R$ 63.288,65 	1582	 R$ 40,00 
mai/2023	Sapiranga	 R$ 62.304,84 	1558	 R$ 40,00 
mai/2023	Shopping Total	 R$ 61.904,43 	1548	 R$ 40,00 
mai/2023	Campo Bom	 R$ 52.015,35 	1300	 R$ 40,00 
mai/2023	Canoas	 R$ 41.164,67 	1029	 R$ 40,00 
mai/2023	Estância Velha	 R$ 31.634,19 	791	 R$ 40,00 
jun/2023	Caxias do Sul	 R$ 139.405,15 	3485	 R$ 40,00 
jun/2023	Novo Hamburgo	 R$ 125.073,62 	3127	 R$ 40,00 
jun/2023	São Leopoldo	 R$ 94.150,53 	2354	 R$ 40,00 
jun/2023	Barra Shopping	 R$ 92.599,35 	2315	 R$ 40,00 
jun/2023	Shopping Total	 R$ 62.318,55 	1558	 R$ 40,00 
jun/2023	Sapiranga	 R$ 58.466,73 	1462	 R$ 40,00 
jun/2023	Bento Gonçalves	 R$ 55.852,14 	1396	 R$ 40,00 
jun/2023	Campo Bom	 R$ 48.823,21 	1221	 R$ 40,00 
jul/2023	Caxias do Sul	 R$ 132.482,84 	3312	 R$ 40,00 
jul/2023	Bento Gonçalves	 R$ 57.698,26 	1442	 R$ 40,00 
jul/2023	Campo Bom	 R$ 45.965,29 	1155	 R$ 40,00 
jul/2023	Canoas Mathias	 R$ 16.644,68 	416	 R$ 39,80 
set/2023	Canoas Mathias	 R$ 32.600,93 	823	 R$ 40,00 
dez/2023	Campo Bom	 R$ 48.229,69 	1194	 R$ 39,61 
dez/2023	Canoas Mathias	 R$ 31.665,10 	794	 R$ 40,39 
jan/2024	Sapiranga	 R$ 49.430,14 	1238	 R$ 39,88 
jan/2024	Canoas Mathias	 R$ 24.545,82 	614	 R$ 39,93 
fev/2024	Campo Bom	 R$ 44.478,20 	1120	 R$ 39,98 
abr/2024	Campo Bom	 R$ 39.746,22 	983	 R$ 39,71 
mai/2024	Erechim	 R$ 82.184,05 	2038	 R$ 40,43 
jul/2024	Zona Norte	 R$ 5.179,86 	130	 R$ 40,33 
set/2024	Erechim	 R$ 86.813,39 	2153	 R$ 39,85 
set/2024	Campo Bom	 R$ 41.789,88 	1033	 R$ 40,32 
out/2024	Sapiranga	 R$ 72.556,09 	1826	 R$ 40,45 
dez/2024	Floresta	 R$ 40.688,48 	1006	 R$ 39,73 
mar/2025	Floresta	 R$ 61.423,00 	1545	 R$ 40,45 
mar/2025	Porto Alegre Zona Norte	 R$ 10.186,00 	256	 R$ 39,76 
abr/2025	Floresta	 R$ 56.489,37 	1404	 R$ 39,79 
set/2019	Sapiranga	 R$ 126.219,14 	3712	 R$ 40,23 
set/2019	Campo Bom	 R$ 89.989,00 	2647	 R$ 34,00 
out/2019	Campo Bom	 R$ 99.432,20 	2924	 R$ 34,00 
nov/2019	Novo Hamburgo	 R$ 41.256,74 	1213	 R$ 34,00 
dez/2019	Novo Hamburgo	 R$ 124.124,47 	3651	 R$ 34,00 
dez/2019	Sapiranga	 R$ 120.044,74 	3531	 R$ 34,00 
dez/2019	Campo Bom	 R$ 112.115,40 	3298	 R$ 34,00 
fev/2020	Sapiranga	 R$ 128.035,24 	3766	 R$ 34,00 
mar/2020	Taquara	 R$ 80.907,16 	2380	 R$ 34,00 
abr/2020	Campo Bom	 R$ 80.743,90 	2375	 R$ 34,00 
abr/2020	Taquara	 R$ 51.500,00 	1515	 R$ 34,00 
mai/2020	Sapiranga	 R$ 89.308,63 	2627	 R$ 34,00 
mai/2020	Novo Hamburgo	 R$ 78.241,61 	2301	 R$ 34,00 
mai/2020	Campo Bom	 R$ 71.073,60 	2090	 R$ 34,00 
mai/2020	Taquara	 R$ 55.241,10 	1625	 R$ 34,00 
jun/2020	Novo Hamburgo	 R$ 71.306,00 	2097	 R$ 34,00 
ago/2020	Sapiranga	 R$ 112.747,91 	3316	 R$ 34,00 
ago/2020	Taquara	 R$ 85.522,48 	2515	 R$ 34,00 
set/2020	Montenegro	 R$ 83.083,34 	2444	 R$ 34,00 
set/2020	Campo Bom	 R$ 73.069,97 	2149	 R$ 34,00 
out/2020	Novo Hamburgo	 R$ 126.337,62 	3716	 R$ 34,00 
out/2020	Campo Bom	 R$ 73.303,70 	2156	 R$ 34,00 
nov/2020	Montenegro	 R$ 89.574,03 	2635	 R$ 34,00 
nov/2020	Sapiranga	 R$ 89.030,31 	2619	 R$ 34,00 
nov/2020	Taquara	 R$ 68.116,66 	2003	 R$ 34,00 
dez/2020	Campo Bom	 R$ 82.911,98 	2439	 R$ 34,00 
jan/2021	Sapiranga	 R$ 84.739,93 	2492	 R$ 34,00 
jan/2021	Campo Bom	 R$ 74.820,76 	2201	 R$ 34,00 
fev/2021	Novo Hamburgo	 R$ 87.144,53 	2563	 R$ 34,00 
fev/2021	Montenegro	 R$ 80.166,19 	2358	 R$ 34,00 
fev/2021	Caxias do Sul	 R$ 76.513,47 	2250	 R$ 34,00 
mar/2021	Caxias do Sul	 R$ 85.503,81 	2515	 R$ 34,00 
mar/2021	Montenegro	 R$ 76.841,08 	2260	 R$ 34,00 
mar/2021	Campo Bom	 R$ 45.668,95 	1343	 R$ 34,00 
abr/2021	Novo Hamburgo	 R$ 82.512,95 	2427	 R$ 34,00 
abr/2021	Estância Velha	 R$ 62.896,70 	1850	 R$ 34,00 
mai/2021	Novo Hamburgo	 R$ 96.870,49 	2849	 R$ 34,00 
mai/2021	Caxias do Sul	 R$ 76.170,19 	2240	 R$ 34,00 
mai/2021	São Leopoldo	 R$ 66.010,93 	1941	 R$ 34,00 
mai/2021	Sapiranga	 R$ 58.601,29 	1724	 R$ 34,00 
mai/2021	Estância Velha	 R$ 55.885,24 	1644	 R$ 34,00 
mai/2021	Campo Bom	 R$ 44.650,24 	1313	 R$ 34,00 
jun/2021	Montenegro	 R$ 63.582,49 	1870	 R$ 34,00 
jun/2021	São Leopoldo	 R$ 62.740,71 	1845	 R$ 34,00 
jun/2021	Sapiranga	 R$ 56.356,27 	1658	 R$ 34,00 
jun/2021	Campo Bom	 R$ 41.537,05 	1222	 R$ 34,00 
jun/2021	Parobé	 R$ 36.707,93 	1080	 R$ 34,00 
jul/2021	Caxias do Sul	 R$ 122.228,36 	3595	 R$ 34,00 
jul/2021	São Leopoldo	 R$ 76.488,20 	2250	 R$ 34,00 
jul/2021	Montenegro	 R$ 67.921,59 	1998	 R$ 34,00 
jul/2021	Sapiranga	 R$ 65.643,80 	1931	 R$ 34,00 
jul/2021	Taquara	 R$ 64.681,96 	1902	 R$ 34,00 
jul/2021	Campo Bom	 R$ 48.317,99 	1421	 R$ 34,00 
ago/2021	Caxias do Sul	 R$ 113.865,70 	3349	 R$ 34,00 
ago/2021	Novo Hamburgo	 R$ 79.826,82 	2348	 R$ 34,00 
ago/2021	Campo Bom	 R$ 51.533,65 	1516	 R$ 34,00 
ago/2021	Parobé	 R$ 41.442,26 	1219	 R$ 34,00 
set/2021	Montenegro	 R$ 68.762,53 	2022	 R$ 34,00 
set/2021	São Leopoldo	 R$ 68.066,58 	2002	 R$ 34,00 
set/2021	Sapiranga	 R$ 60.587,70 	1782	 R$ 34,00 
set/2021	Taquara	 R$ 52.138,94 	1533	 R$ 34,00 
set/2021	Campo Bom	 R$ 48.589,91 	1429	 R$ 34,00 
out/2021	Novo Hamburgo	 R$ 97.430,32 	2866	 R$ 34,00 
out/2021	São Leopoldo	 R$ 86.228,90 	2536	 R$ 34,00 
nov/2021	Novo Hamburgo	 R$ 128.932,78 	3792	 R$ 34,00 
nov/2021	São Leopoldo	 R$ 115.734,97 	3404	 R$ 34,00 
nov/2021	Sapiranga	 R$ 70.634,98 	2077	 R$ 34,00 
nov/2021	Estância Velha	 R$ 59.370,66 	1746	 R$ 34,00 
nov/2021	Taquara	 R$ 49.200,68 	1447	 R$ 34,00 
nov/2021	Nova Hartz	 R$ 30.229,82 	889	 R$ 34,00 
dez/2021	Caxias do Sul	 R$ 129.367,05 	3805	 R$ 34,00 
dez/2021	Novo Hamburgo	 R$ 128.674,31 	3785	 R$ 34,00 
dez/2021	São Leopoldo	 R$ 109.954,22 	3234	 R$ 34,00 
dez/2021	Sapiranga	 R$ 89.459,55 	2631	 R$ 34,00 
dez/2021	Montenegro	 R$ 81.165,45 	2387	 R$ 34,00 
dez/2021	Estância Velha	 R$ 70.612,25 	2077	 R$ 34,00 
dez/2021	Taquara	 R$ 53.985,40 	1588	 R$ 34,00 
dez/2021	Sapucaia do Sul (Vinicius)	 R$ 23.860,12 	702	 R$ 34,00 
ago/2023	Campo Bom	 R$ 48.549,15 	1426	 R$ 34,00 
set/2023	Shopping Total	 R$ 64.131,80 	1895	 R$ 34,05 
out/2023	Sapiranga	 R$ 49.952,92 	1468	 R$ 33,84 
nov/2023	Shopping Total	 R$ 75.637,84 	2208	 R$ 34,03 
nov/2023	Campo Bom	 R$ 38.917,01 	1130	 R$ 34,26 
dez/2023	Shopping Total	 R$ 70.758,83 	2066	 R$ 34,44 
nov/2024	Porto Alegre Zona Norte	 R$ 5.888,00 	173	 R$ 34,25 
dez/2024	Vila Mariana	 R$ 946,95 	28	 R$ 34,03 
ago/2019	Sapiranga	 R$ 142.772,54 	4079	 R$ 33,82 
out/2019	Sapiranga	 R$ 115.841,66 	3310	 R$ 35,00 
nov/2019	Sapiranga	 R$ 115.157,85 	3290	 R$ 35,00 
mar/2020	Sapiranga	 R$ 127.734,57 	3650	 R$ 35,00 
mar/2020	Campo Bom	 R$ 107.065,00 	3059	 R$ 35,00 
mar/2020	Novo Hamburgo	 R$ 57.214,54 	1635	 R$ 35,00 
abr/2020	Sapiranga	 R$ 88.876,10 	2539	 R$ 35,00 
abr/2020	Novo Hamburgo	 R$ 8.185,09 	234	 R$ 35,00 
jun/2020	Sapiranga	 R$ 94.462,46 	2699	 R$ 35,00 
jun/2020	Campo Bom	 R$ 71.073,60 	2031	 R$ 35,00 
jun/2020	Taquara	 R$ 54.698,24 	1563	 R$ 35,00 
jul/2020	Sapiranga	 R$ 121.005,43 	3457	 R$ 35,00 
jul/2020	Novo Hamburgo	 R$ 98.495,06 	2814	 R$ 35,00 
jul/2020	Taquara	 R$ 78.443,90 	2241	 R$ 35,00 
jul/2020	Campo Bom	 R$ 63.444,13 	1813	 R$ 35,00 
ago/2020	Novo Hamburgo	 R$ 117.013,71 	3343	 R$ 35,00 
ago/2020	Campo Bom	 R$ 81.349,83 	2324	 R$ 35,00 
set/2020	Novo Hamburgo	 R$ 126.172,13 	3605	 R$ 35,00 
set/2020	Sapiranga	 R$ 108.969,09 	3113	 R$ 35,00 
set/2020	Taquara	 R$ 76.933,95 	2198	 R$ 35,00 
out/2020	Sapiranga	 R$ 97.860,92 	2796	 R$ 35,00 
out/2020	Montenegro	 R$ 91.010,93 	2600	 R$ 35,00 
out/2020	Taquara	 R$ 64.971,70 	1856	 R$ 35,00 
nov/2020	Novo Hamburgo	 R$ 115.220,85 	3292	 R$ 35,00 
nov/2020	Campo Bom	 R$ 65.783,34 	1880	 R$ 35,00 
dez/2020	Novo Hamburgo	 R$ 125.264,46 	3579	 R$ 35,00 
dez/2020	Montenegro	 R$ 110.912,25 	3169	 R$ 35,00 
dez/2020	Sapiranga	 R$ 96.301,48 	2751	 R$ 35,00 
dez/2020	Taquara	 R$ 83.102,13 	2374	 R$ 35,00 
jan/2021	Novo Hamburgo	 R$ 121.668,76 	3476	 R$ 35,00 
jan/2021	Montenegro	 R$ 103.530,47 	2958	 R$ 35,00 
jan/2021	Taquara	 R$ 75.854,00 	2167	 R$ 35,00 
jan/2021	Caxias do Sul	 R$ 55.712,62 	1592	 R$ 35,00 
fev/2021	Taquara	 R$ 60.524,75 	1729	 R$ 35,00 
fev/2021	Sapiranga	 R$ 56.336,05 	1610	 R$ 35,00 
fev/2021	Campo Bom	 R$ 50.545,59 	1444	 R$ 35,00 
mar/2021	Novo Hamburgo	 R$ 100.453,20 	2870	 R$ 35,00 
mar/2021	Estância Velha	 R$ 60.834,49 	1738	 R$ 35,00 
mar/2021	Taquara	 R$ 57.260,60 	1636	 R$ 35,00 
mar/2021	Sapiranga	 R$ 53.401,10 	1526	 R$ 35,00 
abr/2021	Caxias do Sul	 R$ 80.413,13 	2298	 R$ 35,00 
abr/2021	Montenegro	 R$ 76.649,17 	2190	 R$ 35,00 
abr/2021	Sapiranga	 R$ 50.044,07 	1430	 R$ 35,00 
abr/2021	Taquara	 R$ 48.176,58 	1376	 R$ 35,00 
abr/2021	Campo Bom	 R$ 38.543,65 	1101	 R$ 35,00 
abr/2021	São Leopoldo	 R$ 35.792,76 	1023	 R$ 35,00 
mai/2021	Montenegro	 R$ 70.634,88 	2018	 R$ 35,00 
mai/2021	Taquara	 R$ 47.152,73 	1347	 R$ 35,00 
mai/2021	Parobé	 R$ 43.594,45 	1246	 R$ 35,00 
jun/2021	Caxias do Sul	 R$ 106.367,28 	3039	 R$ 35,00 
jun/2021	Novo Hamburgo	 R$ 95.885,60 	2740	 R$ 35,00 
jun/2021	Estância Velha	 R$ 54.633,36 	1561	 R$ 35,00 
jun/2021	Taquara	 R$ 48.281,96 	1379	 R$ 35,00 
jul/2021	Novo Hamburgo	 R$ 102.538,92 	2930	 R$ 35,00 
jul/2021	Estância Velha	 R$ 61.813,26 	1766	 R$ 35,00 
jul/2021	Parobé	 R$ 50.531,31 	1444	 R$ 35,00 
ago/2021	São Leopoldo	 R$ 75.557,68 	2159	 R$ 35,00 
ago/2021	Montenegro	 R$ 71.824,14 	2052	 R$ 35,00 
ago/2021	Estância Velha	 R$ 66.495,99 	1900	 R$ 35,00 
ago/2021	Sapiranga	 R$ 58.241,67 	1664	 R$ 35,00 
ago/2021	Taquara	 R$ 52.457,07 	1499	 R$ 35,00 
set/2021	Caxias do Sul	 R$ 116.732,15 	3335	 R$ 35,00 
set/2021	Novo Hamburgo	 R$ 84.039,94 	2401	 R$ 35,00 
set/2021	Estância Velha	 R$ 55.907,89 	1597	 R$ 35,00 
set/2021	Parobé	 R$ 37.249,98 	1064	 R$ 35,00 
out/2021	Caxias do Sul	 R$ 125.607,10 	3589	 R$ 35,00 
out/2021	Montenegro	 R$ 76.242,56 	2178	 R$ 35,00 
out/2021	Sapiranga	 R$ 61.765,38 	1765	 R$ 35,00 
out/2021	Estância Velha	 R$ 61.110,82 	1746	 R$ 35,00 
out/2021	Campo Bom	 R$ 56.334,86 	1610	 R$ 35,00 
out/2021	Taquara	 R$ 55.475,49 	1585	 R$ 35,00 
out/2021	Nova Hartz	 R$ 54.020,31 	1543	 R$ 35,00 
out/2021	Parobé	 R$ 38.801,00 	1109	 R$ 35,00 
nov/2021	Caxias do Sul	 R$ 112.935,10 	3227	 R$ 35,00 
nov/2021	Montenegro	 R$ 75.866,31 	2168	 R$ 35,00 
nov/2021	Campo Bom	 R$ 59.256,74 	1693	 R$ 35,00 
nov/2021	Parobé	 R$ 40.369,96 	1153	 R$ 35,00 
dez/2021	Campo Bom	 R$ 76.737,74 	2193	 R$ 35,00 
dez/2021	Parobé	 R$ 44.442,81 	1270	 R$ 35,00 
dez/2021	Nova Hartz	 R$ 27.450,25 	784	 R$ 35,00 
set/2023	Campo Bom	 R$ 53.519,50 	1535	 R$ 35,00 
set/2024	Vila Mariana	 R$ 14.469,71 	408	 R$ 34,87 
out/2024	Vila Mariana	 R$ 12.812,21 	363	 R$ 35,46 
nov/2024	Vila Mariana	 R$ 9.373,41 	271	 R$ 35,30 
nov/2019	Campo Bom	 R$ 96.558,80 	2682	 R$ 34,59 
jan/2020	Campo Bom	 R$ 108.411,90 	2853	 R$ 36,00 
fev/2020	Campo Bom	 R$ 109.765,05 	2889	 R$ 38,00 
jun/2023	Estância Velha	 R$ 37.345,03 	258	 R$ 38,00 
jul/2023	Shopping Total	 R$ 77.424,34 	1454	 R$ 144,75 
jul/2023	Estância Velha	 R$ 35.667,05 	729	 R$ 53,25 
ago/2023	Shopping Total	 R$ 63.398,44 	1949	 R$ 48,93 
ago/2023	Estância Velha	 R$ 36.320,73 	709	 R$ 32,53 
ago/2023	Canoas Mathias	 R$ 23.597,15 	616	 R$ 51,23 
set/2023	Estância Velha	 R$ 31.673,07 	599	 R$ 38,31 
out/2023	Shopping Total	 R$ 67.732,27 	2065	 R$ 52,88 
out/2023	Campo Bom	 R$ 49.065,70 	1312	 R$ 32,80 
out/2023	Canoas Mathias	 R$ 31.515,35 	853	 R$ 37,40 
out/2023	Estância Velha	 R$ 29.393,68 	523	 R$ 36,95 
nov/2023	Canoas Mathias	 R$ 32.376,32 	892	 R$ 56,20 
nov/2023	Estância Velha	 R$ 27.534,78 	502	 R$ 36,30 
dez/2023	Estância Velha	 R$ 30.766,13 	573	 R$ 54,85 
jan/2024	Shopping Total	 R$ 64.336,66 	3790	 R$ 53,69 
jan/2024	Campo Bom	 R$ 46.038,12 	1209	 R$ 16,98 
jan/2024	Estância Velha	 R$ 28.495,57 	546	 R$ 38,08 
fev/2024	Estância Velha	 R$ 27.780,44 	1643	 R$ 52,19 
fev/2024	Canoas Mathias	 R$ 25.615,22 	629	 R$ 16,91 
mar/2024	Campo Bom	 R$ 44.037,47 	1004	 R$ 40,72 
mar/2024	Canoas Mathias	 R$ 30.717,53 	720	 R$ 43,86 
mar/2024	Estância Velha	 R$ 29.860,34 	589	 R$ 42,66 
abr/2024	Canoas Mathias	 R$ 29.395,57 	666	 R$ 50,70 
abr/2024	Estância Velha	 R$ 26.239,64 	518	 R$ 44,14 
mai/2024	Campo Bom	 R$ 48.029,05 	1107	 R$ 50,66 
mai/2024	Estância Velha	 R$ 22.452,50 	369	 R$ 43,39 
mai/2024	Canoas Mathias	 R$ 2.991,05 	68	 R$ 60,85 
jun/2024	Campo Bom	 R$ 49.242,99 	1118	 R$ 43,99 
jun/2024	Estância Velha	 R$ 15.281,83 	256	 R$ 44,05 
jul/2024	Campo Bom	 R$ 43.562,42 	1072	 R$ 59,69 
jul/2024	Vila Mariana	 R$ 7.986,61 	205	 R$ 40,64 
ago/2024	Campo Bom	 R$ 46.952,66 	1149	 R$ 38,96 
ago/2024	Vila Mariana	 R$ 15.800,20 	516	 R$ 40,86 
out/2024	Campo Bom	 R$ 28.810,57 	680	 R$ 30,62 
jan/2020	Sapiranga	 R$ 127.060,67 	3344	 R$ 42,37 
jan/2020	Novo Hamburgo	 R$ 89.000,40 	2342	 R$ 38,00 
fev/2020	Novo Hamburgo	 R$ 85.490,36 	2250	 R$ 38,00 
mai/2023	Montenegro	 R$ 70.944,42 	444	 R$ 38,00 
jun/2023	Montenegro	 R$ 65.804,52 	1342	 R$ 159,78 
jun/2023	Canoas	 R$ 45.835,29 	436	 R$ 49,03 
jun/2023	Zona Norte	 R$ 608,49 	16	 R$ 105,13 
jul/2023	Barra Shopping	 R$ 114.585,33 	2678	 R$ 38,03 
jul/2023	Novo Hamburgo	 R$ 110.336,19 	1398	 R$ 42,79 
jul/2023	São Leopoldo	 R$ 94.551,10 	1258	 R$ 78,92 
jul/2023	Montenegro	 R$ 73.669,81 	1475	 R$ 75,16 
jul/2023	Sapiranga	 R$ 55.605,78 	685	 R$ 49,95 
jul/2023	Canoas	 R$ 46.610,68 	1140	 R$ 81,18 
jul/2023	Zona Norte	 R$ 8.096,47 	215	 R$ 40,89 
ago/2023	Caxias do Sul	 R$ 124.636,95 	2034	 R$ 37,66 
ago/2023	Novo Hamburgo	 R$ 112.562,21 	2298	 R$ 61,28 
ago/2023	Barra Shopping	 R$ 90.829,42 	3108	 R$ 48,98 
ago/2023	São Leopoldo	 R$ 83.582,26 	1763	 R$ 29,22 
ago/2023	Montenegro	 R$ 64.729,04 	1367	 R$ 47,41 
ago/2023	Canoas	 R$ 48.631,22 	1269	 R$ 47,35 
ago/2023	Bento Gonçalves	 R$ 48.074,80 	669	 R$ 38,32 
ago/2023	Sapiranga	 R$ 44.810,23 	1239	 R$ 71,86 
ago/2023	Gravataí	 R$ 15.552,29 	335	 R$ 36,17 
ago/2023	Zona Norte	 R$ 8.596,64 	196	 R$ 46,42 
set/2023	Caxias do Sul	 R$ 139.221,51 	2736	 R$ 43,86 
set/2023	Novo Hamburgo	 R$ 110.366,61 	2171	 R$ 50,89 
set/2023	Barra Shopping	 R$ 99.475,46 	3345	 R$ 50,84 
set/2023	São Leopoldo	 R$ 81.491,40 	1693	 R$ 29,74 
set/2023	Montenegro	 R$ 65.679,95 	1342	 R$ 48,13 
set/2023	Sapiranga	 R$ 54.991,94 	1529	 R$ 48,94 
set/2023	Canoas	 R$ 54.944,45 	1445	 R$ 35,97 
set/2023	Bento Gonçalves	 R$ 51.416,87 	1084	 R$ 38,02 
set/2023	Gravataí	 R$ 21.606,87 	419	 R$ 47,43 
set/2023	Zona Norte	 R$ 4.223,71 	89	 R$ 51,57 
out/2023	Caxias do Sul	 R$ 122.690,30 	2451	 R$ 47,46 
out/2023	Barra Shopping	 R$ 114.033,42 	3795	 R$ 50,06 
out/2023	Novo Hamburgo	 R$ 95.946,27 	1955	 R$ 30,05 
out/2023	São Leopoldo	 R$ 77.013,91 	1666	 R$ 49,08 
out/2023	Montenegro	 R$ 68.491,63 	1432	 R$ 46,23 
out/2023	Canoas	 R$ 51.139,92 	1364	 R$ 47,83 
out/2023	Bento Gonçalves	 R$ 50.399,27 	1050	 R$ 37,49 
out/2023	Gravataí	 R$ 21.609,45 	447	 R$ 48,00 
out/2023	Lajeado	 R$ 15.705,14 	326	 R$ 48,34 
out/2023	Zona Norte	 R$ 5.836,88 	118	 R$ 48,18 
nov/2023	Caxias do Sul	 R$ 135.853,22 	2611	 R$ 49,47 
nov/2023	Barra Shopping	 R$ 133.914,34 	4301	 R$ 52,03 
nov/2023	Novo Hamburgo	 R$ 107.238,45 	2139	 R$ 31,14 
nov/2023	São Leopoldo	 R$ 84.079,28 	1768	 R$ 50,13 
nov/2023	Lajeado	 R$ 78.457,26 	1565	 R$ 47,56 
nov/2023	Bento Gonçalves	 R$ 66.524,61 	1330	 R$ 50,13 
nov/2023	Montenegro	 R$ 62.276,82 	1267	 R$ 50,02 
nov/2023	Canoas	 R$ 53.223,40 	1459	 R$ 49,15 
nov/2023	Sapiranga	 R$ 49.651,62 	1393	 R$ 36,48 
nov/2023	Gravataí	 R$ 23.741,80 	508	 R$ 35,64 
nov/2023	Zona Norte	 R$ 5.873,75 	137	 R$ 46,74 
dez/2023	Barra Shopping	 R$ 166.546,26 	5379	 R$ 42,87 
dez/2023	Caxias do Sul	 R$ 151.384,68 	2815	 R$ 30,96 
dez/2023	Novo Hamburgo	 R$ 107.718,37 	2150	 R$ 53,78 
dez/2023	São Leopoldo	 R$ 88.884,29 	1774	 R$ 50,10 
dez/2023	Montenegro	 R$ 75.822,47 	1454	 R$ 50,10 
dez/2023	Lajeado	 R$ 74.607,39 	1502	 R$ 52,15 
dez/2023	Sapiranga	 R$ 55.652,88 	1478	 R$ 49,67 
dez/2023	Bento Gonçalves	 R$ 55.274,92 	1062	 R$ 37,65 
dez/2023	Canoas	 R$ 49.474,99 	1286	 R$ 52,05 
dez/2023	Gravataí	 R$ 36.071,53 	814	 R$ 38,47 
dez/2023	Zona Norte	 R$ 6.460,09 	134	 R$ 44,31 
jan/2024	Caxias do Sul	 R$ 130.916,46 	2382	 R$ 48,21 
jan/2024	Barra Shopping	 R$ 111.287,82 	3790	 R$ 54,96 
jan/2024	Novo Hamburgo	 R$ 98.238,38 	2047	 R$ 29,36 
jan/2024	São Leopoldo	 R$ 80.734,30 	1608	 R$ 47,99 
jan/2024	Erechim	 R$ 76.932,77 	1738	 R$ 50,21 
jan/2024	Lajeado	 R$ 62.465,62 	1217	 R$ 44,27 
jan/2024	Montenegro	 R$ 61.370,75 	1238	 R$ 51,33 
jan/2024	Bento Gonçalves	 R$ 50.082,27 	971	 R$ 49,57 
jan/2024	Canoas	 R$ 41.690,23 	988	 R$ 51,58 
jan/2024	Gravataí	 R$ 39.179,55 	883	 R$ 42,20 
jan/2024	Zona Norte	 R$ 6.870,47 	165	 R$ 44,37 
fev/2024	Caxias do Sul	 R$ 122.303,16 	2357	 R$ 41,64 
fev/2024	Novo Hamburgo	 R$ 97.897,72 	1961	 R$ 51,89 
fev/2024	Barra Shopping	 R$ 97.247,44 	3139	 R$ 49,92 
fev/2024	São Leopoldo	 R$ 93.863,35 	1933	 R$ 30,98 
fev/2024	Erechim	 R$ 85.055,03 	2035	 R$ 48,56 
fev/2024	Floresta	 R$ 70.084,49 	1643	 R$ 41,80 
fev/2024	Montenegro	 R$ 67.994,34 	1315	 R$ 42,66 
fev/2024	Lajeado	 R$ 50.775,84 	1011	 R$ 51,71 
fev/2024	Sapiranga	 R$ 49.982,44 	1276	 R$ 50,22 
fev/2024	Bento Gonçalves	 R$ 43.437,99 	872	 R$ 39,17 
fev/2024	Gravataí	 R$ 39.577,53 	953	 R$ 49,81 
fev/2024	Canoas	 R$ 35.894,77 	829	 R$ 41,53 
fev/2024	Zona Norte	 R$ 4.439,50 	84	 R$ 43,30 
mar/2024	Caxias do Sul	 R$ 131.901,42 	2510	 R$ 52,85 
mar/2024	Novo Hamburgo	 R$ 105.723,57 	2132	 R$ 52,55 
mar/2024	São Leopoldo	 R$ 103.647,37 	2056	 R$ 49,59 
mar/2024	Barra Shopping	 R$ 92.633,96 	3054	 R$ 50,41 
mar/2024	Erechim	 R$ 84.977,64 	1994	 R$ 30,33 
mar/2024	Montenegro	 R$ 71.080,12 	1385	 R$ 42,62 
mar/2024	Floresta	 R$ 68.845,12 	1630	 R$ 51,32 
mar/2024	Sapiranga	 R$ 53.791,95 	1381	 R$ 42,24 
mar/2024	Lajeado	 R$ 50.756,71 	1065	 R$ 38,95 
mar/2024	Bento Gonçalves	 R$ 49.135,17 	1004	 R$ 47,66 
mar/2024	Protásio Alves	 R$ 46.175,28 	1187	 R$ 48,94 
mar/2024	Canoas	 R$ 38.987,21 	787	 R$ 38,90 
mar/2024	Gravataí	 R$ 38.834,88 	823	 R$ 49,54 
mar/2024	Zona Norte	 R$ 5.024,02 	91	 R$ 47,19 
abr/2024	Caxias do Sul	 R$ 135.423,32 	2469	 R$ 55,21 
abr/2024	Novo Hamburgo	 R$ 105.653,58 	2167	 R$ 54,85 
abr/2024	São Leopoldo	 R$ 99.805,12 	2030	 R$ 48,76 
abr/2024	Barra Shopping	 R$ 83.207,62 	2689	 R$ 49,17 
abr/2024	Erechim	 R$ 77.069,92 	2028	 R$ 30,94 
abr/2024	Montenegro	 R$ 76.189,50 	1533	 R$ 38,00 
abr/2024	Protásio Alves	 R$ 67.765,41 	1749	 R$ 49,70 
abr/2024	Floresta	 R$ 56.965,43 	1450	 R$ 38,75 
abr/2024	Sapiranga	 R$ 52.642,68 	1406	 R$ 39,29 
abr/2024	Esteio	 R$ 52.415,87 	1089	 R$ 37,44 
abr/2024	Bento Gonçalves	 R$ 50.641,44 	1038	 R$ 48,13 
abr/2024	Lajeado	 R$ 46.826,66 	1042	 R$ 48,79 
abr/2024	Gravataí	 R$ 37.520,14 	831	 R$ 44,94 
abr/2024	Canoas	 R$ 35.754,96 	785	 R$ 45,15 
abr/2024	Zona Norte	 R$ 6.791,51 	125	 R$ 45,55 
mai/2024	Caxias do Sul	 R$ 141.678,08 	2417	 R$ 54,33 
mai/2024	Esteio	 R$ 133.434,48 	2601	 R$ 58,62 
mai/2024	Novo Hamburgo	 R$ 127.226,59 	2414	 R$ 51,30 
mai/2024	Protásio Alves	 R$ 101.886,36 	2162	 R$ 52,70 
mai/2024	São Leopoldo	 R$ 87.981,38 	1522	 R$ 47,13 
mai/2024	Sapiranga	 R$ 66.713,12 	1538	 R$ 57,81 
mai/2024	Barra Shopping	 R$ 57.776,07 	1929	 R$ 43,38 
mai/2024	Gravataí	 R$ 55.371,16 	1045	 R$ 29,95 
mai/2024	Lajeado	 R$ 49.855,99 	998	 R$ 52,99 
mai/2024	Bento Gonçalves	 R$ 49.377,77 	907	 R$ 49,96 
mai/2024	Montenegro	 R$ 48.969,65 	851	 R$ 54,44 
mai/2024	Floresta	 R$ 28.198,56 	574	 R$ 57,54 
mai/2024	Canoas	 R$ 25.318,02 	472	 R$ 49,13 
mai/2024	Zona Norte	 R$ 10.180,14 	193	 R$ 53,64 
jun/2024	Caxias do Sul	 R$ 151.206,45 	2542	 R$ 52,75 
jun/2024	Novo Hamburgo	 R$ 129.569,54 	2474	 R$ 59,48 
jun/2024	São Leopoldo	 R$ 126.959,93 	2237	 R$ 52,37 
jun/2024	Esteio	 R$ 120.763,77 	2487	 R$ 56,75 
jun/2024	Barra Shopping	 R$ 95.286,68 	3165	 R$ 48,56 
jun/2024	Erechim	 R$ 83.400,12 	2059	 R$ 30,11 
jun/2024	Protásio Alves	 R$ 81.970,38 	1921	 R$ 40,51 
jun/2024	Montenegro	 R$ 80.544,67 	1506	 R$ 42,67 
jun/2024	Canoas	 R$ 74.125,88 	1273	 R$ 53,48 
jun/2024	Lajeado	 R$ 65.799,34 	1333	 R$ 58,23 
jun/2024	Sapiranga	 R$ 61.943,97 	1426	 R$ 49,36 
jun/2024	Floresta	 R$ 59.044,90 	1309	 R$ 43,44 
jun/2024	Gravataí	 R$ 48.048,68 	954	 R$ 45,11 
jun/2024	Bento Gonçalves	 R$ 45.262,44 	846	 R$ 50,37 
jun/2024	Zona Norte	 R$ 5.165,04 	145	 R$ 53,50 
jul/2024	Caxias do Sul	 R$ 161.427,36 	2714	 R$ 35,62 
jul/2024	Novo Hamburgo	 R$ 124.444,11 	2551	 R$ 59,48 
jul/2024	Esteio	 R$ 114.141,93 	2357	 R$ 48,78 
jul/2024	São Leopoldo	 R$ 108.337,81 	1905	 R$ 48,43 
jul/2024	Barra Shopping	 R$ 92.663,15 	3124	 R$ 56,87 
jul/2024	Protásio Alves	 R$ 77.444,15 	1744	 R$ 29,66 
jul/2024	Montenegro	 R$ 77.268,70 	1449	 R$ 44,41 
jul/2024	Erechim	 R$ 76.299,95 	1835	 R$ 53,33 
jul/2024	Canoas	 R$ 70.092,97 	1286	 R$ 41,58 
jul/2024	Sapiranga	 R$ 64.509,84 	1403	 R$ 54,50 
jul/2024	Lajeado	 R$ 61.190,87 	1233	 R$ 45,98 
jul/2024	Floresta	 R$ 48.956,25 	1132	 R$ 49,63 
jul/2024	Gravataí	 R$ 42.791,32 	866	 R$ 43,25 
jul/2024	Bento Gonçalves	 R$ 40.737,98 	749	 R$ 49,41 
ago/2024	Caxias do Sul	 R$ 140.785,42 	2417	 R$ 54,39 
ago/2024	Novo Hamburgo	 R$ 128.775,61 	2609	 R$ 58,25 
ago/2024	Canoas	 R$ 111.011,97 	2108	 R$ 49,36 
ago/2024	Esteio	 R$ 107.613,59 	2239	 R$ 52,66 
ago/2024	São Leopoldo	 R$ 102.219,21 	1854	 R$ 48,06 
ago/2024	Erechim	 R$ 91.070,88 	2153	 R$ 55,13 
ago/2024	Barra Shopping	 R$ 79.503,36 	2669	 R$ 42,30 
ago/2024	Sapiranga	 R$ 76.528,11 	1799	 R$ 29,79 
ago/2024	Montenegro	 R$ 74.545,34 	1421	 R$ 42,54 
ago/2024	Protásio Alves	 R$ 74.255,98 	1728	 R$ 52,46 
ago/2024	Lajeado	 R$ 47.945,86 	998	 R$ 42,97 
ago/2024	Floresta	 R$ 47.647,18 	1151	 R$ 48,04 
ago/2024	Bento Gonçalves	 R$ 42.220,78 	799	 R$ 41,40 
ago/2024	Gravataí	 R$ 41.265,69 	850	 R$ 52,84 
ago/2024	Zona Norte	 R$ 5.052,35 	131	 R$ 48,55 
set/2024	Caxias do Sul	 R$ 141.996,81 	2502	 R$ 38,57 
set/2024	Novo Hamburgo	 R$ 107.210,77 	2301	 R$ 56,75 
set/2024	Esteio	 R$ 105.540,66 	2255	 R$ 46,59 
set/2024	Canoas	 R$ 95.268,57 	1780	 R$ 46,80 
set/2024	São Leopoldo	 R$ 92.945,77 	1763	 R$ 53,52 
set/2024	Barra Shopping	 R$ 77.426,87 	2620	 R$ 52,72 
set/2024	Sapiranga	 R$ 71.939,29 	1745	 R$ 29,55 
set/2024	Montenegro	 R$ 71.189,06 	1426	 R$ 41,23 
set/2024	Protásio Alves	 R$ 60.922,70 	1405	 R$ 49,92 
set/2024	Lajeado	 R$ 49.130,61 	1095	 R$ 43,36 
set/2024	Gravataí	 R$ 44.314,30 	1007	 R$ 44,87 
set/2024	Bento Gonçalves	 R$ 38.636,81 	729	 R$ 44,01 
set/2024	Floresta	 R$ 37.358,14 	974	 R$ 53,00 
set/2024	Zona Norte	 R$ 5.196,97 	136	 R$ 38,36 
out/2024	Caxias do Sul	 R$ 142.674,81 	2565	 R$ 38,21 
out/2024	Novo Hamburgo	 R$ 108.829,32 	2345	 R$ 55,62 
out/2024	Esteio	 R$ 107.012,63 	2331	 R$ 46,41 
out/2024	São Leopoldo	 R$ 97.405,54 	1961	 R$ 45,91 
out/2024	Erechim	 R$ 96.549,47 	2379	 R$ 49,67 
out/2024	Canoas	 R$ 90.896,16 	1716	 R$ 40,58 
out/2024	Barra Shopping	 R$ 88.359,70 	2863	 R$ 52,97 
out/2024	Montenegro	 R$ 77.197,76 	1622	 R$ 30,86 
out/2024	Protásio Alves	 R$ 65.273,52 	1483	 R$ 47,59 
out/2024	Gravataí	 R$ 55.357,33 	1226	 R$ 44,01 
out/2024	Lajeado	 R$ 52.914,67 	1143	 R$ 45,15 
out/2024	Bento Gonçalves	 R$ 43.387,18 	826	 R$ 46,29 
out/2024	Floresta	 R$ 39.144,54 	1012	 R$ 52,53 
out/2024	Zona Norte	 R$ 5.209,72 	166	 R$ 38,68 
nov/2024	Caxias do Sul	 R$ 140.032,72 	2447	 R$ 31,38 
nov/2024	Esteio	 R$ 114.211,14 	2419	 R$ 57,23 
nov/2024	São Leopoldo	 R$ 103.129,69 	2067	 R$ 47,21 
nov/2024	Novo Hamburgo	 R$ 102.653,05 	1879	 R$ 49,89 
nov/2024	Barra Shopping	 R$ 100.780,77 	3236	 R$ 54,63 
nov/2024	Canoas	 R$ 90.609,35 	1694	 R$ 31,14 
nov/2024	Erechim	 R$ 90.221,21 	2166	 R$ 53,49 
nov/2024	Montenegro	 R$ 85.892,07 	1823	 R$ 41,65 
nov/2024	Sapiranga	 R$ 69.224,49 	1513	 R$ 47,12 
nov/2024	Protásio Alves	 R$ 65.867,97 	1468	 R$ 45,75 
nov/2024	Gravataí	 R$ 57.332,13 	1278	 R$ 44,87 
nov/2024	Capão da Canoa	 R$ 56.077,36 	1248	 R$ 44,86 
nov/2024	Lajeado	 R$ 55.496,73 	1169	 R$ 44,93 
nov/2024	Bento Gonçalves	 R$ 46.826,86 	866	 R$ 47,47 
nov/2024	Floresta	 R$ 33.171,46 	846	 R$ 54,07 
dez/2024	Caxias do Sul	 R$ 150.270,33 	2582	 R$ 39,21 
dez/2024	Barra Shopping	 R$ 131.492,62 	4056	 R$ 58,20 
dez/2024	Novo Hamburgo	 R$ 105.258,11 	2007	 R$ 32,42 
dez/2024	São Leopoldo	 R$ 104.515,80 	2029	 R$ 52,45 
dez/2024	Esteio	 R$ 102.495,07 	2285	 R$ 51,51 
dez/2024	Montenegro	 R$ 99.601,78 	2124	 R$ 44,86 
dez/2024	Erechim	 R$ 93.095,12 	2103	 R$ 46,89 
dez/2024	Sapiranga	 R$ 86.526,51 	1915	 R$ 44,27 
dez/2024	Canoas	 R$ 85.316,04 	1575	 R$ 45,18 
dez/2024	Capão da Canoa	 R$ 68.869,44 	1612	 R$ 54,17 
dez/2024	Protásio Alves	 R$ 64.357,02 	1460	 R$ 42,72 
dez/2024	Gravataí	 R$ 63.225,89 	1430	 R$ 44,08 
dez/2024	Lajeado	 R$ 61.493,46 	1296	 R$ 44,21 
dez/2024	Bento Gonçalves	 R$ 46.900,84 	879	 R$ 47,45 
dez/2024	Porto Alegre Zona Norte	 R$ 9.503,37 	255	 R$ 53,36 
jan/2025	Caxias do Sul	 R$ 127.978,76 	2247	 R$ 37,27 
jan/2025	Capão da Canoa	 R$ 99.116,50 	2099	 R$ 56,96 
jan/2025	Novo Hamburgo	 R$ 93.751,71 	1822	 R$ 47,22 
jan/2025	São Leopoldo	 R$ 91.235,01 	1873	 R$ 51,46 
jan/2025	Barra Shopping	 R$ 90.771,49 	2991	 R$ 48,71 
jan/2025	Montenegro	 R$ 89.321,75 	1857	 R$ 30,35 
jan/2025	Esteio	 R$ 87.167,15 	1991	 R$ 48,10 
jan/2025	Canoas	 R$ 78.297,41 	1472	 R$ 43,78 
jan/2025	Sapiranga	 R$ 77.219,21 	1630	 R$ 53,19 
jan/2025	Erechim	 R$ 70.360,63 	1640	 R$ 47,37 
jan/2025	Gravataí	 R$ 59.346,07 	1321	 R$ 42,90 
jan/2025	Protásio Alves	 R$ 54.874,94 	1186	 R$ 44,93 
jan/2025	Lajeado	 R$ 53.103,51 	1166	 R$ 46,27 
jan/2025	Bento Gonçalves	 R$ 45.230,74 	928	 R$ 45,54 
jan/2025	Floresta	 R$ 31.258,45 	761	 R$ 48,74 
jan/2025	Porto Alegre Zona Norte	 R$ 12.467,50 	323	 R$ 41,08 
fev/2025	Caxias do Sul	 R$ 133.607,09 	2484	 R$ 38,60 
fev/2025	São Leopoldo	 R$ 88.246,18 	1737	 R$ 53,79 
fev/2025	Montenegro	 R$ 87.375,84 	1799	 R$ 50,80 
fev/2025	Novo Hamburgo	 R$ 86.234,26 	1651	 R$ 48,57 
fev/2025	Esteio	 R$ 84.749,04 	1973	 R$ 52,23 
fev/2025	Erechim	 R$ 84.729,98 	1998	 R$ 42,95 
fev/2025	Barra Shopping	 R$ 79.973,70 	2599	 R$ 42,41 
fev/2025	Sapiranga	 R$ 75.661,57 	1688	 R$ 30,77 
fev/2025	Canoas	 R$ 75.361,19 	1333	 R$ 44,82 
fev/2025	Capão da Canoa	 R$ 72.624,66 	1478	 R$ 56,54 
fev/2025	Protásio Alves	 R$ 62.799,38 	1393	 R$ 49,14 
fev/2025	Gravataí	 R$ 53.164,95 	1154	 R$ 45,08 
fev/2025	Lajeado	 R$ 52.071,35 	1164	 R$ 46,07 
fev/2025	Bento Gonçalves	 R$ 50.724,83 	1049	 R$ 44,73 
fev/2025	Floresta	 R$ 48.317,12 	1178	 R$ 48,36 
fev/2025	Porto Alegre Zona Norte	 R$ 8.781,87 	226	 R$ 41,02 
mar/2025	Caxias do Sul	 R$ 155.399,00 	2905	 R$ 38,86 
mar/2025	Esteio	 R$ 110.394,00 	2504	 R$ 53,49 
mar/2025	Novo Hamburgo	 R$ 101.018,00 	1890	 R$ 44,09 
mar/2025	São Leopoldo	 R$ 98.828,00 	1922	 R$ 53,45 
mar/2025	Montenegro	 R$ 97.872,00 	2100	 R$ 51,42 
mar/2025	Canoas	 R$ 95.864,00 	1706	 R$ 46,61 
mar/2025	Barra Shopping	 R$ 92.567,00 	3037	 R$ 56,19 
mar/2025	Erechim	 R$ 89.672,00 	2154	 R$ 30,48 
mar/2025	Protásio Alves	 R$ 86.428,00 	1854	 R$ 41,63 
mar/2025	Sapiranga	 R$ 86.119,00 	2058	 R$ 46,62 
mar/2025	Capão da Canoa	 R$ 75.427,00 	1769	 R$ 41,85 
mar/2025	Gravataí	 R$ 60.026,00 	1288	 R$ 42,64 
mar/2025	Bento Gonçalves	 R$ 57.857,00 	1119	 R$ 46,60 
mar/2025	Lajeado	 R$ 57.217,00 	1294	 R$ 51,70 
abr/2025	Caxias do Sul	 R$ 142.137,88 	2505	 R$ 44,22 
abr/2025	Esteio	 R$ 99.506,82 	2105	 R$ 56,74 
abr/2025	Novo Hamburgo	 R$ 98.690,79 	1778	 R$ 47,27 
abr/2025	São Leopoldo	 R$ 92.933,28 	1806	 R$ 55,51 
abr/2025	Barra Shopping	 R$ 91.550,13 	2807	 R$ 51,46 
abr/2025	Protásio Alves	 R$ 86.688,06 	1929	 R$ 32,61 
abr/2025	Montenegro	 R$ 86.650,58 	1685	 R$ 44,94 
abr/2025	Canoas	 R$ 76.154,00 	1455	 R$ 51,42 
abr/2025	Erechim	 R$ 75.014,20 	1718	 R$ 52,34 
abr/2025	Sapiranga	 R$ 73.756,62 	1636	 R$ 43,66 
abr/2025	Ijuí	 R$ 72.411,97 	1394	 R$ 45,08 
abr/2025	Lajeado	 R$ 53.282,86 	1111	 R$ 51,95 
abr/2025	Bento Gonçalves	 R$ 50.394,92 	1014	 R$ 47,96 
abr/2025	Gravataí	 R$ 47.434,20 	954	 R$ 49,70 
abr/2025	Capão da Canoa	 R$ 45.931,72 	1000	 R$ 49,72 
abr/2025	Porto Alegre Zona Norte	 R$ 6.793,61 	154	 R$ 45,93 
mai/2025	Montenegro	 R$ 94.364,55 	1930	 R$ 44,11 
mai/2025	Canoas	 R$ 86.708,46 	1679	 R$ 48,89 
mai/2025	Porto Alegre Zona Norte	 R$ 9.487,58 	206	 R$ 51,64 
mai/2025	São Leopoldo	 R$ 97.252,02 	1858	 R$ 46,06 
mai/2025	Novo Hamburgo	 R$ 105.388,86 	2012	 R$ 52,34 
mai/2025	Barra Shopping	 R$ 90.232,32 	2781	 R$ 52,38 
mai/2025	Floresta	 R$ 58.451,14 	1352	 R$ 32,45 
mai/2025	Sapiranga	 R$ 86.357,48 	1887	 R$ 43,23 
mai/2025	Gravataí	 R$ 51.597,09 	1065	 R$ 45,76 
mai/2025	Caxias do Sul	 R$ 162.730,31 	2955	 R$ 48,45 
mai/2025	Bento Gonçalves	 R$ 60.377,65 	1192	 R$ 55,07 
mai/2025	Lajeado	 R$ 46.662,75 	937	 R$ 50,65 
mai/2025	Erechim	 R$ 78.169,32 	1778	 R$ 49,80 
mai/2025	Protásio Alves	 R$ 98.469,56 	2112	 R$ 43,96 
mai/2025	Esteio	 R$ 99.546,11 	2145	 R$ 46,62 
mai/2025	Capão da Canoa	 R$ 52.971,27 	1158	 R$ 46,41 
mai/2025	Ijuí	 R$ 145.404,04 	2830	 R$ 45,74 
jun/2025	Montenegro	 R$ 87.082,52 	1671	 R$ 51,38 
jun/2025	Canoas	 R$ 94.742,73 	1908	 R$ 52,11 
jun/2025	Porto Alegre Zona Norte	 R$ 10.708,26 	226	 R$ 49,66 
jun/2025	São Leopoldo	 R$ 98.705,95 	1870	 R$ 47,38 
jun/2025	Novo Hamburgo	 R$ 100.887,64 	1862	 R$ 52,78 
jun/2025	Barra Shopping	 R$ 78.704,47 	2436	 R$ 54,18 
jun/2025	Porto Alegre Floresta	 R$ 61.375,81 	1412	 R$ 32,31 
jun/2025	Sapiranga	 R$ 90.023,99 	1887	 R$ 43,47 
jun/2025	Gravataí	 R$ 52.623,54 	1079	 R$ 47,71 
jun/2025	Caxias do Sul	 R$ 171.560,83 	3166	 R$ 48,77 
jun/2025	Bento Gonçalves	 R$ 63.012,98 	1211	 R$ 54,19 
jun/2025	Lajeado	 R$ 49.109,65 	971	 R$ 52,03 
jun/2025	Erechim	 R$ 63.328,41 	1396	 R$ 50,58 
jun/2025	Protásio Alves	 R$ 98.645,72 	2073	 R$ 45,36 
jun/2025	Esteio	 R$ 93.758,11 	2031	 R$ 47,59 
jun/2025	Capão da Canoa	 R$ 46.436,04 	1026	 R$ 46,16 
jun/2025	Ijuí	 R$ 94.006,48 	1721	 R$ 45,26 
jul/2025	Montenegro	 R$ 85.435,00 	1690	 R$ 54,62 
jul/2025	Canoas	 R$ 87.936,00 	1762	 R$ 50,55 
jul/2025	Porto Alegre Zona Norte	 R$ 9.833,00 	214	 R$ 49,91 
jul/2025	São Leopoldo	 R$ 89.454,00 	1738	 R$ 45,95 
jul/2025	Novo Hamburgo	 R$ 103.905,00 	1940	 R$ 51,47 
jul/2025	Barra Shopping	 R$ 92.303,00 	2807	 R$ 53,56 
jul/2025	Porto Alegre Floresta	 R$ 62.066,00 	1390	 R$ 32,88 
jul/2025	Sapiranga	 R$ 83.595,00 	1786	 R$ 44,65 
jul/2025	Gravataí	 R$ 46.073,00 	930	 R$ 46,81 
jul/2025	Caxias do Sul	 R$ 150.157,00 	2532	 R$ 49,54 
jul/2025	Bento Gonçalves	 R$ 68.375,00 	1279	 R$ 59,30 
jul/2025	Lajeado	 R$ 46.170,00 	921	 R$ 53,46 
jul/2025	Erechim	 R$ 58.233,00 	1334	 R$ 50,13 
jul/2025	Protásio Alves	 R$ 93.246,00 	2025	 R$ 43,65 
jul/2025	Esteio	 R$ 92.955,00 	1954	 R$ 46,05 
jul/2025	Capão da Canoa	 R$ 43.761,00 	953	 R$ 47,57 
jul/2025	Ijuí	 R$ 84.184,00 	1505	 R$ 45,92 
"""

# ========================
# Funções utilitárias
# ========================

def parse_number_br(value: str) -> float:
    """
    Converte uma string no formato brasileiro (com separador de milhar
    representado por ponto e separador decimal por vírgula) em float.

    Parâmetros
    ----------
    value : str
        String com o número a ser convertido. Se `None` ou vazio, retorna
        `np.nan`.

    Retorna
    -------
    float
        Valor numérico equivalente ou NaN em caso de falha.
    """
    if value is None:
        return np.nan
    s = str(value)
    # Remover o símbolo de moeda e espaços
    s = s.replace("R$", "").replace(" ", "").replace("\u00a0", "")
    # Substituir milhar e decimal
    s = s.replace(".", "").replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return np.nan


def brl(x: float) -> str:
    """
    Formata um número para o padrão monetário brasileiro com R$ e duas
    casas decimais.
    """
    if pd.isna(x):
        return ""
    s = f"{x:,.2f}"
    return "R$ " + s.replace(",", "X").replace(".", ",").replace("X", ".")


MES_MAP = {
    "jan": 1, "fev": 2, "mar": 3, "abr": 4, "mai": 5, "jun": 6,
    "jul": 7, "ago": 8, "set": 9, "out": 10, "nov": 11, "dez": 12
}
MES_NOME = {v: k for k, v in MES_MAP.items()}


ROW_REGEX = re.compile(
    r"^([a-z]{3}/\d{4})\s+(.+?)\s+R\$\s*([\d\.\,]+)\s+(\d+)\s+R\$\s*([\d\.\,]+)",
    flags=re.IGNORECASE
)


@st.cache_data(show_spinner=False)
def load_data(raw_text: str = RAW_DATA) -> pd.DataFrame:
    """
    Carrega os dados a partir do texto bruto `raw_text` e retorna um DataFrame
    limpo e enriquecido.

    O resultado é cacheado para evitar recálculo na interação com os filtros.
    """
    rows = []
    for raw in raw_text.splitlines():
        line = raw.strip()
        if not line or line.lower().startswith("data"):
            continue
        m = ROW_REGEX.match(line)
        if not m:
            # separar com regex se houver tabulações inconsistentes
            parts = re.split(r"\t+", line)
            if len(parts) >= 5:
                data, loja, fat, ped, tick = parts[:5]
                m = ROW_REGEX.match(f"{data} {loja} R$ {fat} {ped} R$ {tick}")
        if m:
            data, loja, fat, ped, tick = m.groups()
            fat = parse_number_br(fat)
            ped = int(ped)
            tick = parse_number_br(tick)
            mes_abrev, ano = data.split("/")
            mes = MES_MAP.get(mes_abrev.lower(), None)
            periodo = None
            if mes is not None:
                try:
                    periodo = datetime(int(ano), mes, 1)
                except Exception:
                    periodo = None
            rows.append({
                "Data": data,
                "Ano": int(ano),
                "Mes": mes,
                "Loja": loja.strip(),
                "Faturamento": fat,
                "Pedidos": ped,
                "TicketMedioDeclarado": tick,
                "Periodo": periodo
            })
        else:
            # Linha não reconhecida — ignorar (poderíamos logar para depuração)
            pass
    df = pd.DataFrame(rows)
    # Calcular ticket médio real como Faturamento / Pedidos
    df["TicketMedioReal"] = (df["Faturamento"] / df["Pedidos"]).replace([np.inf, -np.inf], np.nan)
    # Calcular diferença entre real e declarado (em valores absolutos)
    df["TicketDiferenca"] = df["TicketMedioReal"] - df["TicketMedioDeclarado"]
    return df.dropna(subset=["Ano", "Mes"])


# Carregar dados
df = load_data()


# ========================
# Sidebar - Filtros
# ========================
st.sidebar.header("Filtros")
anos = sorted(df["Ano"].unique())

ano_min, ano_max = st.sidebar.select_slider(
    "Intervalo de anos",
    options=anos,
    value=(min(anos), max(anos)),
    help="Selecione o intervalo de anos que deseja analisar."
)

meses_opts = list(range(1, 13))
meses_sel = st.sidebar.multiselect(
    "Meses",
    options=meses_opts,
    default=meses_opts,
    format_func=lambda m: MES_NOME[m].upper(),
    help="Selecione os meses (jan, fev, etc.) a incluir nos filtros."
)

lojas_opts = sorted(df["Loja"].unique())
lojas_sel = st.sidebar.multiselect(
    "Lojas",
    options=lojas_opts,
    default=lojas_opts,
    help="Você pode selecionar lojas específicas para comparar."
)

q = st.sidebar.text_input(
    "Busca por loja (contém)",
    value="",
    help="Digite um trecho do nome da loja para filtrar por substring."
).strip().lower()

# Seleção de métrica para ranking
metric_options = {
    "Faturamento": "Faturamento",
    "Pedidos": "Pedidos",
    "Ticket médio real": "TicketMedioReal"
}
metric_choice = st.sidebar.selectbox(
    "Métrica para ranking de lojas",
    options=list(metric_options.keys()),
    index=0,
    help="Escolha qual métrica será usada para ranquear as lojas no gráfico de barras."
)
top_n = st.sidebar.slider(
    "Top N por métrica escolhida",
    min_value=3,
    max_value=25,
    value=10,
    step=1,
    help="Número de lojas a mostrar no ranking."
)


# ========================
# Filtragem
# ========================
f = df.copy()
f = f[(f["Ano"] >= ano_min) & (f["Ano"] <= ano_max) & (f["Mes"].isin(meses_sel))]
if lojas_sel:
    f = f[f["Loja"].isin(lojas_sel)]
if q:
    f = f[f["Loja"].str.lower().str.contains(q)]


# ========================
# KPIs
# ========================
fat_total = float(f["Faturamento"].sum())
ped_total = int(f["Pedidos"].sum())
ticket_real = (fat_total / ped_total) if ped_total > 0 else np.nan
lojas_ativas = f["Loja"].nunique()

st.title("📊 Dashboard de Vendas por Loja")
st.caption("Dados 2019–2025 (valores em R$). Os dados foram embutidos no código conforme solicitado.")

# Mostrar KPIs em colunas
kpi1, kpi2, kpi3, kpi4 = st.columns(4)
kpi1.metric("Faturamento (período filtrado)", brl(fat_total))
kpi2.metric("Pedidos (período filtrado)", f"{ped_total:,}".replace(",", "."))
kpi3.metric("Ticket médio real", brl(ticket_real) if not np.isnan(ticket_real) else "—")
kpi4.metric("Lojas ativas", f"{lojas_ativas}")

# ========================
# Gráficos
# ========================
st.subheader("Tendência de faturamento no tempo")

serie = f.dropna(subset=["Periodo"]).groupby("Periodo", as_index=False)["Faturamento"].sum()
if len(serie) > 0:
    chart = (
        alt.Chart(serie)
        .mark_line(point=True)
        .encode(
            x=alt.X("Periodo:T", title="Período (mês)", axis=alt.Axis(format="%Y-%m")),
            y=alt.Y("Faturamento:Q", title="Faturamento total (R$)", axis=alt.Axis(format="~s")),
            tooltip=[
                alt.Tooltip("Periodo:T", title="Período", format="%Y-%m"),
                alt.Tooltip("Faturamento:Q", title="Faturamento", format=",.2f")
            ]
        )
        .properties(height=320)
    )
    st.altair_chart(chart, use_container_width=True)
else:
    st.info("Sem dados para o período/lojas selecionados.")

st.subheader(f"Top lojas por {metric_choice.lower()} (no filtro)")

metric_col = metric_options[metric_choice]
rank = (
    f.groupby("Loja", as_index=False)[metric_col].sum()
    .sort_values(metric_col, ascending=False)
    .head(top_n)
)
if len(rank) > 0:
    bar = (
        alt.Chart(rank)
        .mark_bar()
        .encode(
            x=alt.X(f"{metric_col}:Q", title=metric_choice, axis=alt.Axis(format="~s")),
            y=alt.Y("Loja:N", sort="-x", title="Loja"),
            color=alt.value("#3BA1C9"),
            tooltip=[
                alt.Tooltip("Loja:N"),
                alt.Tooltip(f"{metric_col}:Q", format=",.2f", title=metric_choice)
            ]
        )
        .properties(height=28 * len(rank) + 40)
    )
    st.altair_chart(bar, use_container_width=True)

    # Dispersão entre faturamento e ticket médio real para lojas selecionadas
    if metric_col == "Faturamento":
        scatter = (
            alt.Chart(rank.merge(f.groupby("Loja", as_index=False)["TicketMedioReal"].mean(), on="Loja"))
            .mark_circle(size=80, color="#F86624")
            .encode(
                x=alt.X("Faturamento:Q", title="Faturamento total", axis=alt.Axis(format="~s")),
                y=alt.Y("TicketMedioReal:Q", title="Ticket médio real"),
                tooltip=[
                    alt.Tooltip("Loja:N"),
                    alt.Tooltip("Faturamento:Q", format=",.2f", title="Faturamento total"),
                    alt.Tooltip("TicketMedioReal:Q", format=",.2f", title="Ticket médio real")
                ]
            )
            .properties(height=300)
        )
        st.altair_chart(scatter, use_container_width=True)
else:
    st.info("Sem lojas suficientes no filtro atual para mostrar ranking.")


# ========================
# Tabela detalhada
# ========================
st.subheader("Detalhamento (linhas)")

# Opções para mostrar ou não algumas colunas adicionais
show_decl = st.toggle("Mostrar coluna 'TicketMedioDeclarado'")
show_diff = st.toggle("Mostrar diferença entre ticket real e declarado")

cols = ["Data", "Loja", "Faturamento", "Pedidos", "TicketMedioReal"]
if show_decl:
    cols.append("TicketMedioDeclarado")
if show_diff:
    cols.append("TicketDiferenca")

tbl = f[cols].sort_values(["Data", "Loja"], ascending=[True, True]).copy()
st.dataframe(
    tbl.style.format({
        "Faturamento": lambda x: brl(x),
        "TicketMedioReal": lambda x: brl(x),
        "TicketMedioDeclarado": lambda x: brl(x),
        "TicketDiferenca": lambda x: brl(x)
    }),
    use_container_width=True,
    height=500
)


# ========================
# Download CSV
# ========================
csv = f.to_csv(index=False).encode("utf-8")
st.download_button(
    "Baixar dados filtrados (CSV)",
    data=csv,
    file_name="dados_filtrados.csv",
    mime="text/csv",
)

st.caption("💡 Dica: use os filtros na barra lateral para focar em anos, meses ou lojas específicas. "
           "O ticket médio real é calculado como Faturamento dividido por Pedidos. "
           "A diferença entre real e declarado pode ajudar a detectar inconsistências nos dados.")