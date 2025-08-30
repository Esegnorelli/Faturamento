// Dados de faturamento da Hora do Pastel
const faturamentoData = [
  {
    "mes": 1,
    "ano": 2022,
    "loja": "Caxias do Sul",
    "faturamento": 113573.38,
    "pedidos": 2839,
    "ticket": 40.0,
    "periodo": "2022-01"
  },
  {
    "mes": 1,
    "ano": 2022,
    "loja": "Novo Hamburgo",
    "faturamento": 109347.46,
    "pedidos": 2734,
    "ticket": 40.0,
    "periodo": "2022-01"
  },
  {
    "mes": 1,
    "ano": 2022,
    "loja": "São Leopoldo",
    "faturamento": 86398.5,
    "pedidos": 2160,
    "ticket": 40.0,
    "periodo": "2022-01"
  },
  {
    "mes": 1,
    "ano": 2022,
    "loja": "Barra Shopping",
    "faturamento": 81688.37,
    "pedidos": 2042,
    "ticket": 40.0,
    "periodo": "2022-01"
  },
  {
    "mes": 1,
    "ano": 2022,
    "loja": "Campo Bom",
    "faturamento": 78692.98,
    "pedidos": 1967,
    "ticket": 40.0,
    "periodo": "2022-01"
  },
  {
    "mes": 1,
    "ano": 2022,
    "loja": "Sapiranga",
    "faturamento": 68319.94,
    "pedidos": 1708,
    "ticket": 40.0,
    "periodo": "2022-01"
  },
  {
    "mes": 1,
    "ano": 2022,
    "loja": "Montenegro",
    "faturamento": 66613.24,
    "pedidos": 1665,
    "ticket": 40.0,
    "periodo": "2022-01"
  },
  {
    "mes": 1,
    "ano": 2022,
    "loja": "Estância Velha",
    "faturamento": 56168.43,
    "pedidos": 1404,
    "ticket": 40.0,
    "periodo": "2022-01"
  },
  {
    "mes": 1,
    "ano": 2022,
    "loja": "Taquara",
    "faturamento": 44794.83,
    "pedidos": 1120,
    "ticket": 40.0,
    "periodo": "2022-01"
  },
  {
    "mes": 1,
    "ano": 2022,
    "loja": "Sapucaia do Sul (Vinicius)",
    "faturamento": 41060.43,
    "pedidos": 1027,
    "ticket": 40.0,
    "periodo": "2022-01"
  },
  {
    "mes": 1,
    "ano": 2022,
    "loja": "Nova Hartz",
    "faturamento": 27738.19,
    "pedidos": 693,
    "ticket": 40.0,
    "periodo": "2022-01"
  },
  {
    "mes": 1,
    "ano": 2022,
    "loja": "Parobé",
    "faturamento": 24847.49,
    "pedidos": 621,
    "ticket": 40.0,
    "periodo": "2022-01"
  },
  {
    "mes": 2,
    "ano": 2022,
    "loja": "Caxias do Sul",
    "faturamento": 122454.15,
    "pedidos": 3061,
    "ticket": 40.0,
    "periodo": "2022-02"
  },
  {
    "mes": 2,
    "ano": 2022,
    "loja": "Novo Hamburgo",
    "faturamento": 102136.1,
    "pedidos": 2553,
    "ticket": 40.0,
    "periodo": "2022-02"
  },
  {
    "mes": 2,
    "ano": 2022,
    "loja": "São Leopoldo",
    "faturamento": 86518.36,
    "pedidos": 2163,
    "ticket": 40.0,
    "periodo": "2022-02"
  },
  {
    "mes": 2,
    "ano": 2022,
    "loja": "Campo Bom",
    "faturamento": 67834.05,
    "pedidos": 1696,
    "ticket": 40.0,
    "periodo": "2022-02"
  },
  {
    "mes": 2,
    "ano": 2022,
    "loja": "Montenegro",
    "faturamento": 61861.5,
    "pedidos": 1547,
    "ticket": 40.0,
    "periodo": "2022-02"
  },
  {
    "mes": 2,
    "ano": 2022,
    "loja": "Barra Shopping",
    "faturamento": 57842.73,
    "pedidos": 1446,
    "ticket": 40.0,
    "periodo": "2022-02"
  },
  {
    "mes": 2,
    "ano": 2022,
    "loja": "Estância Velha",
    "faturamento": 54014.92,
    "pedidos": 1350,
    "ticket": 40.0,
    "periodo": "2022-02"
  },
  {
    "mes": 2,
    "ano": 2022,
    "loja": "Sapiranga",
    "faturamento": 50509.72,
    "pedidos": 1263,
    "ticket": 40.0,
    "periodo": "2022-02"
  },
  {
    "mes": 2,
    "ano": 2022,
    "loja": "Sapucaia do Sul (Vinicius)",
    "faturamento": 43636.69,
    "pedidos": 1091,
    "ticket": 40.0,
    "periodo": "2022-02"
  },
  {
    "mes": 2,
    "ano": 2022,
    "loja": "Taquara",
    "faturamento": 38904.13,
    "pedidos": 973,
    "ticket": 40.0,
    "periodo": "2022-02"
  },
  {
    "mes": 2,
    "ano": 2022,
    "loja": "Nova Hartz",
    "faturamento": 26261.47,
    "pedidos": 657,
    "ticket": 40.0,
    "periodo": "2022-02"
  },
  {
    "mes": 2,
    "ano": 2022,
    "loja": "Parobé",
    "faturamento": 11423.74,
    "pedidos": 286,
    "ticket": 40.0,
    "periodo": "2022-02"
  },
  {
    "mes": 3,
    "ano": 2022,
    "loja": "Caxias do Sul",
    "faturamento": 127887.89,
    "pedidos": 3197,
    "ticket": 40.0,
    "periodo": "2022-03"
  },
  {
    "mes": 3,
    "ano": 2022,
    "loja": "Novo Hamburgo",
    "faturamento": 97693.73,
    "pedidos": 2442,
    "ticket": 40.0,
    "periodo": "2022-03"
  },
  {
    "mes": 3,
    "ano": 2022,
    "loja": "São Leopoldo",
    "faturamento": 88948.6,
    "pedidos": 2224,
    "ticket": 40.0,
    "periodo": "2022-03"
  },
  {
    "mes": 3,
    "ano": 2022,
    "loja": "Barra Shopping",
    "faturamento": 74019.52,
    "pedidos": 1850,
    "ticket": 40.0,
    "periodo": "2022-03"
  },
  {
    "mes": 3,
    "ano": 2022,
    "loja": "Campo Bom",
    "faturamento": 70078.83,
    "pedidos": 1752,
    "ticket": 40.0,
    "periodo": "2022-03"
  },
  {
    "mes": 3,
    "ano": 2022,
    "loja": "Montenegro",
    "faturamento": 64166.06,
    "pedidos": 1604,
    "ticket": 40.0,
    "periodo": "2022-03"
  },
  {
    "mes": 3,
    "ano": 2022,
    "loja": "Sapucaia do Sul (Vinicius)",
    "faturamento": 53045.75,
    "pedidos": 1326,
    "ticket": 40.0,
    "periodo": "2022-03"
  },
  {
    "mes": 3,
    "ano": 2022,
    "loja": "Estância Velha",
    "faturamento": 49705.28,
    "pedidos": 1243,
    "ticket": 40.0,
    "periodo": "2022-03"
  },
  {
    "mes": 3,
    "ano": 2022,
    "loja": "Taquara",
    "faturamento": 44695.67,
    "pedidos": 1117,
    "ticket": 40.0,
    "periodo": "2022-03"
  },
  {
    "mes": 3,
    "ano": 2022,
    "loja": "Sapiranga",
    "faturamento": 43262.25,
    "pedidos": 1082,
    "ticket": 40.0,
    "periodo": "2022-03"
  },
  {
    "mes": 3,
    "ano": 2022,
    "loja": "Nova Hartz",
    "faturamento": 23973.83,
    "pedidos": 599,
    "ticket": 40.0,
    "periodo": "2022-03"
  },
  {
    "mes": 3,
    "ano": 2022,
    "loja": "Parobé",
    "faturamento": 20954.39,
    "pedidos": 524,
    "ticket": 40.0,
    "periodo": "2022-03"
  },
  {
    "mes": 4,
    "ano": 2022,
    "loja": "Caxias do Sul",
    "faturamento": 127511.71,
    "pedidos": 3188,
    "ticket": 40.0,
    "periodo": "2022-04"
  },
  {
    "mes": 4,
    "ano": 2022,
    "loja": "Novo Hamburgo",
    "faturamento": 87438.82,
    "pedidos": 2186,
    "ticket": 40.0,
    "periodo": "2022-04"
  },
  {
    "mes": 4,
    "ano": 2022,
    "loja": "São Leopoldo",
    "faturamento": 86395.35,
    "pedidos": 2160,
    "ticket": 40.0,
    "periodo": "2022-04"
  },
  {
    "mes": 4,
    "ano": 2022,
    "loja": "Campo Bom",
    "faturamento": 72548.83,
    "pedidos": 1814,
    "ticket": 40.0,
    "periodo": "2022-04"
  },
  {
    "mes": 4,
    "ano": 2022,
    "loja": "Sapiranga",
    "faturamento": 70844.9,
    "pedidos": 1771,
    "ticket": 40.0,
    "periodo": "2022-04"
  },
  {
    "mes": 4,
    "ano": 2022,
    "loja": "Barra Shopping",
    "faturamento": 70409.25,
    "pedidos": 1760,
    "ticket": 40.0,
    "periodo": "2022-04"
  },
  {
    "mes": 4,
    "ano": 2022,
    "loja": "Montenegro",
    "faturamento": 61582.73,
    "pedidos": 1540,
    "ticket": 40.0,
    "periodo": "2022-04"
  },
  {
    "mes": 4,
    "ano": 2022,
    "loja": "Sapucaia do Sul (Vinicius)",
    "faturamento": 51947.18,
    "pedidos": 1299,
    "ticket": 40.0,
    "periodo": "2022-04"
  },
  {
    "mes": 4,
    "ano": 2022,
    "loja": "Estância Velha",
    "faturamento": 49985.1,
    "pedidos": 1250,
    "ticket": 40.0,
    "periodo": "2022-04"
  },
  {
    "mes": 4,
    "ano": 2022,
    "loja": "Taquara",
    "faturamento": 39036.29,
    "pedidos": 976,
    "ticket": 40.0,
    "periodo": "2022-04"
  },
  {
    "mes": 4,
    "ano": 2022,
    "loja": "Parobé",
    "faturamento": 27114.7,
    "pedidos": 678,
    "ticket": 40.0,
    "periodo": "2022-04"
  },
  {
    "mes": 4,
    "ano": 2022,
    "loja": "Gravataí (Vinicius)",
    "faturamento": 1692.97,
    "pedidos": 42,
    "ticket": 40.0,
    "periodo": "2022-04"
  },
  {
    "mes": 4,
    "ano": 2022,
    "loja": "Passo d' Areia",
    "faturamento": 180.6,
    "pedidos": 5,
    "ticket": 40.0,
    "periodo": "2022-04"
  },
  {
    "mes": 5,
    "ano": 2022,
    "loja": "Caxias do Sul",
    "faturamento": 135340.16,
    "pedidos": 3384,
    "ticket": 40.0,
    "periodo": "2022-05"
  },
  {
    "mes": 5,
    "ano": 2022,
    "loja": "Novo Hamburgo",
    "faturamento": 111853.53,
    "pedidos": 2796,
    "ticket": 40.0,
    "periodo": "2022-05"
  },
  {
    "mes": 5,
    "ano": 2022,
    "loja": "Barra Shopping",
    "faturamento": 86642.4,
    "pedidos": 2166,
    "ticket": 40.0,
    "periodo": "2022-05"
  },
  {
    "mes": 5,
    "ano": 2022,
    "loja": "São Leopoldo",
    "faturamento": 84227.59,
    "pedidos": 2106,
    "ticket": 40.0,
    "periodo": "2022-05"
  },
  {
    "mes": 5,
    "ano": 2022,
    "loja": "Montenegro",
    "faturamento": 69828.35,
    "pedidos": 1746,
    "ticket": 40.0,
    "periodo": "2022-05"
  },
  {
    "mes": 5,
    "ano": 2022,
    "loja": "Sapucaia do Sul (Vinicius)",
    "faturamento": 68833.79,
    "pedidos": 1721,
    "ticket": 40.0,
    "periodo": "2022-05"
  },
  {
    "mes": 5,
    "ano": 2022,
    "loja": "Campo Bom",
    "faturamento": 68185.28,
    "pedidos": 1705,
    "ticket": 40.0,
    "periodo": "2022-05"
  },
  {
    "mes": 5,
    "ano": 2022,
    "loja": "Sapiranga",
    "faturamento": 67072.06,
    "pedidos": 1677,
    "ticket": 40.0,
    "periodo": "2022-05"
  },
  {
    "mes": 5,
    "ano": 2022,
    "loja": "Estância Velha",
    "faturamento": 50851.68,
    "pedidos": 1271,
    "ticket": 40.0,
    "periodo": "2022-05"
  },
  {
    "mes": 5,
    "ano": 2022,
    "loja": "Passo d' Areia",
    "faturamento": 42091.48,
    "pedidos": 1052,
    "ticket": 40.0,
    "periodo": "2022-05"
  },
  {
    "mes": 5,
    "ano": 2022,
    "loja": "Parobé",
    "faturamento": 28810.59,
    "pedidos": 720,
    "ticket": 40.0,
    "periodo": "2022-05"
  },
  {
    "mes": 5,
    "ano": 2022,
    "loja": "Gravataí (Vinicius)",
    "faturamento": 27093.4,
    "pedidos": 677,
    "ticket": 40.0,
    "periodo": "2022-05"
  },
  {
    "mes": 6,
    "ano": 2022,
    "loja": "Caxias do Sul",
    "faturamento": 130231.57,
    "pedidos": 3256,
    "ticket": 40.0,
    "periodo": "2022-06"
  },
  {
    "mes": 6,
    "ano": 2022,
    "loja": "Novo Hamburgo",
    "faturamento": 106592.51,
    "pedidos": 2665,
    "ticket": 40.0,
    "periodo": "2022-06"
  },
  {
    "mes": 6,
    "ano": 2022,
    "loja": "São Leopoldo",
    "faturamento": 89730.24,
    "pedidos": 2243,
    "ticket": 40.0,
    "periodo": "2022-06"
  },
  {
    "mes": 6,
    "ano": 2022,
    "loja": "Barra Shopping",
    "faturamento": 84036.62,
    "pedidos": 2101,
    "ticket": 40.0,
    "periodo": "2022-06"
  },
  {
    "mes": 6,
    "ano": 2022,
    "loja": "Sapucaia do Sul (Vinicius)",
    "faturamento": 66492.38,
    "pedidos": 1662,
    "ticket": 40.0,
    "periodo": "2022-06"
  },
  {
    "mes": 6,
    "ano": 2022,
    "loja": "Montenegro",
    "faturamento": 66385.92,
    "pedidos": 1660,
    "ticket": 40.0,
    "periodo": "2022-06"
  },
  {
    "mes": 6,
    "ano": 2022,
    "loja": "Campo Bom",
    "faturamento": 63918.09,
    "pedidos": 1598,
    "ticket": 40.0,
    "periodo": "2022-06"
  },
  {
    "mes": 6,
    "ano": 2022,
    "loja": "Sapiranga",
    "faturamento": 60678.24,
    "pedidos": 1517,
    "ticket": 40.0,
    "periodo": "2022-06"
  },
  {
    "mes": 6,
    "ano": 2022,
    "loja": "Parobé",
    "faturamento": 48445.97,
    "pedidos": 1211,
    "ticket": 40.0,
    "periodo": "2022-06"
  },
  {
    "mes": 6,
    "ano": 2022,
    "loja": "Estância Velha",
    "faturamento": 46494.03,
    "pedidos": 1162,
    "ticket": 40.0,
    "periodo": "2022-06"
  },
  {
    "mes": 6,
    "ano": 2022,
    "loja": "Bento Gonçalves",
    "faturamento": 45032.19,
    "pedidos": 1126,
    "ticket": 39.99,
    "periodo": "2022-06"
  },
  {
    "mes": 6,
    "ano": 2022,
    "loja": "Passo d' Areia",
    "faturamento": 42644.73,
    "pedidos": 1066,
    "ticket": 40.0,
    "periodo": "2022-06"
  },
  {
    "mes": 6,
    "ano": 2022,
    "loja": "Gravataí (Vinicius)",
    "faturamento": 41151.55,
    "pedidos": 1029,
    "ticket": 40.0,
    "periodo": "2022-06"
  },
  {
    "mes": 7,
    "ano": 2022,
    "loja": "Caxias do Sul",
    "faturamento": 126122.31,
    "pedidos": 3153,
    "ticket": 40.0,
    "periodo": "2022-07"
  },
  {
    "mes": 7,
    "ano": 2022,
    "loja": "Barra Shopping",
    "faturamento": 94378.55,
    "pedidos": 2359,
    "ticket": 40.0,
    "periodo": "2022-07"
  },
  {
    "mes": 7,
    "ano": 2022,
    "loja": "Novo Hamburgo",
    "faturamento": 93868.42,
    "pedidos": 2347,
    "ticket": 40.0,
    "periodo": "2022-07"
  },
  {
    "mes": 7,
    "ano": 2022,
    "loja": "São Leopoldo",
    "faturamento": 93484.16,
    "pedidos": 2337,
    "ticket": 40.0,
    "periodo": "2022-07"
  },
  {
    "mes": 7,
    "ano": 2022,
    "loja": "Montenegro",
    "faturamento": 62273.25,
    "pedidos": 1557,
    "ticket": 40.0,
    "periodo": "2022-07"
  },
  {
    "mes": 7,
    "ano": 2022,
    "loja": "Campo Bom",
    "faturamento": 60282.49,
    "pedidos": 1507,
    "ticket": 40.0,
    "periodo": "2022-07"
  },
  {
    "mes": 7,
    "ano": 2022,
    "loja": "Sapiranga",
    "faturamento": 59075.81,
    "pedidos": 1477,
    "ticket": 40.0,
    "periodo": "2022-07"
  },
  {
    "mes": 7,
    "ano": 2022,
    "loja": "Bento Gonçalves",
    "faturamento": 55899.24,
    "pedidos": 1397,
    "ticket": 40.0,
    "periodo": "2022-07"
  },
  {
    "mes": 7,
    "ano": 2022,
    "loja": "Sapucaia do Sul (Vinicius)",
    "faturamento": 50160.99,
    "pedidos": 1254,
    "ticket": 40.0,
    "periodo": "2022-07"
  },
  {
    "mes": 7,
    "ano": 2022,
    "loja": "Parobé",
    "faturamento": 48783.03,
    "pedidos": 1220,
    "ticket": 40.0,
    "periodo": "2022-07"
  },
  {
    "mes": 7,
    "ano": 2022,
    "loja": "Passo d' Areia",
    "faturamento": 46669.99,
    "pedidos": 1167,
    "ticket": 40.0,
    "periodo": "2022-07"
  },
  {
    "mes": 7,
    "ano": 2022,
    "loja": "Estância Velha",
    "faturamento": 46650.15,
    "pedidos": 1166,
    "ticket": 40.0,
    "periodo": "2022-07"
  },
  {
    "mes": 7,
    "ano": 2022,
    "loja": "Gravataí (Vinicius)",
    "faturamento": 30008.27,
    "pedidos": 750,
    "ticket": 40.0,
    "periodo": "2022-07"
  },
  {
    "mes": 8,
    "ano": 2022,
    "loja": "Caxias do Sul",
    "faturamento": 116023.31,
    "pedidos": 2901,
    "ticket": 40.0,
    "periodo": "2022-08"
  },
  {
    "mes": 8,
    "ano": 2022,
    "loja": "Montenegro",
    "faturamento": 94135.13,
    "pedidos": 2353,
    "ticket": 40.0,
    "periodo": "2022-08"
  },
  {
    "mes": 8,
    "ano": 2022,
    "loja": "Novo Hamburgo",
    "faturamento": 94135.13,
    "pedidos": 2353,
    "ticket": 40.0,
    "periodo": "2022-08"
  },
  {
    "mes": 8,
    "ano": 2022,
    "loja": "São Leopoldo",
    "faturamento": 86221.9,
    "pedidos": 2156,
    "ticket": 40.0,
    "periodo": "2022-08"
  },
  {
    "mes": 8,
    "ano": 2022,
    "loja": "Barra Shopping",
    "faturamento": 83047.57,
    "pedidos": 2076,
    "ticket": 40.0,
    "periodo": "2022-08"
  },
  {
    "mes": 8,
    "ano": 2022,
    "loja": "Sapiranga",
    "faturamento": 52366.66,
    "pedidos": 1309,
    "ticket": 40.0,
    "periodo": "2022-08"
  },
  {
    "mes": 8,
    "ano": 2022,
    "loja": "Campo Bom",
    "faturamento": 52123.38,
    "pedidos": 1303,
    "ticket": 40.0,
    "periodo": "2022-08"
  },
  {
    "mes": 8,
    "ano": 2022,
    "loja": "Sapucaia do Sul (Vinicius)",
    "faturamento": 41486.34,
    "pedidos": 1037,
    "ticket": 40.0,
    "periodo": "2022-08"
  },
  {
    "mes": 8,
    "ano": 2022,
    "loja": "Parobé",
    "faturamento": 41095.43,
    "pedidos": 1027,
    "ticket": 40.0,
    "periodo": "2022-08"
  },
  {
    "mes": 8,
    "ano": 2022,
    "loja": "Bento Gonçalves",
    "faturamento": 40071.78,
    "pedidos": 1002,
    "ticket": 40.0,
    "periodo": "2022-08"
  },
  {
    "mes": 8,
    "ano": 2022,
    "loja": "Estância Velha",
    "faturamento": 38714.55,
    "pedidos": 968,
    "ticket": 40.0,
    "periodo": "2022-08"
  },
  {
    "mes": 8,
    "ano": 2022,
    "loja": "Passo d' Areia",
    "faturamento": 37741.12,
    "pedidos": 944,
    "ticket": 40.0,
    "periodo": "2022-08"
  },
  {
    "mes": 8,
    "ano": 2022,
    "loja": "Gravataí (Vinicius)",
    "faturamento": 28416.1,
    "pedidos": 710,
    "ticket": 40.0,
    "periodo": "2022-08"
  },
  {
    "mes": 9,
    "ano": 2022,
    "loja": "Caxias do Sul",
    "faturamento": 123235.02,
    "pedidos": 3081,
    "ticket": 40.0,
    "periodo": "2022-09"
  },
  {
    "mes": 9,
    "ano": 2022,
    "loja": "Novo Hamburgo",
    "faturamento": 92538.74,
    "pedidos": 2313,
    "ticket": 40.0,
    "periodo": "2022-09"
  },
  {
    "mes": 9,
    "ano": 2022,
    "loja": "São Leopoldo",
    "faturamento": 85438.97,
    "pedidos": 2136,
    "ticket": 40.0,
    "periodo": "2022-09"
  },
  {
    "mes": 9,
    "ano": 2022,
    "loja": "Barra Shopping",
    "faturamento": 83890.1,
    "pedidos": 2097,
    "ticket": 40.0,
    "periodo": "2022-09"
  },
  {
    "mes": 9,
    "ano": 2022,
    "loja": "Montenegro",
    "faturamento": 64800.87,
    "pedidos": 1620,
    "ticket": 40.0,
    "periodo": "2022-09"
  },
  {
    "mes": 9,
    "ano": 2022,
    "loja": "Shopping Total",
    "faturamento": 58868.46,
    "pedidos": 1472,
    "ticket": 40.0,
    "periodo": "2022-09"
  },
  {
    "mes": 9,
    "ano": 2022,
    "loja": "Sapiranga",
    "faturamento": 54144.32,
    "pedidos": 1354,
    "ticket": 40.0,
    "periodo": "2022-09"
  },
  {
    "mes": 9,
    "ano": 2022,
    "loja": "Estância Velha",
    "faturamento": 46719.89,
    "pedidos": 1168,
    "ticket": 40.0,
    "periodo": "2022-09"
  },
  {
    "mes": 9,
    "ano": 2022,
    "loja": "Sapucaia do Sul (Vinicius)",
    "faturamento": 46310.48,
    "pedidos": 1158,
    "ticket": 40.0,
    "periodo": "2022-09"
  },
  {
    "mes": 9,
    "ano": 2022,
    "loja": "Campo Bom",
    "faturamento": 45891.05,
    "pedidos": 1147,
    "ticket": 40.0,
    "periodo": "2022-09"
  },
  {
    "mes": 9,
    "ano": 2022,
    "loja": "Gravataí (Vinicius)",
    "faturamento": 32847.16,
    "pedidos": 821,
    "ticket": 40.0,
    "periodo": "2022-09"
  },
  {
    "mes": 9,
    "ano": 2022,
    "loja": "Bento Gonçalves",
    "faturamento": 31500.99,
    "pedidos": 788,
    "ticket": 40.0,
    "periodo": "2022-09"
  },
  {
    "mes": 9,
    "ano": 2022,
    "loja": "Parobé",
    "faturamento": 11096.31,
    "pedidos": 277,
    "ticket": 40.0,
    "periodo": "2022-09"
  },
  {
    "mes": 10,
    "ano": 2022,
    "loja": "Caxias do Sul",
    "faturamento": 129679.96,
    "pedidos": 3242,
    "ticket": 40.0,
    "periodo": "2022-10"
  },
  {
    "mes": 10,
    "ano": 2022,
    "loja": "Novo Hamburgo",
    "faturamento": 102027.07,
    "pedidos": 2551,
    "ticket": 40.0,
    "periodo": "2022-10"
  },
  {
    "mes": 10,
    "ano": 2022,
    "loja": "São Leopoldo",
    "faturamento": 82395.81,
    "pedidos": 2060,
    "ticket": 40.0,
    "periodo": "2022-10"
  },
  {
    "mes": 10,
    "ano": 2022,
    "loja": "Barra Shopping",
    "faturamento": 78738.54,
    "pedidos": 1968,
    "ticket": 40.0,
    "periodo": "2022-10"
  },
  {
    "mes": 10,
    "ano": 2022,
    "loja": "Shopping Total",
    "faturamento": 65063.39,
    "pedidos": 1627,
    "ticket": 40.0,
    "periodo": "2022-10"
  },
  {
    "mes": 10,
    "ano": 2022,
    "loja": "Montenegro",
    "faturamento": 64401.12,
    "pedidos": 1610,
    "ticket": 40.0,
    "periodo": "2022-10"
  },
  {
    "mes": 10,
    "ano": 2022,
    "loja": "Sapiranga",
    "faturamento": 60550.62,
    "pedidos": 1514,
    "ticket": 40.0,
    "periodo": "2022-10"
  },
  {
    "mes": 10,
    "ano": 2022,
    "loja": "Campo Bom",
    "faturamento": 52520.45,
    "pedidos": 1313,
    "ticket": 40.0,
    "periodo": "2022-10"
  },
  {
    "mes": 10,
    "ano": 2022,
    "loja": "Sapucaia do Sul (Vinicius)",
    "faturamento": 47915.71,
    "pedidos": 1198,
    "ticket": 40.0,
    "periodo": "2022-10"
  },
  {
    "mes": 10,
    "ano": 2022,
    "loja": "Estância Velha",
    "faturamento": 47000.79,
    "pedidos": 1175,
    "ticket": 40.0,
    "periodo": "2022-10"
  },
  {
    "mes": 10,
    "ano": 2022,
    "loja": "Bento Gonçalves",
    "faturamento": 45406.58,
    "pedidos": 1135,
    "ticket": 40.0,
    "periodo": "2022-10"
  },
  {
    "mes": 10,
    "ano": 2022,
    "loja": "Gravataí (Vinicius)",
    "faturamento": 35190.0,
    "pedidos": 880,
    "ticket": 40.0,
    "periodo": "2022-10"
  },
  {
    "mes": 11,
    "ano": 2022,
    "loja": "Caxias do Sul",
    "faturamento": 133514.87,
    "pedidos": 3338,
    "ticket": 40.0,
    "periodo": "2022-11"
  },
  {
    "mes": 11,
    "ano": 2022,
    "loja": "Novo Hamburgo",
    "faturamento": 104316.03,
    "pedidos": 2608,
    "ticket": 40.0,
    "periodo": "2022-11"
  },
  {
    "mes": 11,
    "ano": 2022,
    "loja": "Barra Shopping",
    "faturamento": 84529.68,
    "pedidos": 2113,
    "ticket": 40.0,
    "periodo": "2022-11"
  },
  {
    "mes": 11,
    "ano": 2022,
    "loja": "São Leopoldo",
    "faturamento": 80987.86,
    "pedidos": 2025,
    "ticket": 40.0,
    "periodo": "2022-11"
  },
  {
    "mes": 11,
    "ano": 2022,
    "loja": "Shopping Total",
    "faturamento": 67357.39,
    "pedidos": 1684,
    "ticket": 40.0,
    "periodo": "2022-11"
  },
  {
    "mes": 11,
    "ano": 2022,
    "loja": "Sapiranga",
    "faturamento": 62609.36,
    "pedidos": 1565,
    "ticket": 40.0,
    "periodo": "2022-11"
  },
  {
    "mes": 11,
    "ano": 2022,
    "loja": "Montenegro",
    "faturamento": 56334.0,
    "pedidos": 1408,
    "ticket": 40.0,
    "periodo": "2022-11"
  },
  {
    "mes": 11,
    "ano": 2022,
    "loja": "Bento Gonçalves",
    "faturamento": 50879.49,
    "pedidos": 1272,
    "ticket": 40.0,
    "periodo": "2022-11"
  },
  {
    "mes": 11,
    "ano": 2022,
    "loja": "Campo Bom",
    "faturamento": 47723.57,
    "pedidos": 1193,
    "ticket": 40.0,
    "periodo": "2022-11"
  },
  {
    "mes": 11,
    "ano": 2022,
    "loja": "Sapucaia do Sul (Vinicius)",
    "faturamento": 47287.04,
    "pedidos": 1182,
    "ticket": 40.0,
    "periodo": "2022-11"
  },
  {
    "mes": 11,
    "ano": 2022,
    "loja": "Estância Velha",
    "faturamento": 42640.67,
    "pedidos": 1066,
    "ticket": 40.0,
    "periodo": "2022-11"
  },
  {
    "mes": 11,
    "ano": 2022,
    "loja": "Gravataí (Vinicius)",
    "faturamento": 36978.01,
    "pedidos": 924,
    "ticket": 40.0,
    "periodo": "2022-11"
  },
  {
    "mes": 12,
    "ano": 2022,
    "loja": "Caxias do Sul",
    "faturamento": 145346.63,
    "pedidos": 3634,
    "ticket": 40.0,
    "periodo": "2022-12"
  },
  {
    "mes": 12,
    "ano": 2022,
    "loja": "Barra Shopping",
    "faturamento": 114359.18,
    "pedidos": 2859,
    "ticket": 40.0,
    "periodo": "2022-12"
  },
  {
    "mes": 12,
    "ano": 2022,
    "loja": "Novo Hamburgo",
    "faturamento": 113302.42,
    "pedidos": 2833,
    "ticket": 40.0,
    "periodo": "2022-12"
  },
  {
    "mes": 12,
    "ano": 2022,
    "loja": "São Leopoldo",
    "faturamento": 83431.88,
    "pedidos": 2086,
    "ticket": 40.0,
    "periodo": "2022-12"
  },
  {
    "mes": 12,
    "ano": 2022,
    "loja": "Sapiranga",
    "faturamento": 68956.81,
    "pedidos": 1724,
    "ticket": 40.0,
    "periodo": "2022-12"
  },
  {
    "mes": 12,
    "ano": 2022,
    "loja": "Bento Gonçalves",
    "faturamento": 68852.98,
    "pedidos": 1721,
    "ticket": 40.0,
    "periodo": "2022-12"
  },
  {
    "mes": 12,
    "ano": 2022,
    "loja": "Shopping Total",
    "faturamento": 68654.19,
    "pedidos": 1716,
    "ticket": 40.0,
    "periodo": "2022-12"
  },
  {
    "mes": 12,
    "ano": 2022,
    "loja": "Montenegro",
    "faturamento": 66423.65,
    "pedidos": 1661,
    "ticket": 40.0,
    "periodo": "2022-12"
  },
  {
    "mes": 12,
    "ano": 2022,
    "loja": "Campo Bom",
    "faturamento": 60028.88,
    "pedidos": 1501,
    "ticket": 40.0,
    "periodo": "2022-12"
  },
  {
    "mes": 12,
    "ano": 2022,
    "loja": "Estância Velha",
    "faturamento": 47261.17,
    "pedidos": 1182,
    "ticket": 40.0,
    "periodo": "2022-12"
  },
  {
    "mes": 12,
    "ano": 2022,
    "loja": "Sapucaia do Sul (Vinicius)",
    "faturamento": 23224.55,
    "pedidos": 581,
    "ticket": 40.0,
    "periodo": "2022-12"
  },
  {
    "mes": 12,
    "ano": 2022,
    "loja": "Gravataí (Vinicius)",
    "faturamento": 19684.1,
    "pedidos": 492,
    "ticket": 40.0,
    "periodo": "2022-12"
  },
  {
    "mes": 1,
    "ano": 2023,
    "loja": "Caxias do Sul",
    "faturamento": 121801.28,
    "pedidos": 3045,
    "ticket": 40.0,
    "periodo": "2023-01"
  },
  {
    "mes": 1,
    "ano": 2023,
    "loja": "Barra Shopping",
    "faturamento": 90203.42,
    "pedidos": 2255,
    "ticket": 40.0,
    "periodo": "2023-01"
  },
  {
    "mes": 1,
    "ano": 2023,
    "loja": "Novo Hamburgo",
    "faturamento": 88157.53,
    "pedidos": 2204,
    "ticket": 40.0,
    "periodo": "2023-01"
  },
  {
    "mes": 1,
    "ano": 2023,
    "loja": "São Leopoldo",
    "faturamento": 69874.45,
    "pedidos": 1747,
    "ticket": 40.0,
    "periodo": "2023-01"
  },
  {
    "mes": 1,
    "ano": 2023,
    "loja": "Sapiranga",
    "faturamento": 65163.84,
    "pedidos": 1629,
    "ticket": 40.0,
    "periodo": "2023-01"
  },
  {
    "mes": 1,
    "ano": 2023,
    "loja": "Bento Gonçalves",
    "faturamento": 61906.28,
    "pedidos": 1548,
    "ticket": 40.0,
    "periodo": "2023-01"
  },
  {
    "mes": 1,
    "ano": 2023,
    "loja": "Montenegro",
    "faturamento": 55479.93,
    "pedidos": 1387,
    "ticket": 40.0,
    "periodo": "2023-01"
  },
  {
    "mes": 1,
    "ano": 2023,
    "loja": "Campo Bom",
    "faturamento": 52482.61,
    "pedidos": 1312,
    "ticket": 40.0,
    "periodo": "2023-01"
  },
  {
    "mes": 1,
    "ano": 2023,
    "loja": "Shopping Total",
    "faturamento": 52314.19,
    "pedidos": 1308,
    "ticket": 40.0,
    "periodo": "2023-01"
  },
  {
    "mes": 1,
    "ano": 2023,
    "loja": "Estância Velha",
    "faturamento": 46351.82,
    "pedidos": 1159,
    "ticket": 40.0,
    "periodo": "2023-01"
  },
  {
    "mes": 2,
    "ano": 2023,
    "loja": "Caxias do Sul",
    "faturamento": 115396.2,
    "pedidos": 2885,
    "ticket": 40.0,
    "periodo": "2023-02"
  },
  {
    "mes": 2,
    "ano": 2023,
    "loja": "Novo Hamburgo",
    "faturamento": 83940.51,
    "pedidos": 2099,
    "ticket": 40.0,
    "periodo": "2023-02"
  },
  {
    "mes": 2,
    "ano": 2023,
    "loja": "Barra Shopping",
    "faturamento": 79548.77,
    "pedidos": 1989,
    "ticket": 40.0,
    "periodo": "2023-02"
  },
  {
    "mes": 2,
    "ano": 2023,
    "loja": "São Leopoldo",
    "faturamento": 73886.59,
    "pedidos": 1847,
    "ticket": 40.0,
    "periodo": "2023-02"
  },
  {
    "mes": 2,
    "ano": 2023,
    "loja": "Bento Gonçalves",
    "faturamento": 62326.83,
    "pedidos": 1558,
    "ticket": 40.0,
    "periodo": "2023-02"
  },
  {
    "mes": 2,
    "ano": 2023,
    "loja": "Sapiranga",
    "faturamento": 57426.75,
    "pedidos": 1436,
    "ticket": 40.0,
    "periodo": "2023-02"
  },
  {
    "mes": 2,
    "ano": 2023,
    "loja": "Montenegro",
    "faturamento": 55948.57,
    "pedidos": 1399,
    "ticket": 40.0,
    "periodo": "2023-02"
  },
  {
    "mes": 2,
    "ano": 2023,
    "loja": "Shopping Total",
    "faturamento": 48738.83,
    "pedidos": 1218,
    "ticket": 40.0,
    "periodo": "2023-02"
  },
  {
    "mes": 2,
    "ano": 2023,
    "loja": "Campo Bom",
    "faturamento": 48259.78,
    "pedidos": 1206,
    "ticket": 40.0,
    "periodo": "2023-02"
  },
  {
    "mes": 2,
    "ano": 2023,
    "loja": "Estância Velha",
    "faturamento": 40680.68,
    "pedidos": 1017,
    "ticket": 40.0,
    "periodo": "2023-02"
  },
  {
    "mes": 3,
    "ano": 2023,
    "loja": "Caxias do Sul",
    "faturamento": 128478.28,
    "pedidos": 3212,
    "ticket": 40.0,
    "periodo": "2023-03"
  },
  {
    "mes": 3,
    "ano": 2023,
    "loja": "Novo Hamburgo",
    "faturamento": 88832.21,
    "pedidos": 2221,
    "ticket": 40.0,
    "periodo": "2023-03"
  },
  {
    "mes": 3,
    "ano": 2023,
    "loja": "Barra Shopping",
    "faturamento": 78838.03,
    "pedidos": 1971,
    "ticket": 40.0,
    "periodo": "2023-03"
  },
  {
    "mes": 3,
    "ano": 2023,
    "loja": "São Leopoldo",
    "faturamento": 78824.79,
    "pedidos": 1971,
    "ticket": 40.0,
    "periodo": "2023-03"
  },
  {
    "mes": 3,
    "ano": 2023,
    "loja": "Bento Gonçalves",
    "faturamento": 60864.02,
    "pedidos": 1522,
    "ticket": 40.0,
    "periodo": "2023-03"
  },
  {
    "mes": 3,
    "ano": 2023,
    "loja": "Shopping Total",
    "faturamento": 60182.44,
    "pedidos": 1505,
    "ticket": 40.0,
    "periodo": "2023-03"
  },
  {
    "mes": 3,
    "ano": 2023,
    "loja": "Sapiranga",
    "faturamento": 58512.69,
    "pedidos": 1463,
    "ticket": 40.0,
    "periodo": "2023-03"
  },
  {
    "mes": 3,
    "ano": 2023,
    "loja": "Montenegro",
    "faturamento": 52020.93,
    "pedidos": 1301,
    "ticket": 40.0,
    "periodo": "2023-03"
  },
  {
    "mes": 3,
    "ano": 2023,
    "loja": "Campo Bom",
    "faturamento": 51589.66,
    "pedidos": 1290,
    "ticket": 40.0,
    "periodo": "2023-03"
  },
  {
    "mes": 3,
    "ano": 2023,
    "loja": "Estância Velha",
    "faturamento": 46230.41,
    "pedidos": 1156,
    "ticket": 40.0,
    "periodo": "2023-03"
  },
  {
    "mes": 3,
    "ano": 2023,
    "loja": "Canoas",
    "faturamento": 22604.87,
    "pedidos": 565,
    "ticket": 40.0,
    "periodo": "2023-03"
  },
  {
    "mes": 4,
    "ano": 2023,
    "loja": "Caxias do Sul",
    "faturamento": 132170.64,
    "pedidos": 3304,
    "ticket": 40.0,
    "periodo": "2023-04"
  },
  {
    "mes": 4,
    "ano": 2023,
    "loja": "Novo Hamburgo",
    "faturamento": 94378.13,
    "pedidos": 2359,
    "ticket": 40.0,
    "periodo": "2023-04"
  },
  {
    "mes": 4,
    "ano": 2023,
    "loja": "Barra Shopping",
    "faturamento": 84831.17,
    "pedidos": 2121,
    "ticket": 40.0,
    "periodo": "2023-04"
  },
  {
    "mes": 4,
    "ano": 2023,
    "loja": "São Leopoldo",
    "faturamento": 74857.51,
    "pedidos": 1871,
    "ticket": 40.0,
    "periodo": "2023-04"
  },
  {
    "mes": 4,
    "ano": 2023,
    "loja": "Montenegro",
    "faturamento": 60754.93,
    "pedidos": 1519,
    "ticket": 40.0,
    "periodo": "2023-04"
  },
  {
    "mes": 4,
    "ano": 2023,
    "loja": "Shopping Total",
    "faturamento": 58254.57,
    "pedidos": 1456,
    "ticket": 40.0,
    "periodo": "2023-04"
  },
  {
    "mes": 4,
    "ano": 2023,
    "loja": "Sapiranga",
    "faturamento": 58055.34,
    "pedidos": 1451,
    "ticket": 40.0,
    "periodo": "2023-04"
  },
  {
    "mes": 4,
    "ano": 2023,
    "loja": "Bento Gonçalves",
    "faturamento": 55433.79,
    "pedidos": 1386,
    "ticket": 40.0,
    "periodo": "2023-04"
  },
  {
    "mes": 4,
    "ano": 2023,
    "loja": "Campo Bom",
    "faturamento": 49530.31,
    "pedidos": 1238,
    "ticket": 40.0,
    "periodo": "2023-04"
  },
  {
    "mes": 4,
    "ano": 2023,
    "loja": "Estância Velha",
    "faturamento": 42372.51,
    "pedidos": 1059,
    "ticket": 40.0,
    "periodo": "2023-04"
  },
  {
    "mes": 4,
    "ano": 2023,
    "loja": "Canoas",
    "faturamento": 39736.4,
    "pedidos": 993,
    "ticket": 40.0,
    "periodo": "2023-04"
  },
  {
    "mes": 5,
    "ano": 2023,
    "loja": "Caxias do Sul",
    "faturamento": 150441.83,
    "pedidos": 3761,
    "ticket": 40.0,
    "periodo": "2023-05"
  },
  {
    "mes": 5,
    "ano": 2023,
    "loja": "Novo Hamburgo",
    "faturamento": 118109.36,
    "pedidos": 2953,
    "ticket": 40.0,
    "periodo": "2023-05"
  },
  {
    "mes": 5,
    "ano": 2023,
    "loja": "Barra Shopping",
    "faturamento": 85362.47,
    "pedidos": 2134,
    "ticket": 40.0,
    "periodo": "2023-05"
  },
  {
    "mes": 5,
    "ano": 2023,
    "loja": "São Leopoldo",
    "faturamento": 76057.43,
    "pedidos": 1901,
    "ticket": 40.0,
    "periodo": "2023-05"
  },
  {
    "mes": 5,
    "ano": 2023,
    "loja": "Bento Gonçalves",
    "faturamento": 63288.65,
    "pedidos": 1582,
    "ticket": 40.0,
    "periodo": "2023-05"
  },
  {
    "mes": 5,
    "ano": 2023,
    "loja": "Sapiranga",
    "faturamento": 62304.84,
    "pedidos": 1558,
    "ticket": 40.0,
    "periodo": "2023-05"
  },
  {
    "mes": 5,
    "ano": 2023,
    "loja": "Shopping Total",
    "faturamento": 61904.43,
    "pedidos": 1548,
    "ticket": 40.0,
    "periodo": "2023-05"
  },
  {
    "mes": 5,
    "ano": 2023,
    "loja": "Campo Bom",
    "faturamento": 52015.35,
    "pedidos": 1300,
    "ticket": 40.0,
    "periodo": "2023-05"
  },
  {
    "mes": 5,
    "ano": 2023,
    "loja": "Canoas",
    "faturamento": 41164.67,
    "pedidos": 1029,
    "ticket": 40.0,
    "periodo": "2023-05"
  },
  {
    "mes": 5,
    "ano": 2023,
    "loja": "Estância Velha",
    "faturamento": 31634.19,
    "pedidos": 791,
    "ticket": 40.0,
    "periodo": "2023-05"
  },
  {
    "mes": 6,
    "ano": 2023,
    "loja": "Caxias do Sul",
    "faturamento": 139405.15,
    "pedidos": 3485,
    "ticket": 40.0,
    "periodo": "2023-06"
  },
  {
    "mes": 6,
    "ano": 2023,
    "loja": "Novo Hamburgo",
    "faturamento": 125073.62,
    "pedidos": 3127,
    "ticket": 40.0,
    "periodo": "2023-06"
  },
  {
    "mes": 6,
    "ano": 2023,
    "loja": "São Leopoldo",
    "faturamento": 94150.53,
    "pedidos": 2354,
    "ticket": 40.0,
    "periodo": "2023-06"
  },
  {
    "mes": 6,
    "ano": 2023,
    "loja": "Barra Shopping",
    "faturamento": 92599.35,
    "pedidos": 2315,
    "ticket": 40.0,
    "periodo": "2023-06"
  },
  {
    "mes": 6,
    "ano": 2023,
    "loja": "Shopping Total",
    "faturamento": 62318.55,
    "pedidos": 1558,
    "ticket": 40.0,
    "periodo": "2023-06"
  },
  {
    "mes": 6,
    "ano": 2023,
    "loja": "Sapiranga",
    "faturamento": 58466.73,
    "pedidos": 1462,
    "ticket": 40.0,
    "periodo": "2023-06"
  },
  {
    "mes": 6,
    "ano": 2023,
    "loja": "Bento Gonçalves",
    "faturamento": 55852.14,
    "pedidos": 1396,
    "ticket": 40.0,
    "periodo": "2023-06"
  },
  {
    "mes": 6,
    "ano": 2023,
    "loja": "Campo Bom",
    "faturamento": 48823.21,
    "pedidos": 1221,
    "ticket": 40.0,
    "periodo": "2023-06"
  },
  {
    "mes": 7,
    "ano": 2023,
    "loja": "Caxias do Sul",
    "faturamento": 132482.84,
    "pedidos": 3312,
    "ticket": 40.0,
    "periodo": "2023-07"
  },
  {
    "mes": 7,
    "ano": 2023,
    "loja": "Bento Gonçalves",
    "faturamento": 57698.26,
    "pedidos": 1442,
    "ticket": 40.0,
    "periodo": "2023-07"
  },
  {
    "mes": 7,
    "ano": 2023,
    "loja": "Campo Bom",
    "faturamento": 45965.29,
    "pedidos": 1155,
    "ticket": 39.8,
    "periodo": "2023-07"
  },
  {
    "mes": 7,
    "ano": 2023,
    "loja": "Canoas Mathias",
    "faturamento": 16644.68,
    "pedidos": 416,
    "ticket": 40.0,
    "periodo": "2023-07"
  },
  {
    "mes": 9,
    "ano": 2023,
    "loja": "Canoas Mathias",
    "faturamento": 32600.93,
    "pedidos": 823,
    "ticket": 39.61,
    "periodo": "2023-09"
  },
  {
    "mes": 12,
    "ano": 2023,
    "loja": "Campo Bom",
    "faturamento": 48229.69,
    "pedidos": 1194,
    "ticket": 40.39,
    "periodo": "2023-12"
  },
  {
    "mes": 12,
    "ano": 2023,
    "loja": "Canoas Mathias",
    "faturamento": 31665.1,
    "pedidos": 794,
    "ticket": 39.88,
    "periodo": "2023-12"
  },
  {
    "mes": 1,
    "ano": 2024,
    "loja": "Sapiranga",
    "faturamento": 49430.14,
    "pedidos": 1238,
    "ticket": 39.93,
    "periodo": "2024-01"
  },
  {
    "mes": 1,
    "ano": 2024,
    "loja": "Canoas Mathias",
    "faturamento": 24545.82,
    "pedidos": 614,
    "ticket": 39.98,
    "periodo": "2024-01"
  },
  {
    "mes": 2,
    "ano": 2024,
    "loja": "Campo Bom",
    "faturamento": 44478.2,
    "pedidos": 1120,
    "ticket": 39.71,
    "periodo": "2024-02"
  },
  {
    "mes": 4,
    "ano": 2024,
    "loja": "Campo Bom",
    "faturamento": 39746.22,
    "pedidos": 983,
    "ticket": 40.43,
    "periodo": "2024-04"
  },
  {
    "mes": 5,
    "ano": 2024,
    "loja": "Erechim",
    "faturamento": 82184.05,
    "pedidos": 2038,
    "ticket": 40.33,
    "periodo": "2024-05"
  },
  {
    "mes": 7,
    "ano": 2024,
    "loja": "Zona Norte",
    "faturamento": 5179.86,
    "pedidos": 130,
    "ticket": 39.85,
    "periodo": "2024-07"
  },
  {
    "mes": 9,
    "ano": 2024,
    "loja": "Erechim",
    "faturamento": 86813.39,
    "pedidos": 2153,
    "ticket": 40.32,
    "periodo": "2024-09"
  },
  {
    "mes": 9,
    "ano": 2024,
    "loja": "Campo Bom",
    "faturamento": 41789.88,
    "pedidos": 1033,
    "ticket": 40.45,
    "periodo": "2024-09"
  },
  {
    "mes": 10,
    "ano": 2024,
    "loja": "Sapiranga",
    "faturamento": 72556.09,
    "pedidos": 1826,
    "ticket": 39.73,
    "periodo": "2024-10"
  },
  {
    "mes": 12,
    "ano": 2024,
    "loja": "Floresta",
    "faturamento": 40688.48,
    "pedidos": 1006,
    "ticket": 40.45,
    "periodo": "2024-12"
  },
  {
    "mes": 3,
    "ano": 2025,
    "loja": "Floresta",
    "faturamento": 61423.0,
    "pedidos": 1545,
    "ticket": 39.76,
    "periodo": "2025-03"
  },
  {
    "mes": 3,
    "ano": 2025,
    "loja": "Porto Alegre Zona Norte",
    "faturamento": 10186.0,
    "pedidos": 256,
    "ticket": 39.79,
    "periodo": "2025-03"
  },
  {
    "mes": 4,
    "ano": 2025,
    "loja": "Floresta",
    "faturamento": 56489.37,
    "pedidos": 1404,
    "ticket": 40.23,
    "periodo": "2025-04"
  },
  {
    "mes": 9,
    "ano": 2019,
    "loja": "Sapiranga",
    "faturamento": 126219.14,
    "pedidos": 3712,
    "ticket": 34.0,
    "periodo": "2019-09"
  },
  {
    "mes": 9,
    "ano": 2019,
    "loja": "Campo Bom",
    "faturamento": 89989.0,
    "pedidos": 2647,
    "ticket": 34.0,
    "periodo": "2019-09"
  },
  {
    "mes": 10,
    "ano": 2019,
    "loja": "Campo Bom",
    "faturamento": 99432.2,
    "pedidos": 2924,
    "ticket": 34.0,
    "periodo": "2019-10"
  },
  {
    "mes": 11,
    "ano": 2019,
    "loja": "Novo Hamburgo",
    "faturamento": 41256.74,
    "pedidos": 1213,
    "ticket": 34.0,
    "periodo": "2019-11"
  },
  {
    "mes": 12,
    "ano": 2019,
    "loja": "Novo Hamburgo",
    "faturamento": 124124.47,
    "pedidos": 3651,
    "ticket": 34.0,
    "periodo": "2019-12"
  },
  {
    "mes": 12,
    "ano": 2019,
    "loja": "Sapiranga",
    "faturamento": 120044.74,
    "pedidos": 3531,
    "ticket": 34.0,
    "periodo": "2019-12"
  },
  {
    "mes": 12,
    "ano": 2019,
    "loja": "Campo Bom",
    "faturamento": 112115.4,
    "pedidos": 3298,
    "ticket": 34.0,
    "periodo": "2019-12"
  },
  {
    "mes": 2,
    "ano": 2020,
    "loja": "Sapiranga",
    "faturamento": 128035.24,
    "pedidos": 3766,
    "ticket": 34.0,
    "periodo": "2020-02"
  },
  {
    "mes": 3,
    "ano": 2020,
    "loja": "Taquara",
    "faturamento": 80907.16,
    "pedidos": 2380,
    "ticket": 34.0,
    "periodo": "2020-03"
  },
  {
    "mes": 4,
    "ano": 2020,
    "loja": "Campo Bom",
    "faturamento": 80743.9,
    "pedidos": 2375,
    "ticket": 34.0,
    "periodo": "2020-04"
  },
  {
    "mes": 4,
    "ano": 2020,
    "loja": "Taquara",
    "faturamento": 51500.0,
    "pedidos": 1515,
    "ticket": 34.0,
    "periodo": "2020-04"
  },
  {
    "mes": 5,
    "ano": 2020,
    "loja": "Sapiranga",
    "faturamento": 89308.63,
    "pedidos": 2627,
    "ticket": 34.0,
    "periodo": "2020-05"
  },
  {
    "mes": 5,
    "ano": 2020,
    "loja": "Novo Hamburgo",
    "faturamento": 78241.61,
    "pedidos": 2301,
    "ticket": 34.0,
    "periodo": "2020-05"
  },
  {
    "mes": 5,
    "ano": 2020,
    "loja": "Campo Bom",
    "faturamento": 71073.6,
    "pedidos": 2090,
    "ticket": 34.0,
    "periodo": "2020-05"
  },
  {
    "mes": 5,
    "ano": 2020,
    "loja": "Taquara",
    "faturamento": 55241.1,
    "pedidos": 1625,
    "ticket": 34.0,
    "periodo": "2020-05"
  },
  {
    "mes": 6,
    "ano": 2020,
    "loja": "Novo Hamburgo",
    "faturamento": 71306.0,
    "pedidos": 2097,
    "ticket": 34.0,
    "periodo": "2020-06"
  },
  {
    "mes": 8,
    "ano": 2020,
    "loja": "Sapiranga",
    "faturamento": 112747.91,
    "pedidos": 3316,
    "ticket": 34.0,
    "periodo": "2020-08"
  },
  {
    "mes": 8,
    "ano": 2020,
    "loja": "Taquara",
    "faturamento": 85522.48,
    "pedidos": 2515,
    "ticket": 34.0,
    "periodo": "2020-08"
  },
  {
    "mes": 9,
    "ano": 2020,
    "loja": "Montenegro",
    "faturamento": 83083.34,
    "pedidos": 2444,
    "ticket": 34.0,
    "periodo": "2020-09"
  },
  {
    "mes": 9,
    "ano": 2020,
    "loja": "Campo Bom",
    "faturamento": 73069.97,
    "pedidos": 2149,
    "ticket": 34.0,
    "periodo": "2020-09"
  },
  {
    "mes": 10,
    "ano": 2020,
    "loja": "Novo Hamburgo",
    "faturamento": 126337.62,
    "pedidos": 3716,
    "ticket": 34.0,
    "periodo": "2020-10"
  },
  {
    "mes": 10,
    "ano": 2020,
    "loja": "Campo Bom",
    "faturamento": 73303.7,
    "pedidos": 2156,
    "ticket": 34.0,
    "periodo": "2020-10"
  },
  {
    "mes": 11,
    "ano": 2020,
    "loja": "Montenegro",
    "faturamento": 89574.03,
    "pedidos": 2635,
    "ticket": 34.0,
    "periodo": "2020-11"
  },
  {
    "mes": 11,
    "ano": 2020,
    "loja": "Sapiranga",
    "faturamento": 89030.31,
    "pedidos": 2619,
    "ticket": 34.0,
    "periodo": "2020-11"
  },
  {
    "mes": 11,
    "ano": 2020,
    "loja": "Taquara",
    "faturamento": 68116.66,
    "pedidos": 2003,
    "ticket": 34.0,
    "periodo": "2020-11"
  },
  {
    "mes": 12,
    "ano": 2020,
    "loja": "Campo Bom",
    "faturamento": 82911.98,
    "pedidos": 2439,
    "ticket": 34.0,
    "periodo": "2020-12"
  },
  {
    "mes": 1,
    "ano": 2021,
    "loja": "Sapiranga",
    "faturamento": 84739.93,
    "pedidos": 2492,
    "ticket": 34.0,
    "periodo": "2021-01"
  },
  {
    "mes": 1,
    "ano": 2021,
    "loja": "Campo Bom",
    "faturamento": 74820.76,
    "pedidos": 2201,
    "ticket": 34.0,
    "periodo": "2021-01"
  },
  {
    "mes": 2,
    "ano": 2021,
    "loja": "Novo Hamburgo",
    "faturamento": 87144.53,
    "pedidos": 2563,
    "ticket": 34.0,
    "periodo": "2021-02"
  },
  {
    "mes": 2,
    "ano": 2021,
    "loja": "Montenegro",
    "faturamento": 80166.19,
    "pedidos": 2358,
    "ticket": 34.0,
    "periodo": "2021-02"
  },
  {
    "mes": 2,
    "ano": 2021,
    "loja": "Caxias do Sul",
    "faturamento": 76513.47,
    "pedidos": 2250,
    "ticket": 34.0,
    "periodo": "2021-02"
  },
  {
    "mes": 3,
    "ano": 2021,
    "loja": "Caxias do Sul",
    "faturamento": 85503.81,
    "pedidos": 2515,
    "ticket": 34.0,
    "periodo": "2021-03"
  },
  {
    "mes": 3,
    "ano": 2021,
    "loja": "Montenegro",
    "faturamento": 76841.08,
    "pedidos": 2260,
    "ticket": 34.0,
    "periodo": "2021-03"
  },
  {
    "mes": 3,
    "ano": 2021,
    "loja": "Campo Bom",
    "faturamento": 45668.95,
    "pedidos": 1343,
    "ticket": 34.0,
    "periodo": "2021-03"
  },
  {
    "mes": 4,
    "ano": 2021,
    "loja": "Novo Hamburgo",
    "faturamento": 82512.95,
    "pedidos": 2427,
    "ticket": 34.0,
    "periodo": "2021-04"
  },
  {
    "mes": 4,
    "ano": 2021,
    "loja": "Estância Velha",
    "faturamento": 62896.7,
    "pedidos": 1850,
    "ticket": 34.0,
    "periodo": "2021-04"
  },
  {
    "mes": 5,
    "ano": 2021,
    "loja": "Novo Hamburgo",
    "faturamento": 96870.49,
    "pedidos": 2849,
    "ticket": 34.0,
    "periodo": "2021-05"
  },
  {
    "mes": 5,
    "ano": 2021,
    "loja": "Caxias do Sul",
    "faturamento": 76170.19,
    "pedidos": 2240,
    "ticket": 34.0,
    "periodo": "2021-05"
  },
  {
    "mes": 5,
    "ano": 2021,
    "loja": "São Leopoldo",
    "faturamento": 66010.93,
    "pedidos": 1941,
    "ticket": 34.0,
    "periodo": "2021-05"
  },
  {
    "mes": 5,
    "ano": 2021,
    "loja": "Sapiranga",
    "faturamento": 58601.29,
    "pedidos": 1724,
    "ticket": 34.0,
    "periodo": "2021-05"
  },
  {
    "mes": 5,
    "ano": 2021,
    "loja": "Estância Velha",
    "faturamento": 55885.24,
    "pedidos": 1644,
    "ticket": 34.0,
    "periodo": "2021-05"
  },
  {
    "mes": 5,
    "ano": 2021,
    "loja": "Campo Bom",
    "faturamento": 44650.24,
    "pedidos": 1313,
    "ticket": 34.0,
    "periodo": "2021-05"
  },
  {
    "mes": 6,
    "ano": 2021,
    "loja": "Montenegro",
    "faturamento": 63582.49,
    "pedidos": 1870,
    "ticket": 34.0,
    "periodo": "2021-06"
  },
  {
    "mes": 6,
    "ano": 2021,
    "loja": "São Leopoldo",
    "faturamento": 62740.71,
    "pedidos": 1845,
    "ticket": 34.0,
    "periodo": "2021-06"
  },
  {
    "mes": 6,
    "ano": 2021,
    "loja": "Sapiranga",
    "faturamento": 56356.27,
    "pedidos": 1658,
    "ticket": 34.0,
    "periodo": "2021-06"
  },
  {
    "mes": 6,
    "ano": 2021,
    "loja": "Campo Bom",
    "faturamento": 41537.05,
    "pedidos": 1222,
    "ticket": 34.0,
    "periodo": "2021-06"
  },
  {
    "mes": 6,
    "ano": 2021,
    "loja": "Parobé",
    "faturamento": 36707.93,
    "pedidos": 1080,
    "ticket": 34.0,
    "periodo": "2021-06"
  },
  {
    "mes": 7,
    "ano": 2021,
    "loja": "Caxias do Sul",
    "faturamento": 122228.36,
    "pedidos": 3595,
    "ticket": 34.0,
    "periodo": "2021-07"
  },
  {
    "mes": 7,
    "ano": 2021,
    "loja": "São Leopoldo",
    "faturamento": 76488.2,
    "pedidos": 2250,
    "ticket": 34.0,
    "periodo": "2021-07"
  },
  {
    "mes": 7,
    "ano": 2021,
    "loja": "Montenegro",
    "faturamento": 67921.59,
    "pedidos": 1998,
    "ticket": 34.0,
    "periodo": "2021-07"
  },
  {
    "mes": 7,
    "ano": 2021,
    "loja": "Sapiranga",
    "faturamento": 65643.8,
    "pedidos": 1931,
    "ticket": 34.0,
    "periodo": "2021-07"
  },
  {
    "mes": 7,
    "ano": 2021,
    "loja": "Taquara",
    "faturamento": 64681.96,
    "pedidos": 1902,
    "ticket": 34.0,
    "periodo": "2021-07"
  },
  {
    "mes": 7,
    "ano": 2021,
    "loja": "Campo Bom",
    "faturamento": 48317.99,
    "pedidos": 1421,
    "ticket": 34.0,
    "periodo": "2021-07"
  },
  {
    "mes": 8,
    "ano": 2021,
    "loja": "Caxias do Sul",
    "faturamento": 113865.7,
    "pedidos": 3349,
    "ticket": 34.0,
    "periodo": "2021-08"
  },
  {
    "mes": 8,
    "ano": 2021,
    "loja": "Novo Hamburgo",
    "faturamento": 79826.82,
    "pedidos": 2348,
    "ticket": 34.0,
    "periodo": "2021-08"
  },
  {
    "mes": 8,
    "ano": 2021,
    "loja": "Campo Bom",
    "faturamento": 51533.65,
    "pedidos": 1516,
    "ticket": 34.0,
    "periodo": "2021-08"
  },
  {
    "mes": 8,
    "ano": 2021,
    "loja": "Parobé",
    "faturamento": 41442.26,
    "pedidos": 1219,
    "ticket": 34.0,
    "periodo": "2021-08"
  },
  {
    "mes": 9,
    "ano": 2021,
    "loja": "Montenegro",
    "faturamento": 68762.53,
    "pedidos": 2022,
    "ticket": 34.0,
    "periodo": "2021-09"
  },
  {
    "mes": 9,
    "ano": 2021,
    "loja": "São Leopoldo",
    "faturamento": 68066.58,
    "pedidos": 2002,
    "ticket": 34.0,
    "periodo": "2021-09"
  },
  {
    "mes": 9,
    "ano": 2021,
    "loja": "Sapiranga",
    "faturamento": 60587.7,
    "pedidos": 1782,
    "ticket": 34.0,
    "periodo": "2021-09"
  },
  {
    "mes": 9,
    "ano": 2021,
    "loja": "Taquara",
    "faturamento": 52138.94,
    "pedidos": 1533,
    "ticket": 34.0,
    "periodo": "2021-09"
  },
  {
    "mes": 9,
    "ano": 2021,
    "loja": "Campo Bom",
    "faturamento": 48589.91,
    "pedidos": 1429,
    "ticket": 34.0,
    "periodo": "2021-09"
  },
  {
    "mes": 10,
    "ano": 2021,
    "loja": "Novo Hamburgo",
    "faturamento": 97430.32,
    "pedidos": 2866,
    "ticket": 34.0,
    "periodo": "2021-10"
  },
  {
    "mes": 10,
    "ano": 2021,
    "loja": "São Leopoldo",
    "faturamento": 86228.9,
    "pedidos": 2536,
    "ticket": 34.0,
    "periodo": "2021-10"
  },
  {
    "mes": 11,
    "ano": 2021,
    "loja": "Novo Hamburgo",
    "faturamento": 128932.78,
    "pedidos": 3792,
    "ticket": 34.0,
    "periodo": "2021-11"
  },
  {
    "mes": 11,
    "ano": 2021,
    "loja": "São Leopoldo",
    "faturamento": 115734.97,
    "pedidos": 3404,
    "ticket": 34.0,
    "periodo": "2021-11"
  },
  {
    "mes": 11,
    "ano": 2021,
    "loja": "Sapiranga",
    "faturamento": 70634.98,
    "pedidos": 2077,
    "ticket": 34.0,
    "periodo": "2021-11"
  },
  {
    "mes": 11,
    "ano": 2021,
    "loja": "Estância Velha",
    "faturamento": 59370.66,
    "pedidos": 1746,
    "ticket": 34.0,
    "periodo": "2021-11"
  },
  {
    "mes": 11,
    "ano": 2021,
    "loja": "Taquara",
    "faturamento": 49200.68,
    "pedidos": 1447,
    "ticket": 34.0,
    "periodo": "2021-11"
  },
  {
    "mes": 11,
    "ano": 2021,
    "loja": "Nova Hartz",
    "faturamento": 30229.82,
    "pedidos": 889,
    "ticket": 34.0,
    "periodo": "2021-11"
  },
  {
    "mes": 12,
    "ano": 2021,
    "loja": "Caxias do Sul",
    "faturamento": 129367.05,
    "pedidos": 3805,
    "ticket": 34.0,
    "periodo": "2021-12"
  },
  {
    "mes": 12,
    "ano": 2021,
    "loja": "Novo Hamburgo",
    "faturamento": 128674.31,
    "pedidos": 3785,
    "ticket": 34.0,
    "periodo": "2021-12"
  },
  {
    "mes": 12,
    "ano": 2021,
    "loja": "São Leopoldo",
    "faturamento": 109954.22,
    "pedidos": 3234,
    "ticket": 34.0,
    "periodo": "2021-12"
  },
  {
    "mes": 12,
    "ano": 2021,
    "loja": "Sapiranga",
    "faturamento": 89459.55,
    "pedidos": 2631,
    "ticket": 34.0,
    "periodo": "2021-12"
  },
  {
    "mes": 12,
    "ano": 2021,
    "loja": "Montenegro",
    "faturamento": 81165.45,
    "pedidos": 2387,
    "ticket": 34.0,
    "periodo": "2021-12"
  },
  {
    "mes": 12,
    "ano": 2021,
    "loja": "Estância Velha",
    "faturamento": 70612.25,
    "pedidos": 2077,
    "ticket": 34.0,
    "periodo": "2021-12"
  },
  {
    "mes": 12,
    "ano": 2021,
    "loja": "Taquara",
    "faturamento": 53985.4,
    "pedidos": 1588,
    "ticket": 34.0,
    "periodo": "2021-12"
  },
  {
    "mes": 12,
    "ano": 2021,
    "loja": "Sapucaia do Sul (Vinicius)",
    "faturamento": 23860.12,
    "pedidos": 702,
    "ticket": 34.0,
    "periodo": "2021-12"
  },
  {
    "mes": 8,
    "ano": 2023,
    "loja": "Campo Bom",
    "faturamento": 48549.15,
    "pedidos": 1426,
    "ticket": 34.05,
    "periodo": "2023-08"
  },
  {
    "mes": 9,
    "ano": 2023,
    "loja": "Shopping Total",
    "faturamento": 64131.8,
    "pedidos": 1895,
    "ticket": 33.84,
    "periodo": "2023-09"
  },
  {
    "mes": 10,
    "ano": 2023,
    "loja": "Sapiranga",
    "faturamento": 49952.92,
    "pedidos": 1468,
    "ticket": 34.03,
    "periodo": "2023-10"
  },
  {
    "mes": 11,
    "ano": 2023,
    "loja": "Shopping Total",
    "faturamento": 75637.84,
    "pedidos": 2208,
    "ticket": 34.26,
    "periodo": "2023-11"
  },
  {
    "mes": 11,
    "ano": 2023,
    "loja": "Campo Bom",
    "faturamento": 38917.01,
    "pedidos": 1130,
    "ticket": 34.44,
    "periodo": "2023-11"
  },
  {
    "mes": 12,
    "ano": 2023,
    "loja": "Shopping Total",
    "faturamento": 70758.83,
    "pedidos": 2066,
    "ticket": 34.25,
    "periodo": "2023-12"
  },
  {
    "mes": 11,
    "ano": 2024,
    "loja": "Porto Alegre Zona Norte",
    "faturamento": 5888.0,
    "pedidos": 173,
    "ticket": 34.03,
    "periodo": "2024-11"
  },
  {
    "mes": 12,
    "ano": 2024,
    "loja": "Vila Mariana",
    "faturamento": 946.95,
    "pedidos": 28,
    "ticket": 33.82,
    "periodo": "2024-12"
  },
  {
    "mes": 8,
    "ano": 2019,
    "loja": "Sapiranga",
    "faturamento": 142772.54,
    "pedidos": 4079,
    "ticket": 35.0,
    "periodo": "2019-08"
  },
  {
    "mes": 10,
    "ano": 2019,
    "loja": "Sapiranga",
    "faturamento": 115841.66,
    "pedidos": 3310,
    "ticket": 35.0,
    "periodo": "2019-10"
  },
  {
    "mes": 11,
    "ano": 2019,
    "loja": "Sapiranga",
    "faturamento": 115157.85,
    "pedidos": 3290,
    "ticket": 35.0,
    "periodo": "2019-11"
  },
  {
    "mes": 3,
    "ano": 2020,
    "loja": "Sapiranga",
    "faturamento": 127734.57,
    "pedidos": 3650,
    "ticket": 35.0,
    "periodo": "2020-03"
  },
  {
    "mes": 3,
    "ano": 2020,
    "loja": "Campo Bom",
    "faturamento": 107065.0,
    "pedidos": 3059,
    "ticket": 35.0,
    "periodo": "2020-03"
  },
  {
    "mes": 3,
    "ano": 2020,
    "loja": "Novo Hamburgo",
    "faturamento": 57214.54,
    "pedidos": 1635,
    "ticket": 35.0,
    "periodo": "2020-03"
  },
  {
    "mes": 4,
    "ano": 2020,
    "loja": "Sapiranga",
    "faturamento": 88876.1,
    "pedidos": 2539,
    "ticket": 35.0,
    "periodo": "2020-04"
  },
  {
    "mes": 4,
    "ano": 2020,
    "loja": "Novo Hamburgo",
    "faturamento": 8185.09,
    "pedidos": 234,
    "ticket": 35.0,
    "periodo": "2020-04"
  },
  {
    "mes": 6,
    "ano": 2020,
    "loja": "Sapiranga",
    "faturamento": 94462.46,
    "pedidos": 2699,
    "ticket": 35.0,
    "periodo": "2020-06"
  },
  {
    "mes": 6,
    "ano": 2020,
    "loja": "Campo Bom",
    "faturamento": 71073.6,
    "pedidos": 2031,
    "ticket": 35.0,
    "periodo": "2020-06"
  },
  {
    "mes": 6,
    "ano": 2020,
    "loja": "Taquara",
    "faturamento": 54698.24,
    "pedidos": 1563,
    "ticket": 35.0,
    "periodo": "2020-06"
  },
  {
    "mes": 7,
    "ano": 2020,
    "loja": "Sapiranga",
    "faturamento": 121005.43,
    "pedidos": 3457,
    "ticket": 35.0,
    "periodo": "2020-07"
  },
  {
    "mes": 7,
    "ano": 2020,
    "loja": "Novo Hamburgo",
    "faturamento": 98495.06,
    "pedidos": 2814,
    "ticket": 35.0,
    "periodo": "2020-07"
  },
  {
    "mes": 7,
    "ano": 2020,
    "loja": "Taquara",
    "faturamento": 78443.9,
    "pedidos": 2241,
    "ticket": 35.0,
    "periodo": "2020-07"
  },
  {
    "mes": 7,
    "ano": 2020,
    "loja": "Campo Bom",
    "faturamento": 63444.13,
    "pedidos": 1813,
    "ticket": 35.0,
    "periodo": "2020-07"
  },
  {
    "mes": 8,
    "ano": 2020,
    "loja": "Novo Hamburgo",
    "faturamento": 117013.71,
    "pedidos": 3343,
    "ticket": 35.0,
    "periodo": "2020-08"
  },
  {
    "mes": 8,
    "ano": 2020,
    "loja": "Campo Bom",
    "faturamento": 81349.83,
    "pedidos": 2324,
    "ticket": 35.0,
    "periodo": "2020-08"
  },
  {
    "mes": 9,
    "ano": 2020,
    "loja": "Novo Hamburgo",
    "faturamento": 126172.13,
    "pedidos": 3605,
    "ticket": 35.0,
    "periodo": "2020-09"
  },
  {
    "mes": 9,
    "ano": 2020,
    "loja": "Sapiranga",
    "faturamento": 108969.09,
    "pedidos": 3113,
    "ticket": 35.0,
    "periodo": "2020-09"
  },
  {
    "mes": 9,
    "ano": 2020,
    "loja": "Taquara",
    "faturamento": 76933.95,
    "pedidos": 2198,
    "ticket": 35.0,
    "periodo": "2020-09"
  },
  {
    "mes": 10,
    "ano": 2020,
    "loja": "Sapiranga",
    "faturamento": 97860.92,
    "pedidos": 2796,
    "ticket": 35.0,
    "periodo": "2020-10"
  },
  {
    "mes": 10,
    "ano": 2020,
    "loja": "Montenegro",
    "faturamento": 91010.93,
    "pedidos": 2600,
    "ticket": 35.0,
    "periodo": "2020-10"
  },
  {
    "mes": 10,
    "ano": 2020,
    "loja": "Taquara",
    "faturamento": 64971.7,
    "pedidos": 1856,
    "ticket": 35.0,
    "periodo": "2020-10"
  },
  {
    "mes": 11,
    "ano": 2020,
    "loja": "Novo Hamburgo",
    "faturamento": 115220.85,
    "pedidos": 3292,
    "ticket": 35.0,
    "periodo": "2020-11"
  },
  {
    "mes": 11,
    "ano": 2020,
    "loja": "Campo Bom",
    "faturamento": 65783.34,
    "pedidos": 1880,
    "ticket": 35.0,
    "periodo": "2020-11"
  },
  {
    "mes": 12,
    "ano": 2020,
    "loja": "Novo Hamburgo",
    "faturamento": 125264.46,
    "pedidos": 3579,
    "ticket": 35.0,
    "periodo": "2020-12"
  },
  {
    "mes": 12,
    "ano": 2020,
    "loja": "Montenegro",
    "faturamento": 110912.25,
    "pedidos": 3169,
    "ticket": 35.0,
    "periodo": "2020-12"
  },
  {
    "mes": 12,
    "ano": 2020,
    "loja": "Sapiranga",
    "faturamento": 96301.48,
    "pedidos": 2751,
    "ticket": 35.0,
    "periodo": "2020-12"
  },
  {
    "mes": 12,
    "ano": 2020,
    "loja": "Taquara",
    "faturamento": 83102.13,
    "pedidos": 2374,
    "ticket": 35.0,
    "periodo": "2020-12"
  },
  {
    "mes": 1,
    "ano": 2021,
    "loja": "Novo Hamburgo",
    "faturamento": 121668.76,
    "pedidos": 3476,
    "ticket": 35.0,
    "periodo": "2021-01"
  },
  {
    "mes": 1,
    "ano": 2021,
    "loja": "Montenegro",
    "faturamento": 103530.47,
    "pedidos": 2958,
    "ticket": 35.0,
    "periodo": "2021-01"
  },
  {
    "mes": 1,
    "ano": 2021,
    "loja": "Taquara",
    "faturamento": 75854.0,
    "pedidos": 2167,
    "ticket": 35.0,
    "periodo": "2021-01"
  },
  {
    "mes": 1,
    "ano": 2021,
    "loja": "Caxias do Sul",
    "faturamento": 55712.62,
    "pedidos": 1592,
    "ticket": 35.0,
    "periodo": "2021-01"
  },
  {
    "mes": 2,
    "ano": 2021,
    "loja": "Taquara",
    "faturamento": 60524.75,
    "pedidos": 1729,
    "ticket": 35.0,
    "periodo": "2021-02"
  },
  {
    "mes": 2,
    "ano": 2021,
    "loja": "Sapiranga",
    "faturamento": 56336.05,
    "pedidos": 1610,
    "ticket": 35.0,
    "periodo": "2021-02"
  },
  {
    "mes": 2,
    "ano": 2021,
    "loja": "Campo Bom",
    "faturamento": 50545.59,
    "pedidos": 1444,
    "ticket": 35.0,
    "periodo": "2021-02"
  },
  {
    "mes": 3,
    "ano": 2021,
    "loja": "Novo Hamburgo",
    "faturamento": 100453.2,
    "pedidos": 2870,
    "ticket": 35.0,
    "periodo": "2021-03"
  },
  {
    "mes": 3,
    "ano": 2021,
    "loja": "Estância Velha",
    "faturamento": 60834.49,
    "pedidos": 1738,
    "ticket": 35.0,
    "periodo": "2021-03"
  },
  {
    "mes": 3,
    "ano": 2021,
    "loja": "Taquara",
    "faturamento": 57260.6,
    "pedidos": 1636,
    "ticket": 35.0,
    "periodo": "2021-03"
  },
  {
    "mes": 3,
    "ano": 2021,
    "loja": "Sapiranga",
    "faturamento": 53401.1,
    "pedidos": 1526,
    "ticket": 35.0,
    "periodo": "2021-03"
  },
  {
    "mes": 4,
    "ano": 2021,
    "loja": "Caxias do Sul",
    "faturamento": 80413.13,
    "pedidos": 2298,
    "ticket": 35.0,
    "periodo": "2021-04"
  },
  {
    "mes": 4,
    "ano": 2021,
    "loja": "Montenegro",
    "faturamento": 76649.17,
    "pedidos": 2190,
    "ticket": 35.0,
    "periodo": "2021-04"
  },
  {
    "mes": 4,
    "ano": 2021,
    "loja": "Sapiranga",
    "faturamento": 50044.07,
    "pedidos": 1430,
    "ticket": 35.0,
    "periodo": "2021-04"
  },
  {
    "mes": 4,
    "ano": 2021,
    "loja": "Taquara",
    "faturamento": 48176.58,
    "pedidos": 1376,
    "ticket": 35.0,
    "periodo": "2021-04"
  },
  {
    "mes": 4,
    "ano": 2021,
    "loja": "Campo Bom",
    "faturamento": 38543.65,
    "pedidos": 1101,
    "ticket": 35.0,
    "periodo": "2021-04"
  },
  {
    "mes": 4,
    "ano": 2021,
    "loja": "São Leopoldo",
    "faturamento": 35792.76,
    "pedidos": 1023,
    "ticket": 35.0,
    "periodo": "2021-04"
  },
  {
    "mes": 5,
    "ano": 2021,
    "loja": "Montenegro",
    "faturamento": 70634.88,
    "pedidos": 2018,
    "ticket": 35.0,
    "periodo": "2021-05"
  },
  {
    "mes": 5,
    "ano": 2021,
    "loja": "Taquara",
    "faturamento": 47152.73,
    "pedidos": 1347,
    "ticket": 35.0,
    "periodo": "2021-05"
  },
  {
    "mes": 5,
    "ano": 2021,
    "loja": "Parobé",
    "faturamento": 43594.45,
    "pedidos": 1246,
    "ticket": 35.0,
    "periodo": "2021-05"
  },
  {
    "mes": 6,
    "ano": 2021,
    "loja": "Caxias do Sul",
    "faturamento": 106367.28,
    "pedidos": 3039,
    "ticket": 35.0,
    "periodo": "2021-06"
  },
  {
    "mes": 6,
    "ano": 2021,
    "loja": "Novo Hamburgo",
    "faturamento": 95885.6,
    "pedidos": 2740,
    "ticket": 35.0,
    "periodo": "2021-06"
  },
  {
    "mes": 6,
    "ano": 2021,
    "loja": "Estância Velha",
    "faturamento": 54633.36,
    "pedidos": 1561,
    "ticket": 35.0,
    "periodo": "2021-06"
  },
  {
    "mes": 6,
    "ano": 2021,
    "loja": "Taquara",
    "faturamento": 48281.96,
    "pedidos": 1379,
    "ticket": 35.0,
    "periodo": "2021-06"
  },
  {
    "mes": 7,
    "ano": 2021,
    "loja": "Novo Hamburgo",
    "faturamento": 102538.92,
    "pedidos": 2930,
    "ticket": 35.0,
    "periodo": "2021-07"
  },
  {
    "mes": 7,
    "ano": 2021,
    "loja": "Estância Velha",
    "faturamento": 61813.26,
    "pedidos": 1766,
    "ticket": 35.0,
    "periodo": "2021-07"
  },
  {
    "mes": 7,
    "ano": 2021,
    "loja": "Parobé",
    "faturamento": 50531.31,
    "pedidos": 1444,
    "ticket": 35.0,
    "periodo": "2021-07"
  },
  {
    "mes": 8,
    "ano": 2021,
    "loja": "São Leopoldo",
    "faturamento": 75557.68,
    "pedidos": 2159,
    "ticket": 35.0,
    "periodo": "2021-08"
  },
  {
    "mes": 8,
    "ano": 2021,
    "loja": "Montenegro",
    "faturamento": 71824.14,
    "pedidos": 2052,
    "ticket": 35.0,
    "periodo": "2021-08"
  },
  {
    "mes": 8,
    "ano": 2021,
    "loja": "Estância Velha",
    "faturamento": 66495.99,
    "pedidos": 1900,
    "ticket": 35.0,
    "periodo": "2021-08"
  },
  {
    "mes": 8,
    "ano": 2021,
    "loja": "Sapiranga",
    "faturamento": 58241.67,
    "pedidos": 1664,
    "ticket": 35.0,
    "periodo": "2021-08"
  },
  {
    "mes": 8,
    "ano": 2021,
    "loja": "Taquara",
    "faturamento": 52457.07,
    "pedidos": 1499,
    "ticket": 35.0,
    "periodo": "2021-08"
  },
  {
    "mes": 9,
    "ano": 2021,
    "loja": "Caxias do Sul",
    "faturamento": 116732.15,
    "pedidos": 3335,
    "ticket": 35.0,
    "periodo": "2021-09"
  },
  {
    "mes": 9,
    "ano": 2021,
    "loja": "Novo Hamburgo",
    "faturamento": 84039.94,
    "pedidos": 2401,
    "ticket": 35.0,
    "periodo": "2021-09"
  },
  {
    "mes": 9,
    "ano": 2021,
    "loja": "Estância Velha",
    "faturamento": 55907.89,
    "pedidos": 1597,
    "ticket": 35.0,
    "periodo": "2021-09"
  },
  {
    "mes": 9,
    "ano": 2021,
    "loja": "Parobé",
    "faturamento": 37249.98,
    "pedidos": 1064,
    "ticket": 35.0,
    "periodo": "2021-09"
  },
  {
    "mes": 10,
    "ano": 2021,
    "loja": "Caxias do Sul",
    "faturamento": 125607.1,
    "pedidos": 3589,
    "ticket": 35.0,
    "periodo": "2021-10"
  },
  {
    "mes": 10,
    "ano": 2021,
    "loja": "Montenegro",
    "faturamento": 76242.56,
    "pedidos": 2178,
    "ticket": 35.0,
    "periodo": "2021-10"
  },
  {
    "mes": 10,
    "ano": 2021,
    "loja": "Sapiranga",
    "faturamento": 61765.38,
    "pedidos": 1765,
    "ticket": 35.0,
    "periodo": "2021-10"
  },
  {
    "mes": 10,
    "ano": 2021,
    "loja": "Estância Velha",
    "faturamento": 61110.82,
    "pedidos": 1746,
    "ticket": 35.0,
    "periodo": "2021-10"
  },
  {
    "mes": 10,
    "ano": 2021,
    "loja": "Campo Bom",
    "faturamento": 56334.86,
    "pedidos": 1610,
    "ticket": 35.0,
    "periodo": "2021-10"
  },
  {
    "mes": 10,
    "ano": 2021,
    "loja": "Taquara",
    "faturamento": 55475.49,
    "pedidos": 1585,
    "ticket": 35.0,
    "periodo": "2021-10"
  },
  {
    "mes": 10,
    "ano": 2021,
    "loja": "Nova Hartz",
    "faturamento": 54020.31,
    "pedidos": 1543,
    "ticket": 35.0,
    "periodo": "2021-10"
  },
  {
    "mes": 10,
    "ano": 2021,
    "loja": "Parobé",
    "faturamento": 38801.0,
    "pedidos": 1109,
    "ticket": 35.0,
    "periodo": "2021-10"
  },
  {
    "mes": 11,
    "ano": 2021,
    "loja": "Caxias do Sul",
    "faturamento": 112935.1,
    "pedidos": 3227,
    "ticket": 35.0,
    "periodo": "2021-11"
  },
  {
    "mes": 11,
    "ano": 2021,
    "loja": "Montenegro",
    "faturamento": 75866.31,
    "pedidos": 2168,
    "ticket": 35.0,
    "periodo": "2021-11"
  },
  {
    "mes": 11,
    "ano": 2021,
    "loja": "Campo Bom",
    "faturamento": 59256.74,
    "pedidos": 1693,
    "ticket": 35.0,
    "periodo": "2021-11"
  },
  {
    "mes": 11,
    "ano": 2021,
    "loja": "Parobé",
    "faturamento": 40369.96,
    "pedidos": 1153,
    "ticket": 35.0,
    "periodo": "2021-11"
  },
  {
    "mes": 12,
    "ano": 2021,
    "loja": "Campo Bom",
    "faturamento": 76737.74,
    "pedidos": 2193,
    "ticket": 35.0,
    "periodo": "2021-12"
  },
  {
    "mes": 12,
    "ano": 2021,
    "loja": "Parobé",
    "faturamento": 44442.81,
    "pedidos": 1270,
    "ticket": 35.0,
    "periodo": "2021-12"
  },
  {
    "mes": 12,
    "ano": 2021,
    "loja": "Nova Hartz",
    "faturamento": 27450.25,
    "pedidos": 784,
    "ticket": 35.0,
    "periodo": "2021-12"
  },
  {
    "mes": 9,
    "ano": 2023,
    "loja": "Campo Bom",
    "faturamento": 53519.5,
    "pedidos": 1535,
    "ticket": 34.87,
    "periodo": "2023-09"
  },
  {
    "mes": 9,
    "ano": 2024,
    "loja": "Vila Mariana",
    "faturamento": 14469.71,
    "pedidos": 408,
    "ticket": 35.46,
    "periodo": "2024-09"
  },
  {
    "mes": 10,
    "ano": 2024,
    "loja": "Vila Mariana",
    "faturamento": 12812.21,
    "pedidos": 363,
    "ticket": 35.3,
    "periodo": "2024-10"
  },
  {
    "mes": 11,
    "ano": 2024,
    "loja": "Vila Mariana",
    "faturamento": 9373.41,
    "pedidos": 271,
    "ticket": 34.59,
    "periodo": "2024-11"
  },
  {
    "mes": 11,
    "ano": 2019,
    "loja": "Campo Bom",
    "faturamento": 96558.8,
    "pedidos": 2682,
    "ticket": 36.0,
    "periodo": "2019-11"
  },
  {
    "mes": 1,
    "ano": 2020,
    "loja": "Campo Bom",
    "faturamento": 108411.9,
    "pedidos": 2853,
    "ticket": 38.0,
    "periodo": "2020-01"
  },
  {
    "mes": 2,
    "ano": 2020,
    "loja": "Campo Bom",
    "faturamento": 109765.05,
    "pedidos": 2889,
    "ticket": 38.0,
    "periodo": "2020-02"
  },
  {
    "mes": 6,
    "ano": 2023,
    "loja": "Estância Velha",
    "faturamento": 37345.03,
    "pedidos": 258,
    "ticket": 144.75,
    "periodo": "2023-06"
  },
  {
    "mes": 7,
    "ano": 2023,
    "loja": "Shopping Total",
    "faturamento": 77424.34,
    "pedidos": 1454,
    "ticket": 53.25,
    "periodo": "2023-07"
  },
  {
    "mes": 7,
    "ano": 2023,
    "loja": "Estância Velha",
    "faturamento": 35667.05,
    "pedidos": 729,
    "ticket": 48.93,
    "periodo": "2023-07"
  },
  {
    "mes": 8,
    "ano": 2023,
    "loja": "Shopping Total",
    "faturamento": 63398.44,
    "pedidos": 1949,
    "ticket": 32.53,
    "periodo": "2023-08"
  },
  {
    "mes": 8,
    "ano": 2023,
    "loja": "Estância Velha",
    "faturamento": 36320.73,
    "pedidos": 709,
    "ticket": 51.23,
    "periodo": "2023-08"
  },
  {
    "mes": 8,
    "ano": 2023,
    "loja": "Canoas Mathias",
    "faturamento": 23597.15,
    "pedidos": 616,
    "ticket": 38.31,
    "periodo": "2023-08"
  },
  {
    "mes": 9,
    "ano": 2023,
    "loja": "Estância Velha",
    "faturamento": 31673.07,
    "pedidos": 599,
    "ticket": 52.88,
    "periodo": "2023-09"
  },
  {
    "mes": 10,
    "ano": 2023,
    "loja": "Shopping Total",
    "faturamento": 67732.27,
    "pedidos": 2065,
    "ticket": 32.8,
    "periodo": "2023-10"
  },
  {
    "mes": 10,
    "ano": 2023,
    "loja": "Campo Bom",
    "faturamento": 49065.7,
    "pedidos": 1312,
    "ticket": 37.4,
    "periodo": "2023-10"
  },
  {
    "mes": 10,
    "ano": 2023,
    "loja": "Canoas Mathias",
    "faturamento": 31515.35,
    "pedidos": 853,
    "ticket": 36.95,
    "periodo": "2023-10"
  },
  {
    "mes": 10,
    "ano": 2023,
    "loja": "Estância Velha",
    "faturamento": 29393.68,
    "pedidos": 523,
    "ticket": 56.2,
    "periodo": "2023-10"
  },
  {
    "mes": 11,
    "ano": 2023,
    "loja": "Canoas Mathias",
    "faturamento": 32376.32,
    "pedidos": 892,
    "ticket": 36.3,
    "periodo": "2023-11"
  },
  {
    "mes": 11,
    "ano": 2023,
    "loja": "Estância Velha",
    "faturamento": 27534.78,
    "pedidos": 502,
    "ticket": 54.85,
    "periodo": "2023-11"
  },
  {
    "mes": 12,
    "ano": 2023,
    "loja": "Estância Velha",
    "faturamento": 30766.13,
    "pedidos": 573,
    "ticket": 53.69,
    "periodo": "2023-12"
  },
  {
    "mes": 1,
    "ano": 2024,
    "loja": "Shopping Total",
    "faturamento": 64336.66,
    "pedidos": 3790,
    "ticket": 16.98,
    "periodo": "2024-01"
  },
  {
    "mes": 1,
    "ano": 2024,
    "loja": "Campo Bom",
    "faturamento": 46038.12,
    "pedidos": 1209,
    "ticket": 38.08,
    "periodo": "2024-01"
  },
  {
    "mes": 1,
    "ano": 2024,
    "loja": "Estância Velha",
    "faturamento": 28495.57,
    "pedidos": 546,
    "ticket": 52.19,
    "periodo": "2024-01"
  },
  {
    "mes": 2,
    "ano": 2024,
    "loja": "Estância Velha",
    "faturamento": 27780.44,
    "pedidos": 1643,
    "ticket": 16.91,
    "periodo": "2024-02"
  },
  {
    "mes": 2,
    "ano": 2024,
    "loja": "Canoas Mathias",
    "faturamento": 25615.22,
    "pedidos": 629,
    "ticket": 40.72,
    "periodo": "2024-02"
  },
  {
    "mes": 3,
    "ano": 2024,
    "loja": "Campo Bom",
    "faturamento": 44037.47,
    "pedidos": 1004,
    "ticket": 43.86,
    "periodo": "2024-03"
  },
  {
    "mes": 3,
    "ano": 2024,
    "loja": "Canoas Mathias",
    "faturamento": 30717.53,
    "pedidos": 720,
    "ticket": 42.66,
    "periodo": "2024-03"
  },
  {
    "mes": 3,
    "ano": 2024,
    "loja": "Estância Velha",
    "faturamento": 29860.34,
    "pedidos": 589,
    "ticket": 50.7,
    "periodo": "2024-03"
  },
  {
    "mes": 4,
    "ano": 2024,
    "loja": "Canoas Mathias",
    "faturamento": 29395.57,
    "pedidos": 666,
    "ticket": 44.14,
    "periodo": "2024-04"
  },
  {
    "mes": 4,
    "ano": 2024,
    "loja": "Estância Velha",
    "faturamento": 26239.64,
    "pedidos": 518,
    "ticket": 50.66,
    "periodo": "2024-04"
  },
  {
    "mes": 5,
    "ano": 2024,
    "loja": "Campo Bom",
    "faturamento": 48029.05,
    "pedidos": 1107,
    "ticket": 43.39,
    "periodo": "2024-05"
  },
  {
    "mes": 5,
    "ano": 2024,
    "loja": "Estância Velha",
    "faturamento": 22452.5,
    "pedidos": 369,
    "ticket": 60.85,
    "periodo": "2024-05"
  },
  {
    "mes": 5,
    "ano": 2024,
    "loja": "Canoas Mathias",
    "faturamento": 2991.05,
    "pedidos": 68,
    "ticket": 43.99,
    "periodo": "2024-05"
  },
  {
    "mes": 6,
    "ano": 2024,
    "loja": "Campo Bom",
    "faturamento": 49242.99,
    "pedidos": 1118,
    "ticket": 44.05,
    "periodo": "2024-06"
  },
  {
    "mes": 6,
    "ano": 2024,
    "loja": "Estância Velha",
    "faturamento": 15281.83,
    "pedidos": 256,
    "ticket": 59.69,
    "periodo": "2024-06"
  },
  {
    "mes": 7,
    "ano": 2024,
    "loja": "Campo Bom",
    "faturamento": 43562.42,
    "pedidos": 1072,
    "ticket": 40.64,
    "periodo": "2024-07"
  },
  {
    "mes": 7,
    "ano": 2024,
    "loja": "Vila Mariana",
    "faturamento": 7986.61,
    "pedidos": 205,
    "ticket": 38.96,
    "periodo": "2024-07"
  },
  {
    "mes": 8,
    "ano": 2024,
    "loja": "Campo Bom",
    "faturamento": 46952.66,
    "pedidos": 1149,
    "ticket": 40.86,
    "periodo": "2024-08"
  },
  {
    "mes": 8,
    "ano": 2024,
    "loja": "Vila Mariana",
    "faturamento": 15800.2,
    "pedidos": 516,
    "ticket": 30.62,
    "periodo": "2024-08"
  },
  {
    "mes": 10,
    "ano": 2024,
    "loja": "Campo Bom",
    "faturamento": 28810.57,
    "pedidos": 680,
    "ticket": 42.37,
    "periodo": "2024-10"
  },
  {
    "mes": 1,
    "ano": 2020,
    "loja": "Sapiranga",
    "faturamento": 127060.67,
    "pedidos": 3344,
    "ticket": 38.0,
    "periodo": "2020-01"
  },
  {
    "mes": 1,
    "ano": 2020,
    "loja": "Novo Hamburgo",
    "faturamento": 89000.4,
    "pedidos": 2342,
    "ticket": 38.0,
    "periodo": "2020-01"
  },
  {
    "mes": 2,
    "ano": 2020,
    "loja": "Novo Hamburgo",
    "faturamento": 85490.36,
    "pedidos": 2250,
    "ticket": 38.0,
    "periodo": "2020-02"
  },
  {
    "mes": 5,
    "ano": 2023,
    "loja": "Montenegro",
    "faturamento": 70944.42,
    "pedidos": 444,
    "ticket": 159.78,
    "periodo": "2023-05"
  },
  {
    "mes": 6,
    "ano": 2023,
    "loja": "Montenegro",
    "faturamento": 65804.52,
    "pedidos": 1342,
    "ticket": 49.03,
    "periodo": "2023-06"
  },
  {
    "mes": 6,
    "ano": 2023,
    "loja": "Canoas",
    "faturamento": 45835.29,
    "pedidos": 436,
    "ticket": 105.13,
    "periodo": "2023-06"
  },
  {
    "mes": 6,
    "ano": 2023,
    "loja": "Zona Norte",
    "faturamento": 608.49,
    "pedidos": 16,
    "ticket": 38.03,
    "periodo": "2023-06"
  },
  {
    "mes": 7,
    "ano": 2023,
    "loja": "Barra Shopping",
    "faturamento": 114585.33,
    "pedidos": 2678,
    "ticket": 42.79,
    "periodo": "2023-07"
  },
  {
    "mes": 7,
    "ano": 2023,
    "loja": "Novo Hamburgo",
    "faturamento": 110336.19,
    "pedidos": 1398,
    "ticket": 78.92,
    "periodo": "2023-07"
  },
  {
    "mes": 7,
    "ano": 2023,
    "loja": "São Leopoldo",
    "faturamento": 94551.1,
    "pedidos": 1258,
    "ticket": 75.16,
    "periodo": "2023-07"
  },
  {
    "mes": 7,
    "ano": 2023,
    "loja": "Montenegro",
    "faturamento": 73669.81,
    "pedidos": 1475,
    "ticket": 49.95,
    "periodo": "2023-07"
  },
  {
    "mes": 7,
    "ano": 2023,
    "loja": "Sapiranga",
    "faturamento": 55605.78,
    "pedidos": 685,
    "ticket": 81.18,
    "periodo": "2023-07"
  },
  {
    "mes": 7,
    "ano": 2023,
    "loja": "Canoas",
    "faturamento": 46610.68,
    "pedidos": 1140,
    "ticket": 40.89,
    "periodo": "2023-07"
  },
  {
    "mes": 7,
    "ano": 2023,
    "loja": "Zona Norte",
    "faturamento": 8096.47,
    "pedidos": 215,
    "ticket": 37.66,
    "periodo": "2023-07"
  },
  {
    "mes": 8,
    "ano": 2023,
    "loja": "Caxias do Sul",
    "faturamento": 124636.95,
    "pedidos": 2034,
    "ticket": 61.28,
    "periodo": "2023-08"
  },
  {
    "mes": 8,
    "ano": 2023,
    "loja": "Novo Hamburgo",
    "faturamento": 112562.21,
    "pedidos": 2298,
    "ticket": 48.98,
    "periodo": "2023-08"
  },
  {
    "mes": 8,
    "ano": 2023,
    "loja": "Barra Shopping",
    "faturamento": 90829.42,
    "pedidos": 3108,
    "ticket": 29.22,
    "periodo": "2023-08"
  },
  {
    "mes": 8,
    "ano": 2023,
    "loja": "São Leopoldo",
    "faturamento": 83582.26,
    "pedidos": 1763,
    "ticket": 47.41,
    "periodo": "2023-08"
  },
  {
    "mes": 8,
    "ano": 2023,
    "loja": "Montenegro",
    "faturamento": 64729.04,
    "pedidos": 1367,
    "ticket": 47.35,
    "periodo": "2023-08"
  },
  {
    "mes": 8,
    "ano": 2023,
    "loja": "Canoas",
    "faturamento": 48631.22,
    "pedidos": 1269,
    "ticket": 38.32,
    "periodo": "2023-08"
  },
  {
    "mes": 8,
    "ano": 2023,
    "loja": "Bento Gonçalves",
    "faturamento": 48074.8,
    "pedidos": 669,
    "ticket": 71.86,
    "periodo": "2023-08"
  },
  {
    "mes": 8,
    "ano": 2023,
    "loja": "Sapiranga",
    "faturamento": 44810.23,
    "pedidos": 1239,
    "ticket": 36.17,
    "periodo": "2023-08"
  },
  {
    "mes": 8,
    "ano": 2023,
    "loja": "Gravataí",
    "faturamento": 15552.29,
    "pedidos": 335,
    "ticket": 46.42,
    "periodo": "2023-08"
  },
  {
    "mes": 8,
    "ano": 2023,
    "loja": "Zona Norte",
    "faturamento": 8596.64,
    "pedidos": 196,
    "ticket": 43.86,
    "periodo": "2023-08"
  },
  {
    "mes": 9,
    "ano": 2023,
    "loja": "Caxias do Sul",
    "faturamento": 139221.51,
    "pedidos": 2736,
    "ticket": 50.89,
    "periodo": "2023-09"
  },
  {
    "mes": 9,
    "ano": 2023,
    "loja": "Novo Hamburgo",
    "faturamento": 110366.61,
    "pedidos": 2171,
    "ticket": 50.84,
    "periodo": "2023-09"
  },
  {
    "mes": 9,
    "ano": 2023,
    "loja": "Barra Shopping",
    "faturamento": 99475.46,
    "pedidos": 3345,
    "ticket": 29.74,
    "periodo": "2023-09"
  },
  {
    "mes": 9,
    "ano": 2023,
    "loja": "São Leopoldo",
    "faturamento": 81491.4,
    "pedidos": 1693,
    "ticket": 48.13,
    "periodo": "2023-09"
  },
  {
    "mes": 9,
    "ano": 2023,
    "loja": "Montenegro",
    "faturamento": 65679.95,
    "pedidos": 1342,
    "ticket": 48.94,
    "periodo": "2023-09"
  },
  {
    "mes": 9,
    "ano": 2023,
    "loja": "Sapiranga",
    "faturamento": 54991.94,
    "pedidos": 1529,
    "ticket": 35.97,
    "periodo": "2023-09"
  },
  {
    "mes": 9,
    "ano": 2023,
    "loja": "Canoas",
    "faturamento": 54944.45,
    "pedidos": 1445,
    "ticket": 38.02,
    "periodo": "2023-09"
  },
  {
    "mes": 9,
    "ano": 2023,
    "loja": "Bento Gonçalves",
    "faturamento": 51416.87,
    "pedidos": 1084,
    "ticket": 47.43,
    "periodo": "2023-09"
  },
  {
    "mes": 9,
    "ano": 2023,
    "loja": "Gravataí",
    "faturamento": 21606.87,
    "pedidos": 419,
    "ticket": 51.57,
    "periodo": "2023-09"
  },
  {
    "mes": 9,
    "ano": 2023,
    "loja": "Zona Norte",
    "faturamento": 4223.71,
    "pedidos": 89,
    "ticket": 47.46,
    "periodo": "2023-09"
  },
  {
    "mes": 10,
    "ano": 2023,
    "loja": "Caxias do Sul",
    "faturamento": 122690.3,
    "pedidos": 2451,
    "ticket": 50.06,
    "periodo": "2023-10"
  },
  {
    "mes": 10,
    "ano": 2023,
    "loja": "Barra Shopping",
    "faturamento": 114033.42,
    "pedidos": 3795,
    "ticket": 30.05,
    "periodo": "2023-10"
  },
  {
    "mes": 10,
    "ano": 2023,
    "loja": "Novo Hamburgo",
    "faturamento": 95946.27,
    "pedidos": 1955,
    "ticket": 49.08,
    "periodo": "2023-10"
  },
  {
    "mes": 10,
    "ano": 2023,
    "loja": "São Leopoldo",
    "faturamento": 77013.91,
    "pedidos": 1666,
    "ticket": 46.23,
    "periodo": "2023-10"
  },
  {
    "mes": 10,
    "ano": 2023,
    "loja": "Montenegro",
    "faturamento": 68491.63,
    "pedidos": 1432,
    "ticket": 47.83,
    "periodo": "2023-10"
  },
  {
    "mes": 10,
    "ano": 2023,
    "loja": "Canoas",
    "faturamento": 51139.92,
    "pedidos": 1364,
    "ticket": 37.49,
    "periodo": "2023-10"
  },
  {
    "mes": 10,
    "ano": 2023,
    "loja": "Bento Gonçalves",
    "faturamento": 50399.27,
    "pedidos": 1050,
    "ticket": 48.0,
    "periodo": "2023-10"
  },
  {
    "mes": 10,
    "ano": 2023,
    "loja": "Gravataí",
    "faturamento": 21609.45,
    "pedidos": 447,
    "ticket": 48.34,
    "periodo": "2023-10"
  },
  {
    "mes": 10,
    "ano": 2023,
    "loja": "Lajeado",
    "faturamento": 15705.14,
    "pedidos": 326,
    "ticket": 48.18,
    "periodo": "2023-10"
  },
  {
    "mes": 10,
    "ano": 2023,
    "loja": "Zona Norte",
    "faturamento": 5836.88,
    "pedidos": 118,
    "ticket": 49.47,
    "periodo": "2023-10"
  },
  {
    "mes": 11,
    "ano": 2023,
    "loja": "Caxias do Sul",
    "faturamento": 135853.22,
    "pedidos": 2611,
    "ticket": 52.03,
    "periodo": "2023-11"
  },
  {
    "mes": 11,
    "ano": 2023,
    "loja": "Barra Shopping",
    "faturamento": 133914.34,
    "pedidos": 4301,
    "ticket": 31.14,
    "periodo": "2023-11"
  },
  {
    "mes": 11,
    "ano": 2023,
    "loja": "Novo Hamburgo",
    "faturamento": 107238.45,
    "pedidos": 2139,
    "ticket": 50.13,
    "periodo": "2023-11"
  },
  {
    "mes": 11,
    "ano": 2023,
    "loja": "São Leopoldo",
    "faturamento": 84079.28,
    "pedidos": 1768,
    "ticket": 47.56,
    "periodo": "2023-11"
  },
  {
    "mes": 11,
    "ano": 2023,
    "loja": "Lajeado",
    "faturamento": 78457.26,
    "pedidos": 1565,
    "ticket": 50.13,
    "periodo": "2023-11"
  },
  {
    "mes": 11,
    "ano": 2023,
    "loja": "Bento Gonçalves",
    "faturamento": 66524.61,
    "pedidos": 1330,
    "ticket": 50.02,
    "periodo": "2023-11"
  },
  {
    "mes": 11,
    "ano": 2023,
    "loja": "Montenegro",
    "faturamento": 62276.82,
    "pedidos": 1267,
    "ticket": 49.15,
    "periodo": "2023-11"
  },
  {
    "mes": 11,
    "ano": 2023,
    "loja": "Canoas",
    "faturamento": 53223.4,
    "pedidos": 1459,
    "ticket": 36.48,
    "periodo": "2023-11"
  },
  {
    "mes": 11,
    "ano": 2023,
    "loja": "Sapiranga",
    "faturamento": 49651.62,
    "pedidos": 1393,
    "ticket": 35.64,
    "periodo": "2023-11"
  },
  {
    "mes": 11,
    "ano": 2023,
    "loja": "Gravataí",
    "faturamento": 23741.8,
    "pedidos": 508,
    "ticket": 46.74,
    "periodo": "2023-11"
  },
  {
    "mes": 11,
    "ano": 2023,
    "loja": "Zona Norte",
    "faturamento": 5873.75,
    "pedidos": 137,
    "ticket": 42.87,
    "periodo": "2023-11"
  },
  {
    "mes": 12,
    "ano": 2023,
    "loja": "Barra Shopping",
    "faturamento": 166546.26,
    "pedidos": 5379,
    "ticket": 30.96,
    "periodo": "2023-12"
  },
  {
    "mes": 12,
    "ano": 2023,
    "loja": "Caxias do Sul",
    "faturamento": 151384.68,
    "pedidos": 2815,
    "ticket": 53.78,
    "periodo": "2023-12"
  },
  {
    "mes": 12,
    "ano": 2023,
    "loja": "Novo Hamburgo",
    "faturamento": 107718.37,
    "pedidos": 2150,
    "ticket": 50.1,
    "periodo": "2023-12"
  },
  {
    "mes": 12,
    "ano": 2023,
    "loja": "São Leopoldo",
    "faturamento": 88884.29,
    "pedidos": 1774,
    "ticket": 50.1,
    "periodo": "2023-12"
  },
  {
    "mes": 12,
    "ano": 2023,
    "loja": "Montenegro",
    "faturamento": 75822.47,
    "pedidos": 1454,
    "ticket": 52.15,
    "periodo": "2023-12"
  },
  {
    "mes": 12,
    "ano": 2023,
    "loja": "Lajeado",
    "faturamento": 74607.39,
    "pedidos": 1502,
    "ticket": 49.67,
    "periodo": "2023-12"
  },
  {
    "mes": 12,
    "ano": 2023,
    "loja": "Sapiranga",
    "faturamento": 55652.88,
    "pedidos": 1478,
    "ticket": 37.65,
    "periodo": "2023-12"
  },
  {
    "mes": 12,
    "ano": 2023,
    "loja": "Bento Gonçalves",
    "faturamento": 55274.92,
    "pedidos": 1062,
    "ticket": 52.05,
    "periodo": "2023-12"
  },
  {
    "mes": 12,
    "ano": 2023,
    "loja": "Canoas",
    "faturamento": 49474.99,
    "pedidos": 1286,
    "ticket": 38.47,
    "periodo": "2023-12"
  },
  {
    "mes": 12,
    "ano": 2023,
    "loja": "Gravataí",
    "faturamento": 36071.53,
    "pedidos": 814,
    "ticket": 44.31,
    "periodo": "2023-12"
  },
  {
    "mes": 12,
    "ano": 2023,
    "loja": "Zona Norte",
    "faturamento": 6460.09,
    "pedidos": 134,
    "ticket": 48.21,
    "periodo": "2023-12"
  },
  {
    "mes": 1,
    "ano": 2024,
    "loja": "Caxias do Sul",
    "faturamento": 130916.46,
    "pedidos": 2382,
    "ticket": 54.96,
    "periodo": "2024-01"
  },
  {
    "mes": 1,
    "ano": 2024,
    "loja": "Barra Shopping",
    "faturamento": 111287.82,
    "pedidos": 3790,
    "ticket": 29.36,
    "periodo": "2024-01"
  },
  {
    "mes": 1,
    "ano": 2024,
    "loja": "Novo Hamburgo",
    "faturamento": 98238.38,
    "pedidos": 2047,
    "ticket": 47.99,
    "periodo": "2024-01"
  },
  {
    "mes": 1,
    "ano": 2024,
    "loja": "São Leopoldo",
    "faturamento": 80734.3,
    "pedidos": 1608,
    "ticket": 50.21,
    "periodo": "2024-01"
  },
  {
    "mes": 1,
    "ano": 2024,
    "loja": "Erechim",
    "faturamento": 76932.77,
    "pedidos": 1738,
    "ticket": 44.27,
    "periodo": "2024-01"
  },
  {
    "mes": 1,
    "ano": 2024,
    "loja": "Lajeado",
    "faturamento": 62465.62,
    "pedidos": 1217,
    "ticket": 51.33,
    "periodo": "2024-01"
  },
  {
    "mes": 1,
    "ano": 2024,
    "loja": "Montenegro",
    "faturamento": 61370.75,
    "pedidos": 1238,
    "ticket": 49.57,
    "periodo": "2024-01"
  },
  {
    "mes": 1,
    "ano": 2024,
    "loja": "Bento Gonçalves",
    "faturamento": 50082.27,
    "pedidos": 971,
    "ticket": 51.58,
    "periodo": "2024-01"
  },
  {
    "mes": 1,
    "ano": 2024,
    "loja": "Canoas",
    "faturamento": 41690.23,
    "pedidos": 988,
    "ticket": 42.2,
    "periodo": "2024-01"
  },
  {
    "mes": 1,
    "ano": 2024,
    "loja": "Gravataí",
    "faturamento": 39179.55,
    "pedidos": 883,
    "ticket": 44.37,
    "periodo": "2024-01"
  },
  {
    "mes": 1,
    "ano": 2024,
    "loja": "Zona Norte",
    "faturamento": 6870.47,
    "pedidos": 165,
    "ticket": 41.64,
    "periodo": "2024-01"
  },
  {
    "mes": 2,
    "ano": 2024,
    "loja": "Caxias do Sul",
    "faturamento": 122303.16,
    "pedidos": 2357,
    "ticket": 51.89,
    "periodo": "2024-02"
  },
  {
    "mes": 2,
    "ano": 2024,
    "loja": "Novo Hamburgo",
    "faturamento": 97897.72,
    "pedidos": 1961,
    "ticket": 49.92,
    "periodo": "2024-02"
  },
  {
    "mes": 2,
    "ano": 2024,
    "loja": "Barra Shopping",
    "faturamento": 97247.44,
    "pedidos": 3139,
    "ticket": 30.98,
    "periodo": "2024-02"
  },
  {
    "mes": 2,
    "ano": 2024,
    "loja": "São Leopoldo",
    "faturamento": 93863.35,
    "pedidos": 1933,
    "ticket": 48.56,
    "periodo": "2024-02"
  },
  {
    "mes": 2,
    "ano": 2024,
    "loja": "Erechim",
    "faturamento": 85055.03,
    "pedidos": 2035,
    "ticket": 41.8,
    "periodo": "2024-02"
  },
  {
    "mes": 2,
    "ano": 2024,
    "loja": "Floresta",
    "faturamento": 70084.49,
    "pedidos": 1643,
    "ticket": 42.66,
    "periodo": "2024-02"
  },
  {
    "mes": 2,
    "ano": 2024,
    "loja": "Montenegro",
    "faturamento": 67994.34,
    "pedidos": 1315,
    "ticket": 51.71,
    "periodo": "2024-02"
  },
  {
    "mes": 2,
    "ano": 2024,
    "loja": "Lajeado",
    "faturamento": 50775.84,
    "pedidos": 1011,
    "ticket": 50.22,
    "periodo": "2024-02"
  },
  {
    "mes": 2,
    "ano": 2024,
    "loja": "Sapiranga",
    "faturamento": 49982.44,
    "pedidos": 1276,
    "ticket": 39.17,
    "periodo": "2024-02"
  },
  {
    "mes": 2,
    "ano": 2024,
    "loja": "Bento Gonçalves",
    "faturamento": 43437.99,
    "pedidos": 872,
    "ticket": 49.81,
    "periodo": "2024-02"
  },
  {
    "mes": 2,
    "ano": 2024,
    "loja": "Gravataí",
    "faturamento": 39577.53,
    "pedidos": 953,
    "ticket": 41.53,
    "periodo": "2024-02"
  },
  {
    "mes": 2,
    "ano": 2024,
    "loja": "Canoas",
    "faturamento": 35894.77,
    "pedidos": 829,
    "ticket": 43.3,
    "periodo": "2024-02"
  },
  {
    "mes": 2,
    "ano": 2024,
    "loja": "Zona Norte",
    "faturamento": 4439.5,
    "pedidos": 84,
    "ticket": 52.85,
    "periodo": "2024-02"
  },
  {
    "mes": 3,
    "ano": 2024,
    "loja": "Caxias do Sul",
    "faturamento": 131901.42,
    "pedidos": 2510,
    "ticket": 52.55,
    "periodo": "2024-03"
  },
  {
    "mes": 3,
    "ano": 2024,
    "loja": "Novo Hamburgo",
    "faturamento": 105723.57,
    "pedidos": 2132,
    "ticket": 49.59,
    "periodo": "2024-03"
  },
  {
    "mes": 3,
    "ano": 2024,
    "loja": "São Leopoldo",
    "faturamento": 103647.37,
    "pedidos": 2056,
    "ticket": 50.41,
    "periodo": "2024-03"
  },
  {
    "mes": 3,
    "ano": 2024,
    "loja": "Barra Shopping",
    "faturamento": 92633.96,
    "pedidos": 3054,
    "ticket": 30.33,
    "periodo": "2024-03"
  },
  {
    "mes": 3,
    "ano": 2024,
    "loja": "Erechim",
    "faturamento": 84977.64,
    "pedidos": 1994,
    "ticket": 42.62,
    "periodo": "2024-03"
  },
  {
    "mes": 3,
    "ano": 2024,
    "loja": "Montenegro",
    "faturamento": 71080.12,
    "pedidos": 1385,
    "ticket": 51.32,
    "periodo": "2024-03"
  },
  {
    "mes": 3,
    "ano": 2024,
    "loja": "Floresta",
    "faturamento": 68845.12,
    "pedidos": 1630,
    "ticket": 42.24,
    "periodo": "2024-03"
  },
  {
    "mes": 3,
    "ano": 2024,
    "loja": "Sapiranga",
    "faturamento": 53791.95,
    "pedidos": 1381,
    "ticket": 38.95,
    "periodo": "2024-03"
  },
  {
    "mes": 3,
    "ano": 2024,
    "loja": "Lajeado",
    "faturamento": 50756.71,
    "pedidos": 1065,
    "ticket": 47.66,
    "periodo": "2024-03"
  },
  {
    "mes": 3,
    "ano": 2024,
    "loja": "Bento Gonçalves",
    "faturamento": 49135.17,
    "pedidos": 1004,
    "ticket": 48.94,
    "periodo": "2024-03"
  },
  {
    "mes": 3,
    "ano": 2024,
    "loja": "Protásio Alves",
    "faturamento": 46175.28,
    "pedidos": 1187,
    "ticket": 38.9,
    "periodo": "2024-03"
  },
  {
    "mes": 3,
    "ano": 2024,
    "loja": "Canoas",
    "faturamento": 38987.21,
    "pedidos": 787,
    "ticket": 49.54,
    "periodo": "2024-03"
  },
  {
    "mes": 3,
    "ano": 2024,
    "loja": "Gravataí",
    "faturamento": 38834.88,
    "pedidos": 823,
    "ticket": 47.19,
    "periodo": "2024-03"
  },
  {
    "mes": 3,
    "ano": 2024,
    "loja": "Zona Norte",
    "faturamento": 5024.02,
    "pedidos": 91,
    "ticket": 55.21,
    "periodo": "2024-03"
  },
  {
    "mes": 4,
    "ano": 2024,
    "loja": "Caxias do Sul",
    "faturamento": 135423.32,
    "pedidos": 2469,
    "ticket": 54.85,
    "periodo": "2024-04"
  },
  {
    "mes": 4,
    "ano": 2024,
    "loja": "Novo Hamburgo",
    "faturamento": 105653.58,
    "pedidos": 2167,
    "ticket": 48.76,
    "periodo": "2024-04"
  },
  {
    "mes": 4,
    "ano": 2024,
    "loja": "São Leopoldo",
    "faturamento": 99805.12,
    "pedidos": 2030,
    "ticket": 49.17,
    "periodo": "2024-04"
  },
  {
    "mes": 4,
    "ano": 2024,
    "loja": "Barra Shopping",
    "faturamento": 83207.62,
    "pedidos": 2689,
    "ticket": 30.94,
    "periodo": "2024-04"
  },
  {
    "mes": 4,
    "ano": 2024,
    "loja": "Erechim",
    "faturamento": 77069.92,
    "pedidos": 2028,
    "ticket": 38.0,
    "periodo": "2024-04"
  },
  {
    "mes": 4,
    "ano": 2024,
    "loja": "Montenegro",
    "faturamento": 76189.5,
    "pedidos": 1533,
    "ticket": 49.7,
    "periodo": "2024-04"
  },
  {
    "mes": 4,
    "ano": 2024,
    "loja": "Protásio Alves",
    "faturamento": 67765.41,
    "pedidos": 1749,
    "ticket": 38.75,
    "periodo": "2024-04"
  },
  {
    "mes": 4,
    "ano": 2024,
    "loja": "Floresta",
    "faturamento": 56965.43,
    "pedidos": 1450,
    "ticket": 39.29,
    "periodo": "2024-04"
  },
  {
    "mes": 4,
    "ano": 2024,
    "loja": "Sapiranga",
    "faturamento": 52642.68,
    "pedidos": 1406,
    "ticket": 37.44,
    "periodo": "2024-04"
  },
  {
    "mes": 4,
    "ano": 2024,
    "loja": "Esteio",
    "faturamento": 52415.87,
    "pedidos": 1089,
    "ticket": 48.13,
    "periodo": "2024-04"
  },
  {
    "mes": 4,
    "ano": 2024,
    "loja": "Bento Gonçalves",
    "faturamento": 50641.44,
    "pedidos": 1038,
    "ticket": 48.79,
    "periodo": "2024-04"
  },
  {
    "mes": 4,
    "ano": 2024,
    "loja": "Lajeado",
    "faturamento": 46826.66,
    "pedidos": 1042,
    "ticket": 44.94,
    "periodo": "2024-04"
  },
  {
    "mes": 4,
    "ano": 2024,
    "loja": "Gravataí",
    "faturamento": 37520.14,
    "pedidos": 831,
    "ticket": 45.15,
    "periodo": "2024-04"
  },
  {
    "mes": 4,
    "ano": 2024,
    "loja": "Canoas",
    "faturamento": 35754.96,
    "pedidos": 785,
    "ticket": 45.55,
    "periodo": "2024-04"
  },
  {
    "mes": 4,
    "ano": 2024,
    "loja": "Zona Norte",
    "faturamento": 6791.51,
    "pedidos": 125,
    "ticket": 54.33,
    "periodo": "2024-04"
  },
  {
    "mes": 5,
    "ano": 2024,
    "loja": "Caxias do Sul",
    "faturamento": 141678.08,
    "pedidos": 2417,
    "ticket": 58.62,
    "periodo": "2024-05"
  },
  {
    "mes": 5,
    "ano": 2024,
    "loja": "Esteio",
    "faturamento": 133434.48,
    "pedidos": 2601,
    "ticket": 51.3,
    "periodo": "2024-05"
  },
  {
    "mes": 5,
    "ano": 2024,
    "loja": "Novo Hamburgo",
    "faturamento": 127226.59,
    "pedidos": 2414,
    "ticket": 52.7,
    "periodo": "2024-05"
  },
  {
    "mes": 5,
    "ano": 2024,
    "loja": "Protásio Alves",
    "faturamento": 101886.36,
    "pedidos": 2162,
    "ticket": 47.13,
    "periodo": "2024-05"
  },
  {
    "mes": 5,
    "ano": 2024,
    "loja": "São Leopoldo",
    "faturamento": 87981.38,
    "pedidos": 1522,
    "ticket": 57.81,
    "periodo": "2024-05"
  },
  {
    "mes": 5,
    "ano": 2024,
    "loja": "Sapiranga",
    "faturamento": 66713.12,
    "pedidos": 1538,
    "ticket": 43.38,
    "periodo": "2024-05"
  },
  {
    "mes": 5,
    "ano": 2024,
    "loja": "Barra Shopping",
    "faturamento": 57776.07,
    "pedidos": 1929,
    "ticket": 29.95,
    "periodo": "2024-05"
  },
  {
    "mes": 5,
    "ano": 2024,
    "loja": "Gravataí",
    "faturamento": 55371.16,
    "pedidos": 1045,
    "ticket": 52.99,
    "periodo": "2024-05"
  },
  {
    "mes": 5,
    "ano": 2024,
    "loja": "Lajeado",
    "faturamento": 49855.99,
    "pedidos": 998,
    "ticket": 49.96,
    "periodo": "2024-05"
  },
  {
    "mes": 5,
    "ano": 2024,
    "loja": "Bento Gonçalves",
    "faturamento": 49377.77,
    "pedidos": 907,
    "ticket": 54.44,
    "periodo": "2024-05"
  },
  {
    "mes": 5,
    "ano": 2024,
    "loja": "Montenegro",
    "faturamento": 48969.65,
    "pedidos": 851,
    "ticket": 57.54,
    "periodo": "2024-05"
  },
  {
    "mes": 5,
    "ano": 2024,
    "loja": "Floresta",
    "faturamento": 28198.56,
    "pedidos": 574,
    "ticket": 49.13,
    "periodo": "2024-05"
  },
  {
    "mes": 5,
    "ano": 2024,
    "loja": "Canoas",
    "faturamento": 25318.02,
    "pedidos": 472,
    "ticket": 53.64,
    "periodo": "2024-05"
  },
  {
    "mes": 5,
    "ano": 2024,
    "loja": "Zona Norte",
    "faturamento": 10180.14,
    "pedidos": 193,
    "ticket": 52.75,
    "periodo": "2024-05"
  },
  {
    "mes": 6,
    "ano": 2024,
    "loja": "Caxias do Sul",
    "faturamento": 151206.45,
    "pedidos": 2542,
    "ticket": 59.48,
    "periodo": "2024-06"
  },
  {
    "mes": 6,
    "ano": 2024,
    "loja": "Novo Hamburgo",
    "faturamento": 129569.54,
    "pedidos": 2474,
    "ticket": 52.37,
    "periodo": "2024-06"
  },
  {
    "mes": 6,
    "ano": 2024,
    "loja": "São Leopoldo",
    "faturamento": 126959.93,
    "pedidos": 2237,
    "ticket": 56.75,
    "periodo": "2024-06"
  },
  {
    "mes": 6,
    "ano": 2024,
    "loja": "Esteio",
    "faturamento": 120763.77,
    "pedidos": 2487,
    "ticket": 48.56,
    "periodo": "2024-06"
  },
  {
    "mes": 6,
    "ano": 2024,
    "loja": "Barra Shopping",
    "faturamento": 95286.68,
    "pedidos": 3165,
    "ticket": 30.11,
    "periodo": "2024-06"
  },
  {
    "mes": 6,
    "ano": 2024,
    "loja": "Erechim",
    "faturamento": 83400.12,
    "pedidos": 2059,
    "ticket": 40.51,
    "periodo": "2024-06"
  },
  {
    "mes": 6,
    "ano": 2024,
    "loja": "Protásio Alves",
    "faturamento": 81970.38,
    "pedidos": 1921,
    "ticket": 42.67,
    "periodo": "2024-06"
  },
  {
    "mes": 6,
    "ano": 2024,
    "loja": "Montenegro",
    "faturamento": 80544.67,
    "pedidos": 1506,
    "ticket": 53.48,
    "periodo": "2024-06"
  },
  {
    "mes": 6,
    "ano": 2024,
    "loja": "Canoas",
    "faturamento": 74125.88,
    "pedidos": 1273,
    "ticket": 58.23,
    "periodo": "2024-06"
  },
  {
    "mes": 6,
    "ano": 2024,
    "loja": "Lajeado",
    "faturamento": 65799.34,
    "pedidos": 1333,
    "ticket": 49.36,
    "periodo": "2024-06"
  },
  {
    "mes": 6,
    "ano": 2024,
    "loja": "Sapiranga",
    "faturamento": 61943.97,
    "pedidos": 1426,
    "ticket": 43.44,
    "periodo": "2024-06"
  },
  {
    "mes": 6,
    "ano": 2024,
    "loja": "Floresta",
    "faturamento": 59044.9,
    "pedidos": 1309,
    "ticket": 45.11,
    "periodo": "2024-06"
  },
  {
    "mes": 6,
    "ano": 2024,
    "loja": "Gravataí",
    "faturamento": 48048.68,
    "pedidos": 954,
    "ticket": 50.37,
    "periodo": "2024-06"
  },
  {
    "mes": 6,
    "ano": 2024,
    "loja": "Bento Gonçalves",
    "faturamento": 45262.44,
    "pedidos": 846,
    "ticket": 53.5,
    "periodo": "2024-06"
  },
  {
    "mes": 6,
    "ano": 2024,
    "loja": "Zona Norte",
    "faturamento": 5165.04,
    "pedidos": 145,
    "ticket": 35.62,
    "periodo": "2024-06"
  },
  {
    "mes": 7,
    "ano": 2024,
    "loja": "Caxias do Sul",
    "faturamento": 161427.36,
    "pedidos": 2714,
    "ticket": 59.48,
    "periodo": "2024-07"
  },
  {
    "mes": 7,
    "ano": 2024,
    "loja": "Novo Hamburgo",
    "faturamento": 124444.11,
    "pedidos": 2551,
    "ticket": 48.78,
    "periodo": "2024-07"
  },
  {
    "mes": 7,
    "ano": 2024,
    "loja": "Esteio",
    "faturamento": 114141.93,
    "pedidos": 2357,
    "ticket": 48.43,
    "periodo": "2024-07"
  },
  {
    "mes": 7,
    "ano": 2024,
    "loja": "São Leopoldo",
    "faturamento": 108337.81,
    "pedidos": 1905,
    "ticket": 56.87,
    "periodo": "2024-07"
  },
  {
    "mes": 7,
    "ano": 2024,
    "loja": "Barra Shopping",
    "faturamento": 92663.15,
    "pedidos": 3124,
    "ticket": 29.66,
    "periodo": "2024-07"
  },
  {
    "mes": 7,
    "ano": 2024,
    "loja": "Protásio Alves",
    "faturamento": 77444.15,
    "pedidos": 1744,
    "ticket": 44.41,
    "periodo": "2024-07"
  },
  {
    "mes": 7,
    "ano": 2024,
    "loja": "Montenegro",
    "faturamento": 77268.7,
    "pedidos": 1449,
    "ticket": 53.33,
    "periodo": "2024-07"
  },
  {
    "mes": 7,
    "ano": 2024,
    "loja": "Erechim",
    "faturamento": 76299.95,
    "pedidos": 1835,
    "ticket": 41.58,
    "periodo": "2024-07"
  },
  {
    "mes": 7,
    "ano": 2024,
    "loja": "Canoas",
    "faturamento": 70092.97,
    "pedidos": 1286,
    "ticket": 54.5,
    "periodo": "2024-07"
  },
  {
    "mes": 7,
    "ano": 2024,
    "loja": "Sapiranga",
    "faturamento": 64509.84,
    "pedidos": 1403,
    "ticket": 45.98,
    "periodo": "2024-07"
  },
  {
    "mes": 7,
    "ano": 2024,
    "loja": "Lajeado",
    "faturamento": 61190.87,
    "pedidos": 1233,
    "ticket": 49.63,
    "periodo": "2024-07"
  },
  {
    "mes": 7,
    "ano": 2024,
    "loja": "Floresta",
    "faturamento": 48956.25,
    "pedidos": 1132,
    "ticket": 43.25,
    "periodo": "2024-07"
  },
  {
    "mes": 7,
    "ano": 2024,
    "loja": "Gravataí",
    "faturamento": 42791.32,
    "pedidos": 866,
    "ticket": 49.41,
    "periodo": "2024-07"
  },
  {
    "mes": 7,
    "ano": 2024,
    "loja": "Bento Gonçalves",
    "faturamento": 40737.98,
    "pedidos": 749,
    "ticket": 54.39,
    "periodo": "2024-07"
  },
  {
    "mes": 8,
    "ano": 2024,
    "loja": "Caxias do Sul",
    "faturamento": 140785.42,
    "pedidos": 2417,
    "ticket": 58.25,
    "periodo": "2024-08"
  },
  {
    "mes": 8,
    "ano": 2024,
    "loja": "Novo Hamburgo",
    "faturamento": 128775.61,
    "pedidos": 2609,
    "ticket": 49.36,
    "periodo": "2024-08"
  },
  {
    "mes": 8,
    "ano": 2024,
    "loja": "Canoas",
    "faturamento": 111011.97,
    "pedidos": 2108,
    "ticket": 52.66,
    "periodo": "2024-08"
  },
  {
    "mes": 8,
    "ano": 2024,
    "loja": "Esteio",
    "faturamento": 107613.59,
    "pedidos": 2239,
    "ticket": 48.06,
    "periodo": "2024-08"
  },
  {
    "mes": 8,
    "ano": 2024,
    "loja": "São Leopoldo",
    "faturamento": 102219.21,
    "pedidos": 1854,
    "ticket": 55.13,
    "periodo": "2024-08"
  },
  {
    "mes": 8,
    "ano": 2024,
    "loja": "Erechim",
    "faturamento": 91070.88,
    "pedidos": 2153,
    "ticket": 42.3,
    "periodo": "2024-08"
  },
  {
    "mes": 8,
    "ano": 2024,
    "loja": "Barra Shopping",
    "faturamento": 79503.36,
    "pedidos": 2669,
    "ticket": 29.79,
    "periodo": "2024-08"
  },
  {
    "mes": 8,
    "ano": 2024,
    "loja": "Sapiranga",
    "faturamento": 76528.11,
    "pedidos": 1799,
    "ticket": 42.54,
    "periodo": "2024-08"
  },
  {
    "mes": 8,
    "ano": 2024,
    "loja": "Montenegro",
    "faturamento": 74545.34,
    "pedidos": 1421,
    "ticket": 52.46,
    "periodo": "2024-08"
  },
  {
    "mes": 8,
    "ano": 2024,
    "loja": "Protásio Alves",
    "faturamento": 74255.98,
    "pedidos": 1728,
    "ticket": 42.97,
    "periodo": "2024-08"
  },
  {
    "mes": 8,
    "ano": 2024,
    "loja": "Lajeado",
    "faturamento": 47945.86,
    "pedidos": 998,
    "ticket": 48.04,
    "periodo": "2024-08"
  },
  {
    "mes": 8,
    "ano": 2024,
    "loja": "Floresta",
    "faturamento": 47647.18,
    "pedidos": 1151,
    "ticket": 41.4,
    "periodo": "2024-08"
  },
  {
    "mes": 8,
    "ano": 2024,
    "loja": "Bento Gonçalves",
    "faturamento": 42220.78,
    "pedidos": 799,
    "ticket": 52.84,
    "periodo": "2024-08"
  },
  {
    "mes": 8,
    "ano": 2024,
    "loja": "Gravataí",
    "faturamento": 41265.69,
    "pedidos": 850,
    "ticket": 48.55,
    "periodo": "2024-08"
  },
  {
    "mes": 8,
    "ano": 2024,
    "loja": "Zona Norte",
    "faturamento": 5052.35,
    "pedidos": 131,
    "ticket": 38.57,
    "periodo": "2024-08"
  },
  {
    "mes": 9,
    "ano": 2024,
    "loja": "Caxias do Sul",
    "faturamento": 141996.81,
    "pedidos": 2502,
    "ticket": 56.75,
    "periodo": "2024-09"
  },
  {
    "mes": 9,
    "ano": 2024,
    "loja": "Novo Hamburgo",
    "faturamento": 107210.77,
    "pedidos": 2301,
    "ticket": 46.59,
    "periodo": "2024-09"
  },
  {
    "mes": 9,
    "ano": 2024,
    "loja": "Esteio",
    "faturamento": 105540.66,
    "pedidos": 2255,
    "ticket": 46.8,
    "periodo": "2024-09"
  },
  {
    "mes": 9,
    "ano": 2024,
    "loja": "Canoas",
    "faturamento": 95268.57,
    "pedidos": 1780,
    "ticket": 53.52,
    "periodo": "2024-09"
  },
  {
    "mes": 9,
    "ano": 2024,
    "loja": "São Leopoldo",
    "faturamento": 92945.77,
    "pedidos": 1763,
    "ticket": 52.72,
    "periodo": "2024-09"
  },
  {
    "mes": 9,
    "ano": 2024,
    "loja": "Barra Shopping",
    "faturamento": 77426.87,
    "pedidos": 2620,
    "ticket": 29.55,
    "periodo": "2024-09"
  },
  {
    "mes": 9,
    "ano": 2024,
    "loja": "Sapiranga",
    "faturamento": 71939.29,
    "pedidos": 1745,
    "ticket": 41.23,
    "periodo": "2024-09"
  },
  {
    "mes": 9,
    "ano": 2024,
    "loja": "Montenegro",
    "faturamento": 71189.06,
    "pedidos": 1426,
    "ticket": 49.92,
    "periodo": "2024-09"
  },
  {
    "mes": 9,
    "ano": 2024,
    "loja": "Protásio Alves",
    "faturamento": 60922.7,
    "pedidos": 1405,
    "ticket": 43.36,
    "periodo": "2024-09"
  },
  {
    "mes": 9,
    "ano": 2024,
    "loja": "Lajeado",
    "faturamento": 49130.61,
    "pedidos": 1095,
    "ticket": 44.87,
    "periodo": "2024-09"
  },
  {
    "mes": 9,
    "ano": 2024,
    "loja": "Gravataí",
    "faturamento": 44314.3,
    "pedidos": 1007,
    "ticket": 44.01,
    "periodo": "2024-09"
  },
  {
    "mes": 9,
    "ano": 2024,
    "loja": "Bento Gonçalves",
    "faturamento": 38636.81,
    "pedidos": 729,
    "ticket": 53.0,
    "periodo": "2024-09"
  },
  {
    "mes": 9,
    "ano": 2024,
    "loja": "Floresta",
    "faturamento": 37358.14,
    "pedidos": 974,
    "ticket": 38.36,
    "periodo": "2024-09"
  },
  {
    "mes": 9,
    "ano": 2024,
    "loja": "Zona Norte",
    "faturamento": 5196.97,
    "pedidos": 136,
    "ticket": 38.21,
    "periodo": "2024-09"
  },
  {
    "mes": 10,
    "ano": 2024,
    "loja": "Caxias do Sul",
    "faturamento": 142674.81,
    "pedidos": 2565,
    "ticket": 55.62,
    "periodo": "2024-10"
  },
  {
    "mes": 10,
    "ano": 2024,
    "loja": "Novo Hamburgo",
    "faturamento": 108829.32,
    "pedidos": 2345,
    "ticket": 46.41,
    "periodo": "2024-10"
  },
  {
    "mes": 10,
    "ano": 2024,
    "loja": "Esteio",
    "faturamento": 107012.63,
    "pedidos": 2331,
    "ticket": 45.91,
    "periodo": "2024-10"
  },
  {
    "mes": 10,
    "ano": 2024,
    "loja": "São Leopoldo",
    "faturamento": 97405.54,
    "pedidos": 1961,
    "ticket": 49.67,
    "periodo": "2024-10"
  },
  {
    "mes": 10,
    "ano": 2024,
    "loja": "Erechim",
    "faturamento": 96549.47,
    "pedidos": 2379,
    "ticket": 40.58,
    "periodo": "2024-10"
  },
  {
    "mes": 10,
    "ano": 2024,
    "loja": "Canoas",
    "faturamento": 90896.16,
    "pedidos": 1716,
    "ticket": 52.97,
    "periodo": "2024-10"
  },
  {
    "mes": 10,
    "ano": 2024,
    "loja": "Barra Shopping",
    "faturamento": 88359.7,
    "pedidos": 2863,
    "ticket": 30.86,
    "periodo": "2024-10"
  },
  {
    "mes": 10,
    "ano": 2024,
    "loja": "Montenegro",
    "faturamento": 77197.76,
    "pedidos": 1622,
    "ticket": 47.59,
    "periodo": "2024-10"
  },
  {
    "mes": 10,
    "ano": 2024,
    "loja": "Protásio Alves",
    "faturamento": 65273.52,
    "pedidos": 1483,
    "ticket": 44.01,
    "periodo": "2024-10"
  },
  {
    "mes": 10,
    "ano": 2024,
    "loja": "Gravataí",
    "faturamento": 55357.33,
    "pedidos": 1226,
    "ticket": 45.15,
    "periodo": "2024-10"
  },
  {
    "mes": 10,
    "ano": 2024,
    "loja": "Lajeado",
    "faturamento": 52914.67,
    "pedidos": 1143,
    "ticket": 46.29,
    "periodo": "2024-10"
  },
  {
    "mes": 10,
    "ano": 2024,
    "loja": "Bento Gonçalves",
    "faturamento": 43387.18,
    "pedidos": 826,
    "ticket": 52.53,
    "periodo": "2024-10"
  },
  {
    "mes": 10,
    "ano": 2024,
    "loja": "Floresta",
    "faturamento": 39144.54,
    "pedidos": 1012,
    "ticket": 38.68,
    "periodo": "2024-10"
  },
  {
    "mes": 10,
    "ano": 2024,
    "loja": "Zona Norte",
    "faturamento": 5209.72,
    "pedidos": 166,
    "ticket": 31.38,
    "periodo": "2024-10"
  },
  {
    "mes": 11,
    "ano": 2024,
    "loja": "Caxias do Sul",
    "faturamento": 140032.72,
    "pedidos": 2447,
    "ticket": 57.23,
    "periodo": "2024-11"
  },
  {
    "mes": 11,
    "ano": 2024,
    "loja": "Esteio",
    "faturamento": 114211.14,
    "pedidos": 2419,
    "ticket": 47.21,
    "periodo": "2024-11"
  },
  {
    "mes": 11,
    "ano": 2024,
    "loja": "São Leopoldo",
    "faturamento": 103129.69,
    "pedidos": 2067,
    "ticket": 49.89,
    "periodo": "2024-11"
  },
  {
    "mes": 11,
    "ano": 2024,
    "loja": "Novo Hamburgo",
    "faturamento": 102653.05,
    "pedidos": 1879,
    "ticket": 54.63,
    "periodo": "2024-11"
  },
  {
    "mes": 11,
    "ano": 2024,
    "loja": "Barra Shopping",
    "faturamento": 100780.77,
    "pedidos": 3236,
    "ticket": 31.14,
    "periodo": "2024-11"
  },
  {
    "mes": 11,
    "ano": 2024,
    "loja": "Canoas",
    "faturamento": 90609.35,
    "pedidos": 1694,
    "ticket": 53.49,
    "periodo": "2024-11"
  },
  {
    "mes": 11,
    "ano": 2024,
    "loja": "Erechim",
    "faturamento": 90221.21,
    "pedidos": 2166,
    "ticket": 41.65,
    "periodo": "2024-11"
  },
  {
    "mes": 11,
    "ano": 2024,
    "loja": "Montenegro",
    "faturamento": 85892.07,
    "pedidos": 1823,
    "ticket": 47.12,
    "periodo": "2024-11"
  },
  {
    "mes": 11,
    "ano": 2024,
    "loja": "Sapiranga",
    "faturamento": 69224.49,
    "pedidos": 1513,
    "ticket": 45.75,
    "periodo": "2024-11"
  },
  {
    "mes": 11,
    "ano": 2024,
    "loja": "Protásio Alves",
    "faturamento": 65867.97,
    "pedidos": 1468,
    "ticket": 44.87,
    "periodo": "2024-11"
  },
  {
    "mes": 11,
    "ano": 2024,
    "loja": "Gravataí",
    "faturamento": 57332.13,
    "pedidos": 1278,
    "ticket": 44.86,
    "periodo": "2024-11"
  },
  {
    "mes": 11,
    "ano": 2024,
    "loja": "Capão da Canoa",
    "faturamento": 56077.36,
    "pedidos": 1248,
    "ticket": 44.93,
    "periodo": "2024-11"
  },
  {
    "mes": 11,
    "ano": 2024,
    "loja": "Lajeado",
    "faturamento": 55496.73,
    "pedidos": 1169,
    "ticket": 47.47,
    "periodo": "2024-11"
  },
  {
    "mes": 11,
    "ano": 2024,
    "loja": "Bento Gonçalves",
    "faturamento": 46826.86,
    "pedidos": 866,
    "ticket": 54.07,
    "periodo": "2024-11"
  },
  {
    "mes": 11,
    "ano": 2024,
    "loja": "Floresta",
    "faturamento": 33171.46,
    "pedidos": 846,
    "ticket": 39.21,
    "periodo": "2024-11"
  },
  {
    "mes": 12,
    "ano": 2024,
    "loja": "Caxias do Sul",
    "faturamento": 150270.33,
    "pedidos": 2582,
    "ticket": 58.2,
    "periodo": "2024-12"
  },
  {
    "mes": 12,
    "ano": 2024,
    "loja": "Barra Shopping",
    "faturamento": 131492.62,
    "pedidos": 4056,
    "ticket": 32.42,
    "periodo": "2024-12"
  },
  {
    "mes": 12,
    "ano": 2024,
    "loja": "Novo Hamburgo",
    "faturamento": 105258.11,
    "pedidos": 2007,
    "ticket": 52.45,
    "periodo": "2024-12"
  },
  {
    "mes": 12,
    "ano": 2024,
    "loja": "São Leopoldo",
    "faturamento": 104515.8,
    "pedidos": 2029,
    "ticket": 51.51,
    "periodo": "2024-12"
  },
  {
    "mes": 12,
    "ano": 2024,
    "loja": "Esteio",
    "faturamento": 102495.07,
    "pedidos": 2285,
    "ticket": 44.86,
    "periodo": "2024-12"
  },
  {
    "mes": 12,
    "ano": 2024,
    "loja": "Montenegro",
    "faturamento": 99601.78,
    "pedidos": 2124,
    "ticket": 46.89,
    "periodo": "2024-12"
  },
  {
    "mes": 12,
    "ano": 2024,
    "loja": "Erechim",
    "faturamento": 93095.12,
    "pedidos": 2103,
    "ticket": 44.27,
    "periodo": "2024-12"
  },
  {
    "mes": 12,
    "ano": 2024,
    "loja": "Sapiranga",
    "faturamento": 86526.51,
    "pedidos": 1915,
    "ticket": 45.18,
    "periodo": "2024-12"
  },
  {
    "mes": 12,
    "ano": 2024,
    "loja": "Canoas",
    "faturamento": 85316.04,
    "pedidos": 1575,
    "ticket": 54.17,
    "periodo": "2024-12"
  },
  {
    "mes": 12,
    "ano": 2024,
    "loja": "Capão da Canoa",
    "faturamento": 68869.44,
    "pedidos": 1612,
    "ticket": 42.72,
    "periodo": "2024-12"
  },
  {
    "mes": 12,
    "ano": 2024,
    "loja": "Protásio Alves",
    "faturamento": 64357.02,
    "pedidos": 1460,
    "ticket": 44.08,
    "periodo": "2024-12"
  },
  {
    "mes": 12,
    "ano": 2024,
    "loja": "Gravataí",
    "faturamento": 63225.89,
    "pedidos": 1430,
    "ticket": 44.21,
    "periodo": "2024-12"
  },
  {
    "mes": 12,
    "ano": 2024,
    "loja": "Lajeado",
    "faturamento": 61493.46,
    "pedidos": 1296,
    "ticket": 47.45,
    "periodo": "2024-12"
  },
  {
    "mes": 12,
    "ano": 2024,
    "loja": "Bento Gonçalves",
    "faturamento": 46900.84,
    "pedidos": 879,
    "ticket": 53.36,
    "periodo": "2024-12"
  },
  {
    "mes": 12,
    "ano": 2024,
    "loja": "Porto Alegre Zona Norte",
    "faturamento": 9503.37,
    "pedidos": 255,
    "ticket": 37.27,
    "periodo": "2024-12"
  },
  {
    "mes": 1,
    "ano": 2025,
    "loja": "Caxias do Sul",
    "faturamento": 127978.76,
    "pedidos": 2247,
    "ticket": 56.96,
    "periodo": "2025-01"
  },
  {
    "mes": 1,
    "ano": 2025,
    "loja": "Capão da Canoa",
    "faturamento": 99116.5,
    "pedidos": 2099,
    "ticket": 47.22,
    "periodo": "2025-01"
  },
  {
    "mes": 1,
    "ano": 2025,
    "loja": "Novo Hamburgo",
    "faturamento": 93751.71,
    "pedidos": 1822,
    "ticket": 51.46,
    "periodo": "2025-01"
  },
  {
    "mes": 1,
    "ano": 2025,
    "loja": "São Leopoldo",
    "faturamento": 91235.01,
    "pedidos": 1873,
    "ticket": 48.71,
    "periodo": "2025-01"
  },
  {
    "mes": 1,
    "ano": 2025,
    "loja": "Barra Shopping",
    "faturamento": 90771.49,
    "pedidos": 2991,
    "ticket": 30.35,
    "periodo": "2025-01"
  },
  {
    "mes": 1,
    "ano": 2025,
    "loja": "Montenegro",
    "faturamento": 89321.75,
    "pedidos": 1857,
    "ticket": 48.1,
    "periodo": "2025-01"
  },
  {
    "mes": 1,
    "ano": 2025,
    "loja": "Esteio",
    "faturamento": 87167.15,
    "pedidos": 1991,
    "ticket": 43.78,
    "periodo": "2025-01"
  },
  {
    "mes": 1,
    "ano": 2025,
    "loja": "Canoas",
    "faturamento": 78297.41,
    "pedidos": 1472,
    "ticket": 53.19,
    "periodo": "2025-01"
  },
  {
    "mes": 1,
    "ano": 2025,
    "loja": "Sapiranga",
    "faturamento": 77219.21,
    "pedidos": 1630,
    "ticket": 47.37,
    "periodo": "2025-01"
  },
  {
    "mes": 1,
    "ano": 2025,
    "loja": "Erechim",
    "faturamento": 70360.63,
    "pedidos": 1640,
    "ticket": 42.9,
    "periodo": "2025-01"
  },
  {
    "mes": 1,
    "ano": 2025,
    "loja": "Gravataí",
    "faturamento": 59346.07,
    "pedidos": 1321,
    "ticket": 44.93,
    "periodo": "2025-01"
  },
  {
    "mes": 1,
    "ano": 2025,
    "loja": "Protásio Alves",
    "faturamento": 54874.94,
    "pedidos": 1186,
    "ticket": 46.27,
    "periodo": "2025-01"
  },
  {
    "mes": 1,
    "ano": 2025,
    "loja": "Lajeado",
    "faturamento": 53103.51,
    "pedidos": 1166,
    "ticket": 45.54,
    "periodo": "2025-01"
  },
  {
    "mes": 1,
    "ano": 2025,
    "loja": "Bento Gonçalves",
    "faturamento": 45230.74,
    "pedidos": 928,
    "ticket": 48.74,
    "periodo": "2025-01"
  },
  {
    "mes": 1,
    "ano": 2025,
    "loja": "Floresta",
    "faturamento": 31258.45,
    "pedidos": 761,
    "ticket": 41.08,
    "periodo": "2025-01"
  },
  {
    "mes": 1,
    "ano": 2025,
    "loja": "Porto Alegre Zona Norte",
    "faturamento": 12467.5,
    "pedidos": 323,
    "ticket": 38.6,
    "periodo": "2025-01"
  },
  {
    "mes": 2,
    "ano": 2025,
    "loja": "Caxias do Sul",
    "faturamento": 133607.09,
    "pedidos": 2484,
    "ticket": 53.79,
    "periodo": "2025-02"
  },
  {
    "mes": 2,
    "ano": 2025,
    "loja": "São Leopoldo",
    "faturamento": 88246.18,
    "pedidos": 1737,
    "ticket": 50.8,
    "periodo": "2025-02"
  },
  {
    "mes": 2,
    "ano": 2025,
    "loja": "Montenegro",
    "faturamento": 87375.84,
    "pedidos": 1799,
    "ticket": 48.57,
    "periodo": "2025-02"
  },
  {
    "mes": 2,
    "ano": 2025,
    "loja": "Novo Hamburgo",
    "faturamento": 86234.26,
    "pedidos": 1651,
    "ticket": 52.23,
    "periodo": "2025-02"
  },
  {
    "mes": 2,
    "ano": 2025,
    "loja": "Esteio",
    "faturamento": 84749.04,
    "pedidos": 1973,
    "ticket": 42.95,
    "periodo": "2025-02"
  },
  {
    "mes": 2,
    "ano": 2025,
    "loja": "Erechim",
    "faturamento": 84729.98,
    "pedidos": 1998,
    "ticket": 42.41,
    "periodo": "2025-02"
  },
  {
    "mes": 2,
    "ano": 2025,
    "loja": "Barra Shopping",
    "faturamento": 79973.7,
    "pedidos": 2599,
    "ticket": 30.77,
    "periodo": "2025-02"
  },
  {
    "mes": 2,
    "ano": 2025,
    "loja": "Sapiranga",
    "faturamento": 75661.57,
    "pedidos": 1688,
    "ticket": 44.82,
    "periodo": "2025-02"
  },
  {
    "mes": 2,
    "ano": 2025,
    "loja": "Canoas",
    "faturamento": 75361.19,
    "pedidos": 1333,
    "ticket": 56.54,
    "periodo": "2025-02"
  },
  {
    "mes": 2,
    "ano": 2025,
    "loja": "Capão da Canoa",
    "faturamento": 72624.66,
    "pedidos": 1478,
    "ticket": 49.14,
    "periodo": "2025-02"
  },
  {
    "mes": 2,
    "ano": 2025,
    "loja": "Protásio Alves",
    "faturamento": 62799.38,
    "pedidos": 1393,
    "ticket": 45.08,
    "periodo": "2025-02"
  },
  {
    "mes": 2,
    "ano": 2025,
    "loja": "Gravataí",
    "faturamento": 53164.95,
    "pedidos": 1154,
    "ticket": 46.07,
    "periodo": "2025-02"
  },
  {
    "mes": 2,
    "ano": 2025,
    "loja": "Lajeado",
    "faturamento": 52071.35,
    "pedidos": 1164,
    "ticket": 44.73,
    "periodo": "2025-02"
  },
  {
    "mes": 2,
    "ano": 2025,
    "loja": "Bento Gonçalves",
    "faturamento": 50724.83,
    "pedidos": 1049,
    "ticket": 48.36,
    "periodo": "2025-02"
  },
  {
    "mes": 2,
    "ano": 2025,
    "loja": "Floresta",
    "faturamento": 48317.12,
    "pedidos": 1178,
    "ticket": 41.02,
    "periodo": "2025-02"
  },
  {
    "mes": 2,
    "ano": 2025,
    "loja": "Porto Alegre Zona Norte",
    "faturamento": 8781.87,
    "pedidos": 226,
    "ticket": 38.86,
    "periodo": "2025-02"
  },
  {
    "mes": 3,
    "ano": 2025,
    "loja": "Caxias do Sul",
    "faturamento": 155399.0,
    "pedidos": 2905,
    "ticket": 53.49,
    "periodo": "2025-03"
  },
  {
    "mes": 3,
    "ano": 2025,
    "loja": "Esteio",
    "faturamento": 110394.0,
    "pedidos": 2504,
    "ticket": 44.09,
    "periodo": "2025-03"
  },
  {
    "mes": 3,
    "ano": 2025,
    "loja": "Novo Hamburgo",
    "faturamento": 101018.0,
    "pedidos": 1890,
    "ticket": 53.45,
    "periodo": "2025-03"
  },
  {
    "mes": 3,
    "ano": 2025,
    "loja": "São Leopoldo",
    "faturamento": 98828.0,
    "pedidos": 1922,
    "ticket": 51.42,
    "periodo": "2025-03"
  },
  {
    "mes": 3,
    "ano": 2025,
    "loja": "Montenegro",
    "faturamento": 97872.0,
    "pedidos": 2100,
    "ticket": 46.61,
    "periodo": "2025-03"
  },
  {
    "mes": 3,
    "ano": 2025,
    "loja": "Canoas",
    "faturamento": 95864.0,
    "pedidos": 1706,
    "ticket": 56.19,
    "periodo": "2025-03"
  },
  {
    "mes": 3,
    "ano": 2025,
    "loja": "Barra Shopping",
    "faturamento": 92567.0,
    "pedidos": 3037,
    "ticket": 30.48,
    "periodo": "2025-03"
  },
  {
    "mes": 3,
    "ano": 2025,
    "loja": "Erechim",
    "faturamento": 89672.0,
    "pedidos": 2154,
    "ticket": 41.63,
    "periodo": "2025-03"
  },
  {
    "mes": 3,
    "ano": 2025,
    "loja": "Protásio Alves",
    "faturamento": 86428.0,
    "pedidos": 1854,
    "ticket": 46.62,
    "periodo": "2025-03"
  },
  {
    "mes": 3,
    "ano": 2025,
    "loja": "Sapiranga",
    "faturamento": 86119.0,
    "pedidos": 2058,
    "ticket": 41.85,
    "periodo": "2025-03"
  },
  {
    "mes": 3,
    "ano": 2025,
    "loja": "Capão da Canoa",
    "faturamento": 75427.0,
    "pedidos": 1769,
    "ticket": 42.64,
    "periodo": "2025-03"
  },
  {
    "mes": 3,
    "ano": 2025,
    "loja": "Gravataí",
    "faturamento": 60026.0,
    "pedidos": 1288,
    "ticket": 46.6,
    "periodo": "2025-03"
  },
  {
    "mes": 3,
    "ano": 2025,
    "loja": "Bento Gonçalves",
    "faturamento": 57857.0,
    "pedidos": 1119,
    "ticket": 51.7,
    "periodo": "2025-03"
  },
  {
    "mes": 3,
    "ano": 2025,
    "loja": "Lajeado",
    "faturamento": 57217.0,
    "pedidos": 1294,
    "ticket": 44.22,
    "periodo": "2025-03"
  },
  {
    "mes": 4,
    "ano": 2025,
    "loja": "Caxias do Sul",
    "faturamento": 142137.88,
    "pedidos": 2505,
    "ticket": 56.74,
    "periodo": "2025-04"
  },
  {
    "mes": 4,
    "ano": 2025,
    "loja": "Esteio",
    "faturamento": 99506.82,
    "pedidos": 2105,
    "ticket": 47.27,
    "periodo": "2025-04"
  },
  {
    "mes": 4,
    "ano": 2025,
    "loja": "Novo Hamburgo",
    "faturamento": 98690.79,
    "pedidos": 1778,
    "ticket": 55.51,
    "periodo": "2025-04"
  },
  {
    "mes": 4,
    "ano": 2025,
    "loja": "São Leopoldo",
    "faturamento": 92933.28,
    "pedidos": 1806,
    "ticket": 51.46,
    "periodo": "2025-04"
  },
  {
    "mes": 4,
    "ano": 2025,
    "loja": "Barra Shopping",
    "faturamento": 91550.13,
    "pedidos": 2807,
    "ticket": 32.61,
    "periodo": "2025-04"
  },
  {
    "mes": 4,
    "ano": 2025,
    "loja": "Protásio Alves",
    "faturamento": 86688.06,
    "pedidos": 1929,
    "ticket": 44.94,
    "periodo": "2025-04"
  },
  {
    "mes": 4,
    "ano": 2025,
    "loja": "Montenegro",
    "faturamento": 86650.58,
    "pedidos": 1685,
    "ticket": 51.42,
    "periodo": "2025-04"
  },
  {
    "mes": 4,
    "ano": 2025,
    "loja": "Canoas",
    "faturamento": 76154.0,
    "pedidos": 1455,
    "ticket": 52.34,
    "periodo": "2025-04"
  },
  {
    "mes": 4,
    "ano": 2025,
    "loja": "Erechim",
    "faturamento": 75014.2,
    "pedidos": 1718,
    "ticket": 43.66,
    "periodo": "2025-04"
  },
  {
    "mes": 4,
    "ano": 2025,
    "loja": "Sapiranga",
    "faturamento": 73756.62,
    "pedidos": 1636,
    "ticket": 45.08,
    "periodo": "2025-04"
  },
  {
    "mes": 4,
    "ano": 2025,
    "loja": "Ijuí",
    "faturamento": 72411.97,
    "pedidos": 1394,
    "ticket": 51.95,
    "periodo": "2025-04"
  },
  {
    "mes": 4,
    "ano": 2025,
    "loja": "Lajeado",
    "faturamento": 53282.86,
    "pedidos": 1111,
    "ticket": 47.96,
    "periodo": "2025-04"
  },
  {
    "mes": 4,
    "ano": 2025,
    "loja": "Bento Gonçalves",
    "faturamento": 50394.92,
    "pedidos": 1014,
    "ticket": 49.7,
    "periodo": "2025-04"
  },
  {
    "mes": 4,
    "ano": 2025,
    "loja": "Gravataí",
    "faturamento": 47434.2,
    "pedidos": 954,
    "ticket": 49.72,
    "periodo": "2025-04"
  },
  {
    "mes": 4,
    "ano": 2025,
    "loja": "Capão da Canoa",
    "faturamento": 45931.72,
    "pedidos": 1000,
    "ticket": 45.93,
    "periodo": "2025-04"
  },
  {
    "mes": 4,
    "ano": 2025,
    "loja": "Porto Alegre Zona Norte",
    "faturamento": 6793.61,
    "pedidos": 154,
    "ticket": 44.11,
    "periodo": "2025-04"
  },
  {
    "mes": 5,
    "ano": 2025,
    "loja": "Montenegro",
    "faturamento": 94364.55,
    "pedidos": 1930,
    "ticket": 48.89,
    "periodo": "2025-05"
  },
  {
    "mes": 5,
    "ano": 2025,
    "loja": "Canoas",
    "faturamento": 86708.46,
    "pedidos": 1679,
    "ticket": 51.64,
    "periodo": "2025-05"
  },
  {
    "mes": 5,
    "ano": 2025,
    "loja": "Porto Alegre Zona Norte",
    "faturamento": 9487.58,
    "pedidos": 206,
    "ticket": 46.06,
    "periodo": "2025-05"
  },
  {
    "mes": 5,
    "ano": 2025,
    "loja": "São Leopoldo",
    "faturamento": 97252.02,
    "pedidos": 1858,
    "ticket": 52.34,
    "periodo": "2025-05"
  },
  {
    "mes": 5,
    "ano": 2025,
    "loja": "Novo Hamburgo",
    "faturamento": 105388.86,
    "pedidos": 2012,
    "ticket": 52.38,
    "periodo": "2025-05"
  },
  {
    "mes": 5,
    "ano": 2025,
    "loja": "Barra Shopping",
    "faturamento": 90232.32,
    "pedidos": 2781,
    "ticket": 32.45,
    "periodo": "2025-05"
  },
  {
    "mes": 5,
    "ano": 2025,
    "loja": "Floresta",
    "faturamento": 58451.14,
    "pedidos": 1352,
    "ticket": 43.23,
    "periodo": "2025-05"
  },
  {
    "mes": 5,
    "ano": 2025,
    "loja": "Sapiranga",
    "faturamento": 86357.48,
    "pedidos": 1887,
    "ticket": 45.76,
    "periodo": "2025-05"
  },
  {
    "mes": 5,
    "ano": 2025,
    "loja": "Gravataí",
    "faturamento": 51597.09,
    "pedidos": 1065,
    "ticket": 48.45,
    "periodo": "2025-05"
  },
  {
    "mes": 5,
    "ano": 2025,
    "loja": "Caxias do Sul",
    "faturamento": 162730.31,
    "pedidos": 2955,
    "ticket": 55.07,
    "periodo": "2025-05"
  },
  {
    "mes": 5,
    "ano": 2025,
    "loja": "Bento Gonçalves",
    "faturamento": 60377.65,
    "pedidos": 1192,
    "ticket": 50.65,
    "periodo": "2025-05"
  },
  {
    "mes": 5,
    "ano": 2025,
    "loja": "Lajeado",
    "faturamento": 46662.75,
    "pedidos": 937,
    "ticket": 49.8,
    "periodo": "2025-05"
  },
  {
    "mes": 5,
    "ano": 2025,
    "loja": "Erechim",
    "faturamento": 78169.32,
    "pedidos": 1778,
    "ticket": 43.96,
    "periodo": "2025-05"
  },
  {
    "mes": 5,
    "ano": 2025,
    "loja": "Protásio Alves",
    "faturamento": 98469.56,
    "pedidos": 2112,
    "ticket": 46.62,
    "periodo": "2025-05"
  },
  {
    "mes": 5,
    "ano": 2025,
    "loja": "Esteio",
    "faturamento": 99546.11,
    "pedidos": 2145,
    "ticket": 46.41,
    "periodo": "2025-05"
  },
  {
    "mes": 5,
    "ano": 2025,
    "loja": "Capão da Canoa",
    "faturamento": 52971.27,
    "pedidos": 1158,
    "ticket": 45.74,
    "periodo": "2025-05"
  },
  {
    "mes": 5,
    "ano": 2025,
    "loja": "Ijuí",
    "faturamento": 145404.04,
    "pedidos": 2830,
    "ticket": 51.38,
    "periodo": "2025-05"
  },
  {
    "mes": 6,
    "ano": 2025,
    "loja": "Montenegro",
    "faturamento": 87082.52,
    "pedidos": 1671,
    "ticket": 52.11,
    "periodo": "2025-06"
  },
  {
    "mes": 6,
    "ano": 2025,
    "loja": "Canoas",
    "faturamento": 94742.73,
    "pedidos": 1908,
    "ticket": 49.66,
    "periodo": "2025-06"
  },
  {
    "mes": 6,
    "ano": 2025,
    "loja": "Porto Alegre Zona Norte",
    "faturamento": 10708.26,
    "pedidos": 226,
    "ticket": 47.38,
    "periodo": "2025-06"
  },
  {
    "mes": 6,
    "ano": 2025,
    "loja": "São Leopoldo",
    "faturamento": 98705.95,
    "pedidos": 1870,
    "ticket": 52.78,
    "periodo": "2025-06"
  },
  {
    "mes": 6,
    "ano": 2025,
    "loja": "Novo Hamburgo",
    "faturamento": 100887.64,
    "pedidos": 1862,
    "ticket": 54.18,
    "periodo": "2025-06"
  },
  {
    "mes": 6,
    "ano": 2025,
    "loja": "Barra Shopping",
    "faturamento": 78704.47,
    "pedidos": 2436,
    "ticket": 32.31,
    "periodo": "2025-06"
  },
  {
    "mes": 6,
    "ano": 2025,
    "loja": "Porto Alegre Floresta",
    "faturamento": 61375.81,
    "pedidos": 1412,
    "ticket": 43.47,
    "periodo": "2025-06"
  },
  {
    "mes": 6,
    "ano": 2025,
    "loja": "Sapiranga",
    "faturamento": 90023.99,
    "pedidos": 1887,
    "ticket": 47.71,
    "periodo": "2025-06"
  },
  {
    "mes": 6,
    "ano": 2025,
    "loja": "Gravataí",
    "faturamento": 52623.54,
    "pedidos": 1079,
    "ticket": 48.77,
    "periodo": "2025-06"
  },
  {
    "mes": 6,
    "ano": 2025,
    "loja": "Caxias do Sul",
    "faturamento": 171560.83,
    "pedidos": 3166,
    "ticket": 54.19,
    "periodo": "2025-06"
  },
  {
    "mes": 6,
    "ano": 2025,
    "loja": "Bento Gonçalves",
    "faturamento": 63012.98,
    "pedidos": 1211,
    "ticket": 52.03,
    "periodo": "2025-06"
  },
  {
    "mes": 6,
    "ano": 2025,
    "loja": "Lajeado",
    "faturamento": 49109.65,
    "pedidos": 971,
    "ticket": 50.58,
    "periodo": "2025-06"
  },
  {
    "mes": 6,
    "ano": 2025,
    "loja": "Erechim",
    "faturamento": 63328.41,
    "pedidos": 1396,
    "ticket": 45.36,
    "periodo": "2025-06"
  },
  {
    "mes": 6,
    "ano": 2025,
    "loja": "Protásio Alves",
    "faturamento": 98645.72,
    "pedidos": 2073,
    "ticket": 47.59,
    "periodo": "2025-06"
  },
  {
    "mes": 6,
    "ano": 2025,
    "loja": "Esteio",
    "faturamento": 93758.11,
    "pedidos": 2031,
    "ticket": 46.16,
    "periodo": "2025-06"
  },
  {
    "mes": 6,
    "ano": 2025,
    "loja": "Capão da Canoa",
    "faturamento": 46436.04,
    "pedidos": 1026,
    "ticket": 45.26,
    "periodo": "2025-06"
  },
  {
    "mes": 6,
    "ano": 2025,
    "loja": "Ijuí",
    "faturamento": 94006.48,
    "pedidos": 1721,
    "ticket": 54.62,
    "periodo": "2025-06"
  },
  {
    "mes": 7,
    "ano": 2025,
    "loja": "Montenegro",
    "faturamento": 85435.0,
    "pedidos": 1690,
    "ticket": 50.55,
    "periodo": "2025-07"
  },
  {
    "mes": 7,
    "ano": 2025,
    "loja": "Canoas",
    "faturamento": 87936.0,
    "pedidos": 1762,
    "ticket": 49.91,
    "periodo": "2025-07"
  },
  {
    "mes": 7,
    "ano": 2025,
    "loja": "Porto Alegre Zona Norte",
    "faturamento": 9833.0,
    "pedidos": 214,
    "ticket": 45.95,
    "periodo": "2025-07"
  },
  {
    "mes": 7,
    "ano": 2025,
    "loja": "São Leopoldo",
    "faturamento": 89454.0,
    "pedidos": 1738,
    "ticket": 51.47,
    "periodo": "2025-07"
  },
  {
    "mes": 7,
    "ano": 2025,
    "loja": "Novo Hamburgo",
    "faturamento": 103905.0,
    "pedidos": 1940,
    "ticket": 53.56,
    "periodo": "2025-07"
  },
  {
    "mes": 7,
    "ano": 2025,
    "loja": "Barra Shopping",
    "faturamento": 92303.0,
    "pedidos": 2807,
    "ticket": 32.88,
    "periodo": "2025-07"
  },
  {
    "mes": 7,
    "ano": 2025,
    "loja": "Porto Alegre Floresta",
    "faturamento": 62066.0,
    "pedidos": 1390,
    "ticket": 44.65,
    "periodo": "2025-07"
  },
  {
    "mes": 7,
    "ano": 2025,
    "loja": "Sapiranga",
    "faturamento": 83595.0,
    "pedidos": 1786,
    "ticket": 46.81,
    "periodo": "2025-07"
  },
  {
    "mes": 7,
    "ano": 2025,
    "loja": "Gravataí",
    "faturamento": 46073.0,
    "pedidos": 930,
    "ticket": 49.54,
    "periodo": "2025-07"
  },
  {
    "mes": 7,
    "ano": 2025,
    "loja": "Caxias do Sul",
    "faturamento": 150157.0,
    "pedidos": 2532,
    "ticket": 59.3,
    "periodo": "2025-07"
  },
  {
    "mes": 7,
    "ano": 2025,
    "loja": "Bento Gonçalves",
    "faturamento": 68375.0,
    "pedidos": 1279,
    "ticket": 53.46,
    "periodo": "2025-07"
  },
  {
    "mes": 7,
    "ano": 2025,
    "loja": "Lajeado",
    "faturamento": 46170.0,
    "pedidos": 921,
    "ticket": 50.13,
    "periodo": "2025-07"
  },
  {
    "mes": 7,
    "ano": 2025,
    "loja": "Erechim",
    "faturamento": 58233.0,
    "pedidos": 1334,
    "ticket": 43.65,
    "periodo": "2025-07"
  },
  {
    "mes": 7,
    "ano": 2025,
    "loja": "Protásio Alves",
    "faturamento": 93246.0,
    "pedidos": 2025,
    "ticket": 46.05,
    "periodo": "2025-07"
  },
  {
    "mes": 7,
    "ano": 2025,
    "loja": "Esteio",
    "faturamento": 92955.0,
    "pedidos": 1954,
    "ticket": 47.57,
    "periodo": "2025-07"
  },
  {
    "mes": 7,
    "ano": 2025,
    "loja": "Capão da Canoa",
    "faturamento": 43761.0,
    "pedidos": 953,
    "ticket": 45.92,
    "periodo": "2025-07"
  },
  {
    "mes": 7,
    "ano": 2025,
    "loja": "Ijuí",
    "faturamento": 84184.0,
    "pedidos": 1505,
    "ticket": 55.94,
    "periodo": "2025-07"
  }
];