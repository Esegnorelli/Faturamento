# treat_faturamento.py
"""
Tratamento da base Faturamento.csv
- Normaliza cabeçalhos (minúsculo, sem acentos, snake_case)
- Mapeia aliases (ex.: 'ticket médio' -> 'ticket')
- Converte mês (1..12, 'jan', 'janeiro' etc.) -> Int64
- Converte ano -> Int64
- Converte 'faturamento' e 'ticket' de BRL para float
- Converte 'pedidos' para inteiro
- Cria coluna 'periodo' no formato AAAA-MM
- Remove colunas duplicadas e 100% nulas
- Gera um CSV limpo

Uso:
    python treat_faturamento.py --input Faturamento.csv --output Faturamento_tratado.csv
"""

import re
import sys
import unicodedata
import argparse
import pandas as pd


# ---------------------------- Helpers ----------------------------------------
def normalize_col(name: str) -> str:
    """minúsculo, remove acentos, troca espaços por underscore."""
    name = name.strip().lower()
    name = "".join(c for c in unicodedata.normalize("NFKD", name) if not unicodedata.combining(c))
    name = re.sub(r"\s+", "_", name)
    return name

def br_to_float(series: pd.Series) -> pd.Series:
    """'R$ 1.234,56' -> 1234.56 | aceita NaN e floats/ints."""
    s = series.astype(str).str.strip()
    s = s.replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    # remove símbolos e letras, preserva dígitos, vírgula, ponto e sinal
    s = s.str.replace(r"[^0-9,.\-]", "", regex=True)
    # se tem vírgula decimal, remove pontos (milhar) e troca vírgula por ponto
    has_comma = s.str.contains(",", na=False)
    s = s.mask(has_comma, s.str.replace(".", "", regex=False))
    s = s.mask(has_comma, s.str.replace(",", ".", regex=False))
    return pd.to_numeric(s, errors="coerce")

def month_to_int(series: pd.Series) -> pd.Series:
    """Aceita 1..12, '01', 'jan', 'janeiro' etc. -> Int64"""
    mapa = {
        "jan":1,"janeiro":1,
        "fev":2,"fevereiro":2,
        "mar":3,"marco":3,"março":3,
        "abr":4,"abril":4,
        "mai":5,"maio":5,
        "jun":6,"junho":6,
        "jul":7,"julho":7,
        "ago":8,"agosto":8,
        "set":9,"setembro":9,"sep":9,
        "out":10,"outubro":10,
        "nov":11,"novembro":11,
        "dez":12,"dezembro":12
    }
    s = series.astype(str).str.strip().str.lower()
    s = s.apply(lambda x: mapa.get(x, x))
    return pd.to_numeric(s, errors="coerce").astype("Int64")

ALIASES = {
    "mes": ["mes", "mês", "month"],
    "ano": ["ano", "year"],
    "loja": ["loja", "filial", "store"],
    "faturamento": ["faturamento", "receita", "vendas", "valor", "total", "valor_total"],
    "pedidos": ["pedidos", "qtde_pedidos", "qtd_pedidos", "qtd", "quantidade_pedidos"],
    "ticket": ["ticket", "ticket_medio", "ticket_médio", "ticket medio", "ticket médio"]
}

def rename_by_alias(cols: list[str]) -> dict:
    """Gera dicionário de renome com base nos aliases acima."""
    ren = {}
    for c in cols:
        for target, opts in ALIASES.items():
            if c in opts:
                ren[c] = target
                break
    return ren


# ---------------------------- Core pipeline -----------------------------------
def process_df(df: pd.DataFrame) -> pd.DataFrame:
    # 1) normalizar cabeçalhos
    df.columns = [normalize_col(c) for c in df.columns]

    # 2) remover colunas duplicadas e 100% nulas
    df = df.loc[:, ~df.columns.duplicated()]
    df = df.dropna(axis=1, how="all")

    # 3) renomear por aliases
    df = df.rename(columns=rename_by_alias(list(df.columns)))

    # 4) garantir colunas alvo
    for col in ["mes", "ano", "loja", "faturamento", "pedidos", "ticket"]:
        if col not in df.columns:
            df[col] = pd.NA

    # 5) tipagem/limpeza
    df["mes"] = month_to_int(df["mes"])
    df["ano"] = pd.to_numeric(df["ano"], errors="coerce").astype("Int64")
    df["loja"] = df["loja"].astype(str).str.strip()
    df["faturamento"] = br_to_float(df["faturamento"])
    df["ticket"] = br_to_float(df["ticket"])
    df["pedidos"] = pd.to_numeric(df["pedidos"], errors="coerce").round().astype("Int64")

    # 6) criar 'periodo' (AAAA-MM)
    mask = df["ano"].notna() & df["mes"].notna()
    df["periodo"] = pd.NA
    df.loc[mask, "periodo"] = (
        pd.PeriodIndex(
            year=df.loc[mask, "ano"].astype(int),
            month=df.loc[mask, "mes"].astype(int),
            freq="M"
        ).astype(str)
    )

    return df


# ---------------------------- CLI --------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Tratar base de faturamento.")
    parser.add_argument("--input", "-i", default="Faturamento.csv", help="Caminho do CSV de entrada.")
    parser.add_argument("--output", "-o", default="Faturamento.csv", help="Caminho do CSV de saída.")
    parser.add_argument("--encoding", default="utf-8", help="Encoding do arquivo (ex.: latin1).")
    args = parser.parse_args()

    try:
        # sep=None detecta automaticamente vírgula ou ponto-e-vírgula
        df_raw = pd.read_csv(args.input, sep=None, engine="python", encoding=args.encoding)
    except Exception as e:
        print(f"[ERRO] Falha ao ler '{args.input}': {e}")
        sys.exit(1)

    df = process_df(df_raw)

    # salvar
    try:
        df.to_csv(args.output, index=False)
        print(f"[OK] Arquivo tratado salvo em: {args.output}")
    except Exception as e:
        print(f"[ERRO] Falha ao salvar '{args.output}': {e}")
        sys.exit(1)

    # relatório rápido
    print("\n[INFO] Estrutura final:")
    print(df.info())
    print("\n[HEAD]")
    print(df.head(10).to_string(index=False))


if __name__ == "__main__":
    main()
