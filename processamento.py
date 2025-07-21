import os
import logging
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import joblib
from sqlalchemy import create_engine, text
from psycopg2.extras import execute_values
import unidecode

# --- Configuração do banco PostgreSQL ---
DB_USER = os.getenv("POSTGRES_USER", "user")
DB_PASSWORD = os.getenv("POSTGRES_PASSWORD", "password")
DB_HOST = os.getenv("POSTGRES_HOST", "db")
DB_PORT = os.getenv("POSTGRES_PORT", "5432")
DB_NAME = os.getenv("POSTGRES_DB", "hodometro_db")

DATABASE_URL = f"postgresql+psycopg2://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
engine = create_engine(DATABASE_URL)

# Diretórios
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "modelos_hodometro")
os.makedirs(MODEL_DIR, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

# ------------------------------------------------------------------ #
# 1. BANCO DE DADOS
# ------------------------------------------------------------------ #
def criar_tabela_se_nao_existe():
    with engine.begin() as conn:
        conn.execute(text("""
            CREATE TABLE IF NOT EXISTS hodometro_data (
                id SERIAL PRIMARY KEY,
                placa TEXT NOT NULL,
                hodometro INTEGER NOT NULL,
                data DATE NOT NULL
            );
        """))

def inserir_planilha_no_banco(df: pd.DataFrame):
    with engine.begin() as conn:
        df_exist = pd.read_sql("SELECT placa AS placa, data AS data, hodometro FROM hodometro_data", conn)
        df_exist["data"] = pd.to_datetime(df_exist["data"])

        df_novo = df.merge(
            df_exist,
            how="left",
            left_on=["Placa", "Data", "Hodômetro"],
            right_on=["placa", "data", "hodometro"],
            indicator=True
        )
        df_novo = df_novo[df_novo["_merge"] == "left_only"]

        if not df_novo.empty:
            values = [(r["Placa"], int(r["Hodômetro"]), r["Data"]) for _, r in df_novo.iterrows()]
            print(values)
            with conn.connection.cursor() as cur:
                execute_values(
                    cur,
                    "INSERT INTO hodometro_data (placa, hodometro, data) VALUES %s",
                    values
                )
            logging.info(f"{len(df_novo)} novos registros inseridos.")
        else:
            logging.info("Nenhum registro novo para inserir.")

def buscar_dados_historicos() -> pd.DataFrame:
    with engine.connect() as conn:
        df = pd.read_sql("SELECT placa, hodometro, data FROM hodometro_data", conn)
    df.rename(columns={"placa": "Placa", "hodometro": "Hodômetro", "data": "Data"}, inplace=True)
    df["Data"] = pd.to_datetime(df["Data"])
    return df

# ------------------------------------------------------------------ #
# 2. FUNÇÕES DE IA / CORREÇÃO
# ------------------------------------------------------------------ #
def calcular_limite_km_por_dia(df: pd.DataFrame, col: str = "Hodômetro") -> float:
    df = df.sort_values("Data").reset_index(drop=True)
    df["diff_dias"] = df["Data"].diff().dt.days
    df["diff_km"] = df[col].diff()

    vel = df.loc[(df["diff_dias"] > 0) & (df["diff_km"] >= 0), "diff_km"] / df.loc[
        (df["diff_dias"] > 0) & (df["diff_km"] >= 0), "diff_dias"
    ]
    if vel.empty:
        return 200.0
    return max(np.percentile(vel, 95), 50.0)

def caminho_modelo_placa(placa: str) -> str:
    return os.path.join(MODEL_DIR, f"modelo_{placa}.pkl")

def carregar_modelo_existente(placa: str):
    caminho = caminho_modelo_placa(placa)
    return joblib.load(caminho) if os.path.exists(caminho) else None

def salvar_modelo(placa: str, modelo_scaler):
    joblib.dump(modelo_scaler, caminho_modelo_placa(placa))

def treinar_modelo(y, dias, modelo_scaler=None):
    X = np.array(dias).reshape(-1, 1)
    y = np.array(y)
    if modelo_scaler is None:
        scaler = StandardScaler()
        Xs = scaler.fit_transform(X)
        mdl = LinearRegression().fit(Xs, y)
    else:
        mdl, scaler = modelo_scaler
        Xs = scaler.transform(X)
        mdl.fit(Xs, y)
    return mdl, scaler

def corrigir_grupo(df_grp: pd.DataFrame, hist: pd.DataFrame, col="Hodômetro"):
    df_grp = df_grp.sort_values("Data").reset_index(drop=True)
    hod = df_grp[col].tolist()
    datas = df_grp["Data"].tolist()
    placa = df_grp.iloc[0]["Placa"]

    y_corr = [hod[0]]
    t0 = datas[0]
    dias = [(d - t0).total_seconds() / 86400 for d in datas]  # 86400 = segundos em 1 dia

    max_km_dia = calcular_limite_km_por_dia(hist[hist["Placa"] == placa], col)
    mdl, scl = treinar_modelo(y_corr, dias[:1], carregar_modelo_existente(placa))

    for i in range(1, len(hod)):
        dias_int = dias[i] - dias[i - 1]
        aumento_max = max_km_dia * dias_int
        aumento_real = hod[i] - y_corr[-1]

        if aumento_real < 0 or aumento_real > aumento_max:
            logging.warning(f"[{placa}] linha {i}: aumento {aumento_real} fora do limite {aumento_max}")
            mdl, scl = treinar_modelo(y_corr, dias[:i], (mdl, scl))
            pred = mdl.predict(scl.transform([[dias[i]]]))[0]

            if pred < y_corr[-1]:
                novo = int(y_corr[-1])
            elif pred - y_corr[-1] > aumento_max:
                novo = int(y_corr[-1] + aumento_max)
            else:
                novo = int(pred)
            logging.info(f"[{placa}] ajustado para {novo}")
            y_corr.append(novo)
        else:
            y_corr.append(hod[i])

    mdl, scl = treinar_modelo(y_corr, dias, (mdl, scl))
    salvar_modelo(placa, (mdl, scl))
    df_grp[col] = y_corr
    return df_grp

# ------------------------------------------------------------------ #
# 3. FORMATADOR DE CNPJ
# ------------------------------------------------------------------ #
def formatar_cnpj(cnpj):
    cnpj_str = str(cnpj).zfill(14)
    return f"{cnpj_str[:2]}.{cnpj_str[2:5]}.{cnpj_str[5:8]}/{cnpj_str[8:12]}-{cnpj_str[12:]}"

# ------------------------------------------------------------------ #
# 4. PIPELINE PRINCIPAL
# ------------------------------------------------------------------ #
def corrigir_planilha_com_ia(path: str, col: str = "Hodômetro") -> pd.DataFrame | None:
    criar_tabela_se_nao_existe()

    try:
        df = pd.read_excel(path) if path.lower().endswith((".xlsx", ".xls")) else pd.read_csv(path)
    except Exception as e:
        logging.error(f"Falha ao ler {path}: {e}")
        return None

    print("Colunas originais da planilha:", df.columns.tolist())
    df.columns = df.columns.str.strip()
    print("Colunas após strip:", df.columns.tolist())

    # Mapeamento de nomes alternativos
    colunas_aceitas = {
        "Data": ["Data", "Data/Hora Transação", "Data Transação"],
        "Placa": ["Placa"],
        "Hodômetro": ["Hodômetro", "Hodômetro - Dig. Motorista", "HODOMETRO OU HORIMETRO"],
    }

    def encontrar_coluna(df_cols, possiveis):
        for col in possiveis:
            if col in df_cols:
                return col
        return None

    col_data = encontrar_coluna(df.columns, colunas_aceitas["Data"])
    col_placa = encontrar_coluna(df.columns, colunas_aceitas["Placa"])
    col_hod = encontrar_coluna(df.columns, colunas_aceitas["Hodômetro"])

    if not all([col_data, col_placa, col_hod]):
        logging.error(f"Planilha deve conter colunas compatíveis com: Data, Placa, Hodômetro.")
        return None

    # Renomeia para nomes padrão
    df = df.rename(columns={col_data: "Data", col_placa: "Placa", col_hod: "Hodômetro"})

    df["Data"] = pd.to_datetime(df["Data"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["Data"])

    inserir_planilha_no_banco(df)
    hist = buscar_dados_historicos()

    corrigidos = [
        corrigir_grupo(grp.copy(), hist, col) for _, grp in df.groupby("Placa")
    ]

    df_final = pd.concat(corrigidos).sort_values(["Placa", "Data"]).reset_index(drop=True)

    df_final["Data"] = df_final["Data"].dt.strftime("%Y-%m-%d")
    df_final = df_final.replace({np.nan: None})

    # Aplica formatação no CNPJ se a coluna existir
    if "CNPJ" in df_final.columns:
        df_final["CNPJ"] = df_final["CNPJ"].apply(formatar_cnpj)

    return df_final
