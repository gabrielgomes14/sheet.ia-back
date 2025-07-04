import os
import logging
import sqlite3
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import joblib

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "data", "hodometro.db")
MODEL_DIR = os.path.join(BASE_DIR, "modelos_hodometro")

os.makedirs(MODEL_DIR, exist_ok=True)
logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")


# ------------------------------------------------------------------ #
# 1. BANCO DE DADOS
# ------------------------------------------------------------------ #
def criar_tabela_se_nao_existe():
    with sqlite3.connect(DB_PATH) as conn:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS hodometro_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                placa TEXT NOT NULL,
                hodometro INTEGER NOT NULL,
                data DATE NOT NULL
            )
        """
        )
        conn.commit()


def inserir_planilha_no_banco(df: pd.DataFrame):
    with sqlite3.connect(DB_PATH) as conn:
        df_exist = pd.read_sql(
            "SELECT placa, data, hodometro FROM hodometro_data", conn
        )
        df_exist["data"] = pd.to_datetime(df_exist["data"])

        df_novo = df.merge(
            df_exist,
            how="left",
            left_on=["Placa", "Data", "hodometro"],
            right_on=["placa", "data", "hodometro"],
            indicator=True,
        )
        df_novo = df_novo[df_novo["_merge"] == "left_only"]

        if not df_novo.empty:
            conn.executemany(
                "INSERT INTO hodometro_data (placa, hodometro, data) VALUES (?, ?, ?)",
                [
                    (r["Placa"], int(r["hodometro"]), r["Data"].strftime("%Y-%m-%d"))
                    for _, r in df_novo.iterrows()
                ],
            )
            conn.commit()
            logging.info(f"{len(df_novo)} novos registros inseridos.")
        else:
            logging.info("Nenhum registro novo para inserir.")

def buscar_dados_historicos() -> pd.DataFrame:
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql(
            "SELECT placa AS Placa, hodometro, data AS Data FROM hodometro_data", conn
        )
    df["Data"] = pd.to_datetime(df["Data"])
    return df

# ------------------------------------------------------------------ #
# 2. FUNÇÕES DE IA / CORREÇÃO
# ------------------------------------------------------------------ #
def calcular_limite_km_por_dia(df: pd.DataFrame, col: str = "hodometro") -> float:
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
    return joblib.load(caminho_modelo_placa(placa)) if os.path.exists(caminho_modelo_placa(placa)) else None

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

def corrigir_grupo(df_grp: pd.DataFrame, hist: pd.DataFrame, col="hodometro"):
    df_grp = df_grp.sort_values("Data").reset_index(drop=True)
    hod = df_grp[col].tolist()
    datas = df_grp["Data"].tolist()
    placa = df_grp.iloc[0]["Placa"]

    y_corr = [hod[0]]
    t0 = datas[0]
    dias = [(d - t0).days for d in datas]

    max_km_dia = calcular_limite_km_por_dia(hist[hist["Placa"] == placa], col)
    mdl, scl = treinar_modelo(y_corr, dias[:1], carregar_modelo_existente(placa))

    for i in range(1, len(hod)):
        dias_int = dias[i] - dias[i - 1]
        aumento_max = max_km_dia * dias_int
        aumento_real = hod[i] - y_corr[-1]

        if aumento_real < 0 or aumento_real > aumento_max:
            logging.warning(
                f"[{placa}] linha {i}: aumento {aumento_real} fora do limite {aumento_max}"
            )
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
# 3. PIPELINE PRINCIPAL
# ------------------------------------------------------------------ #
def corrigir_planilha_com_ia(path: str, col: str = "hodometro") -> pd.DataFrame | None:
    criar_tabela_se_nao_existe()

    try:
        if path.lower().endswith((".xlsx", ".xls")):
            df = pd.read_excel(path)
        else:
            df = pd.read_csv(path)
    except Exception as e:
        logging.error(f"Falha ao ler {path}: {e}")
        return None

    # normaliza nomes das colunas para minúsculo e remoção de acentos simples
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace("ô", "o")
        .str.replace("ó", "o")
    )

    # === Formatar coluna CNPJ se existir ===
    def formatar_cnpj(cnpj):
        if pd.isna(cnpj):
            return ""
        cnpj = ''.join(filter(str.isdigit, str(cnpj)))  # só números
        cnpj = cnpj.zfill(14)  # completa zeros à esquerda até 14 dígitos
        if len(cnpj) == 14:
            return f"{cnpj[:2]}.{cnpj[2:5]}.{cnpj[5:8]}/{cnpj[8:12]}-{cnpj[12:]}"
        else:
            return cnpj

    if "cnpj" in df.columns:
        df["cnpj"] = df["cnpj"].apply(formatar_cnpj)
        df.rename(columns={"cnpj": "CNPJ"}, inplace=True)

    if {"placa", "data", col}.difference(df.columns):
        logging.error("Planilha deve conter 'Placa', 'Data' e 'Hodometro'")
        return None

    df["data"] = pd.to_datetime(df["data"], dayfirst=True, errors="coerce")
    df = df.dropna(subset=["data"])
    df = df.rename(columns={"placa": "Placa", "data": "Data"})

    inserir_planilha_no_banco(df)
    hist = buscar_dados_historicos()

    corrigidos = [
        corrigir_grupo(grp.copy(), hist, col) for _, grp in df.groupby("Placa")
    ]
    df_final = pd.concat(corrigidos).sort_values(["Placa", "Data"]).reset_index(
        drop=True
    )

    # converte Data para string e troca NaN → None para garantir JSON OK
    df_final["Data"] = df_final["Data"].dt.strftime("%Y-%m-%d")
    df_final = df_final.replace({np.nan: None})

    return df_final


if __name__ == "__main__":
    caminho = "/home/gomesgalikosky/Downloads/hodometro_exemplo.xlsx"
    df_corrigido = corrigir_planilha_com_ia(caminho)
    if df_corrigido is not None:
        df_corrigido.to_excel("corrigido_db_aprendido.xlsx", index=False)
        logging.info("Arquivo corrigido salvo como corrigido_db_aprendido.xlsx")
