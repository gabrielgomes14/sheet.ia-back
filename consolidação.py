import pandas as pd
import os

def unificar_planilhas(arquivos: list, saida: str = "/tmp/planilha_unificada_por_placa.xlsx") -> pd.DataFrame:
    dfs = []
    for caminho in arquivos:
        nome_arquivo = os.path.basename(caminho)
        if "posto" in nome_arquivo.lower():
            fonte = "Auto Posto"
            df = pd.read_excel(caminho, sheet_name=0)
            df_pad = df[["Frota", "Data documento", "Hodômetro - FMCP"]].copy()
            df_pad.columns = ["Placa", "Data", "Hodometro"]

        elif "florestal" in nome_arquivo.lower():
            fonte = "Alelo Florestal"
            df = pd.read_excel(caminho, sheet_name=0)
            df_pad = df[["Placa - Dig. Motorista", "Data/Hora Transação", "Hodômetro - Dig. Motorista"]].copy()
            df_pad.columns = ["Placa", "Data", "Hodometro"]

        elif "celulose" in nome_arquivo.lower():
            fonte = "Alelo Celulose"
            df = pd.read_excel(caminho, sheet_name=0)
            df_pad = df[["Placa - Dig. Motorista", "Data/Hora Transação", "Hodômetro - Dig. Motorista"]].copy()
            df_pad.columns = ["Placa", "Data", "Hodometro"]
        else:
            continue

        df_pad["Fonte"] = fonte
        dfs.append(df_pad)

    if not dfs:
        return pd.DataFrame()

    df_final = pd.concat(dfs, ignore_index=True)
    df_final["Data"] = pd.to_datetime(df_final["Data"], errors="coerce")
    df_final["Hodometro"] = pd.to_numeric(df_final["Hodometro"], errors="coerce")
    df_final["Placa"] = df_final["Placa"].astype(str).str.strip().str.upper()
    df_final = df_final.dropna(subset=["Placa", "Data"])
    df_final = df_final.sort_values(by=["Placa", "Data"]).reset_index(drop=True)

    df_final.to_excel(saida, index=False)
    return df_final
