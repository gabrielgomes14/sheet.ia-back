import os
import logging
import sqlite3
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import joblib

DB_PATH = 'hodometro.db'
MODEL_DIR = 'modelos_hodometro'  # pasta para salvar modelos

if not os.path.exists(MODEL_DIR):
    os.makedirs(MODEL_DIR)

logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(message)s')


def criar_tabela_se_nao_existe():
    """Cria a tabela hodometro_data se não existir no banco SQLite."""
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS hodometro_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                placa TEXT NOT NULL,
                hodometro INTEGER NOT NULL,
                data DATE NOT NULL
            )
        ''')
        conn.commit()


def inserir_planilha_no_banco(df):
    """
    Insere os dados do DataFrame no banco, evitando duplicações.
    Usa inserção em lote para melhor performance.
    """
    with sqlite3.connect(DB_PATH) as conn:
        cursor = conn.cursor()
        df_existente = pd.read_sql_query('SELECT placa, data, hodometro FROM hodometro_data', conn)
        df_existente['data'] = pd.to_datetime(df_existente['data'])

        # Merge para identificar registros novos
        df_novo = df.merge(
            df_existente,
            how='left',
            left_on=['Placa', 'Data', 'hodometro'],
            right_on=['placa', 'data', 'hodometro'],
            indicator=True
        )
        df_novo = df_novo[df_novo['_merge'] == 'left_only']

        dados_inserir = [
            (row['Placa'], int(row['hodometro']), row['Data'].strftime('%Y-%m-%d'))
            for _, row in df_novo.iterrows()
        ]

        if dados_inserir:
            cursor.executemany(
                'INSERT INTO hodometro_data (placa, hodometro, data) VALUES (?, ?, ?)',
                dados_inserir
            )
            conn.commit()
            logging.info(f"{len(dados_inserir)} registros inseridos no banco.")
        else:
            logging.info("Nenhum registro novo para inserir.")


def buscar_dados_historicos():
    """Busca todos os dados históricos do banco em um DataFrame."""
    with sqlite3.connect(DB_PATH) as conn:
        df = pd.read_sql_query('SELECT placa AS Placa, hodometro, data AS Data FROM hodometro_data', conn)
    df['Data'] = pd.to_datetime(df['Data'])
    return df


def calcular_limite_km_por_dia(df_grupo, nome_coluna='hodometro'):
    """
    Calcula o limite máximo diário de km baseado no percentil 95 das velocidades históricas.
    """
    df_grupo = df_grupo.sort_values('Data').reset_index(drop=True)
    df_grupo['diff_dias'] = df_grupo['Data'].diff().dt.days
    df_grupo['diff_km'] = df_grupo[nome_coluna].diff()

    velocidades = df_grupo.loc[
        (df_grupo['diff_dias'] > 0) & (df_grupo['diff_km'] >= 0),
        'diff_km'
    ] / df_grupo.loc[
        (df_grupo['diff_dias'] > 0) & (df_grupo['diff_km'] >= 0),
        'diff_dias'
    ]

    if velocidades.empty:
        return 200

    limite = np.percentile(velocidades, 95)
    return max(limite, 50)


def caminho_modelo_placa(placa):
    """Retorna o caminho do arquivo de modelo para uma placa."""
    return os.path.join(MODEL_DIR, f'modelo_{placa}.pkl')


def carregar_modelo_existente(placa):
    """Carrega o modelo salvo para uma placa, se existir."""
    path = caminho_modelo_placa(placa)
    if os.path.exists(path):
        return joblib.load(path)
    return None


def salvar_modelo(placa, modelo_scaler_tuple):
    """Salva o modelo e scaler para uma placa."""
    path = caminho_modelo_placa(placa)
    joblib.dump(modelo_scaler_tuple, path)


def treinar_modelo(valores_treinamento, dias_treinamento, modelo_existente=None):
    """
    Treina (ou retreina) o modelo linear com dados fornecidos.
    """
    X = np.array(dias_treinamento).reshape(-1, 1)
    y = np.array(valores_treinamento)

    if modelo_existente is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        modelo = LinearRegression()
        modelo.fit(X_scaled, y)
    else:
        modelo, scaler = modelo_existente
        X_scaled = scaler.transform(X)
        modelo.fit(X_scaled, y)

    return modelo, scaler


def corrigir_grupo(df_grupo, df_historico, nome_coluna='hodometro'):
    """
    Corrige os valores de hodômetro para um grupo (placa) específico.
    """
    df_grupo = df_grupo.sort_values("Data").reset_index(drop=True)
    hodometros = df_grupo[nome_coluna].tolist()
    datas = df_grupo['Data'].tolist()
    placa = df_grupo.iloc[0]['Placa']

    corrigidos = [hodometros[0]]
    primeira_data = datas[0]
    dias = [
        (d - primeira_data).days if isinstance(d - primeira_data, pd.Timedelta) else ((d - primeira_data) / np.timedelta64(1, 'D'))
        for d in datas
    ]

    historico_placa = df_historico[df_historico['Placa'] == placa]
    max_km_por_dia = calcular_limite_km_por_dia(historico_placa, nome_coluna) if len(historico_placa) > 0 else 200

    modelo_existente = carregar_modelo_existente(placa)
    modelo, scaler = treinar_modelo(corrigidos, dias[:1], modelo_existente)

    for i in range(1, len(hodometros)):
        dias_entre_pontos = dias[i] - dias[i-1]
        aumento_maximo = max_km_por_dia * dias_entre_pontos
        aumento_real = hodometros[i] - corrigidos[-1]

        if aumento_real < 0 or aumento_real > aumento_maximo:
            logging.warning(f"[{placa}] ERRO linha {i}: aumento {aumento_real} fora do limite [0, {aumento_maximo}]")

            modelo, scaler = treinar_modelo(corrigidos, dias[:i], (modelo, scaler))
            X_pred_scaled = scaler.transform([[dias[i]]])
            pred = modelo.predict(X_pred_scaled)[0]

            if pred < corrigidos[-1]:
                valor_corrigido = int(corrigidos[-1])
            elif (pred - corrigidos[-1]) > aumento_maximo:
                valor_corrigido = int(corrigidos[-1] + aumento_maximo)
            else:
                valor_corrigido = int(pred)

            logging.info(f"[{placa}] [CORRIGIDO] Ajustado para {valor_corrigido}")
            corrigidos.append(valor_corrigido)
        else:
            corrigidos.append(hodometros[i])

    modelo, scaler = treinar_modelo(corrigidos, dias, (modelo, scaler))
    salvar_modelo(placa, (modelo, scaler))  # salva modelo atualizado

    df_grupo[nome_coluna] = corrigidos
    return df_grupo


def corrigir_planilha_com_ia(caminho_arquivo, nome_coluna='hodometro'):
    """
    Faz todo o processo de ler o arquivo, inserir no banco, buscar histórico e corrigir os dados.
    Retorna o DataFrame corrigido.
    """
    criar_tabela_se_nao_existe()

    try:
        if caminho_arquivo.endswith(".xlsx"):
            df = pd.read_excel(caminho_arquivo)
        else:
            df = pd.read_csv(caminho_arquivo)
    except Exception as e:
        logging.error(f"Erro ao ler arquivo {caminho_arquivo}: {e}")
        return None

    if 'Data' not in df.columns or 'Placa' not in df.columns or nome_coluna not in df.columns:
        logging.error("Arquivo deve conter as colunas: 'Placa', 'Data' e 'hodometro'")
        return None

    df['Data'] = pd.to_datetime(df['Data'], dayfirst=True, errors='coerce')
    if df['Data'].isnull().any():
        logging.warning("Existem datas inválidas no arquivo. Linhas com datas inválidas serão ignoradas.")
        df = df.dropna(subset=['Data'])

    inserir_planilha_no_banco(df)
    df_historico = buscar_dados_historicos()

    grupos_corrigidos = []
    for placa, grupo in df.groupby("Placa"):
        grupo_corrigido = corrigir_grupo(grupo.copy(), df_historico, nome_coluna)
        grupos_corrigidos.append(grupo_corrigido)

    df_final = pd.concat(grupos_corrigidos).sort_values(by=["Placa", "Data"]).reset_index(drop=True)
    return df_final


if __name__ == "__main__":
    caminho = "/home/gomesgalikosky/Downloads/hodometro_exemplo.xlsx"
    df_corrigido = corrigir_planilha_com_ia(caminho)
    if df_corrigido is not None:
        df_corrigido.to_excel("corrigido_db_aprendido.xlsx", index=False)
        logging.info("Arquivo corrigido salvo como corrigido_db_aprendido.xlsx")
