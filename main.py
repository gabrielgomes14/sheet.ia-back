from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Optional
import shutil
import uuid
import os
import pandas as pd

from processamento import corrigir_planilha_com_ia

app = FastAPI(title="API de Correção de Hodômetro")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Modelo mínimo só pra validar que tem as colunas esperadas
class LinhaCorrigida(BaseModel):
    Placa: str
    Data: str
    hodometro: Optional[int]

class RespostaPlanilha(BaseModel):
    dados: List[LinhaCorrigida]

@app.post("/api/corrigir/")
async def corrigir_hodometro(file: UploadFile = File(...)):
    try:
        ext = file.filename.split(".")[-1]
        temp_path = f"/tmp/{uuid.uuid4()}.{ext}"
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        df_corrigido = corrigir_planilha_com_ia(temp_path)

        if df_corrigido is None or df_corrigido.empty:
            raise HTTPException(status_code=400, detail="Erro ao processar a planilha")

        os.remove(temp_path)

        # Renomeia "Hodômetro" → "hodometro" apenas se ele existir
        if "Hodômetro" in df_corrigido.columns:
            df_corrigido = df_corrigido.rename(columns={"Hodômetro": "hodometro"})

        # Validação básica
        colunas_essenciais = {"Placa", "Data", "hodometro"}
        if not colunas_essenciais.issubset(df_corrigido.columns):
            raise HTTPException(status_code=500, detail="Colunas essenciais ausentes na planilha")

        # Formata e retorna tudo
        df_corrigido["hodometro"] = df_corrigido["hodometro"].astype("Int64")
        df_corrigido["Data"] = df_corrigido["Data"].astype(str)
        return JSONResponse(content={"dados": df_corrigido.to_dict(orient="records")})

    except HTTPException as e:
        raise e
    except Exception as e:
        return JSONResponse(content={"erro": str(e)}, status_code=500)
