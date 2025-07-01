from fastapi import FastAPI, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import shutil
import uuid
import os
import pandas as pd
from processamento import corrigir_planilha_com_ia  # sua função

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/corrigir/")
async def corrigir_hodometro(file: UploadFile = File(...)):
    try:
        ext = file.filename.split('.')[-1]
        temp_path = f"/tmp/{uuid.uuid4()}.{ext}"
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        df_corrigido = corrigir_planilha_com_ia(temp_path)

        os.remove(temp_path)

        if df_corrigido is None or df_corrigido.empty:
            return JSONResponse(content={"erro": "Erro ao processar a planilha"}, status_code=400)

        # Retorna os dados em JSON para o frontend exibir
        return {"dados": df_corrigido.to_dict(orient="records")}
    except Exception as e:
        return JSONResponse(content={"erro": str(e)}, status_code=500)
