from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
import shutil
import uuid
import os
from processamento import corrigir_planilha_com_ia

app = FastAPI(title="API de Correção de Hodômetro")

# Libera CORS para o frontend (inclusive dentro do Docker)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Em produção, coloque o domínio correto
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/api/corrigir/")
async def corrigir_hodometro(file: UploadFile = File(...)):
    try:
        ext = file.filename.split(".")[-1]
        temp_path = f"/tmp/{uuid.uuid4()}.{ext}"
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        df_corrigido = corrigir_planilha_com_ia(temp_path)
        os.remove(temp_path)

        if df_corrigido is None or df_corrigido.empty:
            raise HTTPException(status_code=400, detail="Erro ao processar a planilha")

        data_json = jsonable_encoder(df_corrigido.to_dict(orient="records"))
        return {"dados": data_json}

    except HTTPException as e:
        raise e
    except Exception as e:
        return JSONResponse(content={"erro": str(e)}, status_code=500)
