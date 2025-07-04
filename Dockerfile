FROM python:3.11-slim

# Diretório principal da aplicação
WORKDIR /app

# Copia as dependências
COPY requirements.txt .
RUN pip install --upgrade pip && pip install --no-cache-dir -r requirements.txt

# Copia todo o conteúdo do back (inclusive processamento.py, etc.)
COPY . .

# Expõe a porta usada pela FastAPI
EXPOSE 8000

# Comando para iniciar a API com Uvicorn
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
