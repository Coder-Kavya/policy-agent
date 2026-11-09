FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Build the vector store inside the image so the container starts ready to answer.
RUN python ingest.py

EXPOSE 8000

# The OpenAI key is NOT baked into the image. Pass it at run time.
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "8000"]
