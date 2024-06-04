FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .env ./
RUN pip install --no-cache-dir -r requirements.txt \
    && apt-get update \
    && apt-get install -y jq tesseract-ocr libtesseract-dev \
    && apt-get install -y dos2unix
# Copy project files
COPY . .

# Set permissions for entrypoint
RUN dos2unix entrypoint.sh
RUN chmod +x entrypoint.sh

ENTRYPOINT ["./entrypoint.sh"]
