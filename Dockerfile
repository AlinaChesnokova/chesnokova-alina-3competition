FROM python:3.10-slim

# Устанавливаем системные зависимости для LightGBM
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    gcc \
    g++ \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Копируем зависимости
COPY requirements.txt .

# Устанавливаем Python зависимости
RUN pip install --no-cache-dir -r requirements.txt

# Копируем код
COPY . .

# Запускаем решение
CMD ["python", "main.py"]