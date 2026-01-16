FROM python:3.10-slim

RUN apt-get update && apt-get install -y \
    dcm2niix \
    && rm -rf /var/lib/apt/lists/*

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


COPY core_function.py .
COPY data_cleaning.py .
COPY dcm2niix_p.py .  
COPY main.py .

CMD ["python", "main.py"]