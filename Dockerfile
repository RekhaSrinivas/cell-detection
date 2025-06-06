FROM python:3.11-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --upgrade pip && pip install -r requirements.txt

COPY . .

CMD ["streamlit", "run", "app/Segment.py", "--server.port=8501", "--server.address=0.0.0.0"]
