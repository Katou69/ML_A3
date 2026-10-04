FROM python:3.13-slim

WORKDIR /app

COPY code/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY code/ .

EXPOSE 8050

CMD ["python", "app.py"]
