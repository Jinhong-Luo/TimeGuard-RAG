FROM python:3.12-slim
WORKDIR /opt/timeguard
COPY requirements-app.txt .
RUN pip install --no-cache-dir -r requirements-app.txt
COPY timeguard/ timeguard/
COPY configs/r0-low.json configs/filing_dates.json configs/
ENV PYTHONDONTWRITEBYTECODE=1
CMD ["uvicorn", "timeguard.app:create_app", "--factory", "--host", "127.0.0.1", "--port", "19112", "--workers", "1"]

