FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

RUN mkdir -p sample_outputs

ENTRYPOINT ["python"]
CMD ["main.py", "--help"]