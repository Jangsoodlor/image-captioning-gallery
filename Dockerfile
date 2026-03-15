FROM docker.io/python:3.12-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1\
  PYTHONUNBUFFERED=1

COPY ./requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app3.py .
COPY image_captioning.py .
COPY gunicorn.conf.py .
COPY templates/ templates/
EXPOSE 5000
CMD ["gunicorn", "app:app", "-c", "gunicorn.conf.py"]