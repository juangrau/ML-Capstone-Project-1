FROM python:3.10.12

WORKDIR /app
COPY ["requirements.txt", "./"]

RUN pip install -r requirements.txt

COPY ["predict.py", "terrain-classification.tflite", "./"]

EXPOSE 9696

ENTRYPOINT [ "gunicorn", "--bind=0.0.0.0:9696", "predict:app" ]