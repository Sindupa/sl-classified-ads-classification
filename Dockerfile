FROM tensorflow/tensorflow:2.16.1

WORKDIR /app

COPY . /app

CMD ["python"]