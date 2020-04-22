# Pull a pre-built alpine docker image with nginx and python3 installed
FROM tiangolo/uwsgi-nginx-flask:python3.7-alpine3.7

ENV LISTEN_PORT=8000
EXPOSE 8000

COPY /app /app
COPY /env /env

RUN python3 -m pip install --upgrade pip

# RUN pip install https://storage.googleapis.com/tensorflow/mac/cpu/tensorflow-1.8.0-py3-none-any.whl

# Uncomment to install additional requirements from a requirements.txt file
COPY requirements.txt /
RUN pip install --no-cache-dir -U pip
RUN pip install --no-cache-dir -r /requirements.txt