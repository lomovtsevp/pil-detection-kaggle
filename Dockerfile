FROM python:3.11

WORKDIR /pii-detector

COPY requirements.txt /pii-detector/

RUN pip install -r requirements.txt

COPY . /pii-detector/

CMD [ "python3", "./preprocessing.py" ]