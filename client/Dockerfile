FROM python:3.10.14-bullseye

WORKDIR /client
ADD ./requirements.txt ./requirements.txt
RUN pip install -r requirements.txt

ADD ./client.py ./client.py
RUN mkdir output

ENTRYPOINT ["python", "client.py"]
