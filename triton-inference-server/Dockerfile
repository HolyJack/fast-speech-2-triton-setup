FROM nvcr.io/nvidia/tritonserver:24.04-pyt-python-py3

ADD ./requirements.txt /requirements.txt
RUN pip install -r /requirements.txt

EXPOSE 8000 8001 8002

CMD [ "tritonserver", "--model-repository=/models" ]
