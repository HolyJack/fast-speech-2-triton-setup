FROM nvcr.io/nvidia/tritonserver:24.04-pyt-python-py3

ADD ./FastSpeech2 /FastSpeech2
ADD ./model.yaml /model.yaml
ADD ./800000.pth.tar /800000.pth.tar
ADD ./preprocess.yaml /preprocess.yaml

WORKDIR /FastSpeech2
RUN pip install . && pip install -r requirements.txt

EXPOSE 8000 8001 8002

CMD [ "tritonserver", "--model-repository=/models" ]
