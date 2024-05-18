# Installation

```bash
pip install -r requirements.txt
```

Download LibriTTS weights from [link](https://drive.google.com/drive/folders/1DOhZGlTLMbbAAFZmZGDdc77kz1PloS7F?usp=sharing) and put them at project root.
```bash
mv LibriTTS_800000.pth.tar 800000.pth.tar
cp 800000.pth.tar ./FastSpeech2/FastSpeech2/output/ckpt/LibriTTS/
```

Unpack archives from ./FastSpeech2/FastSpeech2/hifigan
```bash
cd ./FastSpeech2/FastSpeech2/hifigan
tar -xf generator_LJSpeech.pth.tar.zip
tar -xf generator_universal.pth.tar.zip
cd -
```

Create Fastspeech2 package
```
cd FastSpeech2
python setup.py sdist bdist_wheel
pip install .
cd -
```

Test FastSpeech2
```
python test.py
```

Create docker triton image
```
docker build -t fast-speech-triton .
```

Create a container (only http port exposed)
```
docker run -d -v ./models_repository:/models --name test-fast-speech-triton -p 8000:8000 fast-speech-triton
```

Test server
```
python client.py
```
