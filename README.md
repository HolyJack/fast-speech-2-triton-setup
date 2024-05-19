# Installation

Install client.py dependencies
```bash
pip install -r requirements.txt
```

Download LibriTTS weights from [link](https://drive.google.com/drive/folders/1DOhZGlTLMbbAAFZmZGDdc77kz1PloS7F?usp=sharing) and put them at project root.
```bash
mv LibriTTS_800000.pth.tar ./models_repository/fast-speech-2/
```

Unpack archives from ./FastSpeech2/FastSpeech2/hifigan
```bash
cd ./FastSpeech2/FastSpeech2/hifigan
tar -xf generator_universal.pth.tar.zip
cd -
mv ./FastSpeech2/FastSpeech2/hifigan/generator_universal.pth.tar ./models_repository/hifigan_vocoder/
```

Create a triton inference server image and container
```
docker compose up -d
```

Test server
```
python client.py
```
