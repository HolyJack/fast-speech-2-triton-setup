# Installation

Install client.py dependencies
```bash
pip install -r requirements.txt
```

Download LibriTTS weights from [link](https://drive.google.com/drive/folders/1DOhZGlTLMbbAAFZmZGDdc77kz1PloS7F?usp=sharing) and put them at project root.
```bash
mv LibriTTS_800000.pth.tar ./models_repository/fast-speech-2/
```

Unpack archives from ./models_repository/hifigan_vocoder
```bash
tar -xf ./models_repository/hifigan_vocoder/generator_universal.pth.tar.zip
```

Create a triton inference server image and container
```
docker compose up -d
```

Test server
```
python client.py
```
