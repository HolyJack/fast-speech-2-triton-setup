# Installation

Download LibriTTS weights from [link](https://drive.google.com/drive/folders/1DOhZGlTLMbbAAFZmZGDdc77kz1PloS7F?usp=sharing) and put them at project root.
```bash
mv LibriTTS_800000.pth.tar ./models_repository/fast-speech-2/1/data/
```

Unpack archives from ./models_repository/hifigan_vocoder
```bash
tar -xf ./models_repository/hifigan_vocoder/1/data/generator_universal.pth.tar.zip
```

Run compose to build, launch and get results of inference. You can find outputs in "./output/" folder.
```
docker compose up -d
```
