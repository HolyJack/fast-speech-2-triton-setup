import torch
import yaml
from FastSpeech2.model import FastSpeech2
from FastSpeech2.synthesize import preprocess_english
from FastSpeech2.utils.tools import to_device, synth_samples_v2
from FastSpeech2.utils.model import get_vocoder
import numpy as np
import os
import sounddevice
from scipy.io import wavfile
import time

device = torch.device("cpu")

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


text = "A nice day to be alive. Honestly, wow! What a great day!"

preprocess_config = yaml.load(open("preprocess.yaml", "r"), Loader=yaml.FullLoader)
model_config = yaml.load(open("model.yaml", "r"), Loader=yaml.FullLoader)
model = FastSpeech2(preprocess_config=preprocess_config, model_config=model_config)

ckpt_path = os.path.join(ROOT_DIR, "800000.pth.tar")
ckpt = torch.load(ckpt_path, map_location=device)
model.load_state_dict(ckpt["model"])
model.eval()
model.requires_grad_ = False

vocoder = get_vocoder(model_config, device)

ids = raw_texts = text[:100]
speakers = np.array([0])
texts = np.array([preprocess_english(text, preprocess_config)])
text_lens = np.array([len(texts[0])])

batch = (ids, raw_texts, speakers, texts, text_lens, max(text_lens))
batch = to_device(batch, device)

# for field in batch[2:]:
#    print(field.shape)

with torch.no_grad():
    output = model(*(batch[2:]))
    # for o in output:
    #    print(o.shape)
    print(output[9].shape)
    wavs, sample_rate = synth_samples_v2(
        batch,
        output,
        vocoder,
        model_config,
        preprocess_config,
    )


print(len(wavs), wavs[0].shape)
wavs_numpy = np.asarray(wavs, dtype=np.int16)
print(wavs_numpy.shape)
for i, wav in enumerate(wavs):
    sounddevice.play(wav, sample_rate)
    wavfile.write(f"output/{i}.wav", sample_rate, wav)
    time.sleep(10)
