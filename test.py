import torch
import yaml
from FastSpeech2.model import FastSpeech2
from FastSpeech2.synthesize import preprocess_english
from FastSpeech2.utils.tools import synth_samples, to_device
from FastSpeech2.utils.model import get_vocoder
import numpy as np
import os

device = torch.device("cpu")

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


text = "A nice day to be alive. Honestly, wow!"

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

for field in batch[2:]:
    print(field.shape)

with torch.no_grad():
    output = model(*(batch[2:]))
    for o in output:
        print(o.shape)

    synth_samples(
        batch,
        output,
        vocoder,
        model_config,
        preprocess_config,
        os.path.join(ROOT_DIR, "output/"),
    )
# torch.jit.trace(model, (speakers, texts, text_lens, max_text_lens))
