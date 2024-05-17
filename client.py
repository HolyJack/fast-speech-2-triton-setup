import tritonclient.http as httpclient
import numpy as np
import torch
from FastSpeech2.utils.model import get_vocoder
from FastSpeech2.utils.tools import synth_samples
import yaml
import os
import time
from tritonclient.utils import *

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    text = "Chipi chipi chapa chapa"

    client = httpclient.InferenceServerClient(url="localhost:8000")

    np_text_data = np.asarray([text], dtype=object)
    input_text = httpclient.InferInput("text", [1], "BYTES")
    input_text.set_data_from_numpy(np_text_data.reshape([1]))

    fast_speech2_ensemble_responce = client.infer(
        model_name="ensemble_model_english", inputs=[input_text]
    )
    output_names = [
        "output",
        "postnet_output",
        "p_predictions",
        "e_predictions",
        "log_d_predictions",
        "d_rounded",
        "src_masks",
        "mel_masks",
        "src_lens",
        "mel_lens",
    ]

    outputs = []
    for name in output_names:
        outputs.append(torch.from_numpy(fast_speech2_ensemble_responce.as_numpy(name)))
    output = tuple(outputs)

    device = torch.device("cpu")
    model_config = yaml.load(open("model.yaml", "r"), Loader=yaml.FullLoader)
    preprocess_config = yaml.load(open("preprocess.yaml", "r"), Loader=yaml.FullLoader)
    vocoder = get_vocoder(model_config, device)
    synth_samples(
        [text],
        output,
        vocoder,
        model_config,
        preprocess_config,
        os.path.join(ROOT_DIR, "output/"),
    )


if __name__ == "__main__":
    start = time.time()
    main()
    end = time.time()

    print("Time taken:", end - start)
