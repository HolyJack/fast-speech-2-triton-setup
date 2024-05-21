import os
import time
import tritonclient.http as httpclient
import numpy as np
from scipy.io import wavfile

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))


def main():
    text = [
        "Chipi chipi, chapa chapa",
        "Dubidubi, dabadaba",
        "Magico mi dubidubi boom, boom, boom, boom",
        "Chipi chipi, chapa chapa",
        "Dubidubi, dabadaba",
        "Magico mi dubidubi boom",
    ]

    # Creating httpclient
    client = httpclient.InferenceServerClient(url="triton-inference-server:8000")

    for i, line in enumerate(text):
        send_request(client, line, "00-ensemble-fs2-libritts-hifigan", f"hifigan_{i}")
        send_request(client, line, "00-ensemble-fs2-libritts-vocos", f"vocos_{i}")


def send_request(client, text, model_name, tag=None):
    np_text_data = np.asarray([text], dtype=object)
    input_text = httpclient.InferInput("text", [1], "BYTES")
    input_text.set_data_from_numpy(np_text_data.reshape([1]))

    start = time.time()
    fast_speech2_ensemble_responce = client.infer(
        model_name=model_name, inputs=[input_text]
    )
    end = time.time()
    print("Time taken:", end - start)
    wav_predictions = fast_speech2_ensemble_responce.as_numpy("wav_predictions")
    # from preprocessing config
    sampling_rate = 22050

    for i, wav in enumerate(wav_predictions):
        wavfile.write(f"output/{tag}_{i}_{text}.wav", sampling_rate, wav)


if __name__ == "__main__":
    main()
