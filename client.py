import os
import time
import tritonclient.http as httpclient
import numpy as np
import sounddevice
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
    client = httpclient.InferenceServerClient(url="localhost:8000")

    for line in text:
        send_request(client, line)


def send_request(client, text):
    np_text_data = np.asarray([text], dtype=object)
    input_text = httpclient.InferInput("text", [1], "BYTES")
    input_text.set_data_from_numpy(np_text_data.reshape([1]))

    start = time.time()
    fast_speech2_ensemble_responce = client.infer(
        model_name="ensemble_model_english", inputs=[input_text]
    )
    end = time.time()
    print("Time taken:", end - start)
    wav_predictions = fast_speech2_ensemble_responce.as_numpy("wav_predictions")
    # from preprocessing config
    sampling_rate = 22050

    for i, wav in enumerate(wav_predictions):
        sounddevice.play(wav, sampling_rate)
        wavfile.write(f"output/{text}_{i}.wav", sampling_rate, wav)


if __name__ == "__main__":
    main()
