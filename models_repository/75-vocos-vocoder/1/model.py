import triton_python_backend_utils as pb_utils
import json
import torch
import numpy as np
from vocos import Vocos
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TritonPythonModel:

    def initialize(self, args):

        # You must parse model_config. JSON string is not parsed here
        self.model_config = model_config = json.loads(args["model_config"])

        # Get OUTPUTS configuration
        wav_predictions = pb_utils.get_output_config_by_name(
            model_config, "wav_predictions"
        )

        # Convert Triton types to numpy types
        self.wav_predictions_dtype = pb_utils.triton_string_to_numpy(
            wav_predictions["data_type"]
        )

        # Instantiate the PyTorch model
        self.vocoder = Vocos.from_pretrained("charactr/vocos-mel-24khz")

    def execute(self, requests):

        responses = []

        # Every Python backend must iterate over everyone of the requests
        # and create a pb_utils.InferenceResponse for each of them.
        for request in requests:
            # Get Inputs
            mel_spectrogram = torch.from_numpy(
                pb_utils.get_input_tensor_by_name(request, "postnet_output").as_numpy()
            ).transpose(1, 2)

            # padding from 80 channels to 100 since vocos accepts 100 channels
            mel_spectrogram = F.pad(mel_spectrogram, (0, 0, 0, 20), "constant", 0)
            # trying to inrepolate
            # mel_spectrogram = F.interpolate(
            #    mel_spectrogram,
            #    size=(100),
            #    mode="linear",
            #    align_corners=True,
            # )

            lengths = (
                torch.from_numpy(
                    pb_utils.get_input_tensor_by_name(request, "mel_lens").as_numpy()
                )
                * 256
            )

            # Prediction
            with torch.no_grad():
                wavs = self.vocoder.decode(mel_spectrogram)

            wavs = (wavs.cpu().numpy() * 32768.0).astype("int16")

            wavs = [wav for wav in wavs]

            for i in range(len(mel_spectrogram)):
                if lengths is not None:
                    wavs[i] = wavs[i][: lengths[i]]

            # back to numpy
            wav_predictions = np.asarray(wavs, dtype=np.int16)
            # Create output tensors. You need pb_utils.Tensor
            # objects to create pb_utils.InferenceResponse.
            wav_predictions_tensor = pb_utils.Tensor("wav_predictions", wav_predictions)
            # Create InferenceResponse. You can set an error here in case
            # there was a problem with handling this inference request.
            # Below is an example of how you can set errors in inference
            # response:
            #
            # pb_utils.InferenceResponse(
            #    output_tensors=..., TritonError("An error occurred"))
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[
                    wav_predictions_tensor,
                ]
            )
            responses.append(inference_response)

        # You should return a list of pb_utils.InferenceResponse. Length
        # of this list must match the length of `requests` list.
        return responses

    def finalize(self):
        """`finalize` is called only once when the model is being unloaded.
        Implementing `finalize` function is optional. This function allows
        the model to perform any necessary clean ups before exit.
        """
        print("Cleaning up...")
