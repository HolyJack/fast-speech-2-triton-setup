import triton_python_backend_utils as pb_utils
import json
import yaml
from FastSpeech2.utils.model import get_vocoder, vocoder_infer
import torch
import numpy as np
import torch


device = torch.device("cpu")


class TritonPythonModel:

    def initialize(self, args):

        # You must parse model_config. JSON string is not parsed here
        self.model_config = model_config = json.loads(args["model_config"])

        # Get OUTPUTS configuration
        wav_predictions = pb_utils.get_output_config_by_name(
            model_config, "wav_predictions"
        )

        # Convert Triton types to numpy types
        self.speakers_dtype = pb_utils.triton_string_to_numpy(
            wav_predictions["data_type"]
        )
        # Load configs
        self.model_config = yaml.load(open("/model.yaml", "r"), Loader=yaml.FullLoader)
        self.preprocess_config = yaml.load(
            open("/preprocess.yaml", "r"), Loader=yaml.FullLoader
        )

        # Instantiate the PyTorch model
        self.vocoder = get_vocoder(self.model_config, device)

    def execute(self, requests):

        responses = []

        # Every Python backend must iterate over everyone of the requests
        # and create a pb_utils.InferenceResponse for each of them.
        for request in requests:
            # Get Inputs
            mel_predictions = torch.from_numpy(
                pb_utils.get_input_tensor_by_name(request, "postnet_output").as_numpy()
            ).transpose(1, 2)
            lengths = (
                torch.from_numpy(
                    pb_utils.get_input_tensor_by_name(request, "mel_lens").as_numpy()
                )
                * self.preprocess_config["preprocessing"]["stft"]["hop_length"]
            )
            # prediction
            wav_predictions = vocoder_infer(
                mel_predictions,
                self.vocoder,
                self.model_config,
                self.preprocess_config,
                lengths=lengths,
            )
            wav_predictions = np.asarray(wav_predictions, dtype=np.int16)
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
