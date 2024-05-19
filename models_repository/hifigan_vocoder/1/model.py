import triton_python_backend_utils as pb_utils
import json
import yaml
import FastSpeech2.hifigan as hifigan
import torch
import numpy as np
import os


# device = torch.device("cpu")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TritonPythonModel:

    def initialize(self, args):

        # You must parse model_config. JSON string is not parsed here
        self.model_config = model_config = json.loads(args["model_config"])
        repository = args["model_repository"]
        version = args["model_version"]
        path = os.path.join(repository, version, "data")

        # Get OUTPUTS configuration
        wav_predictions = pb_utils.get_output_config_by_name(
            model_config, "wav_predictions"
        )

        # Convert Triton types to numpy types
        self.speakers_dtype = pb_utils.triton_string_to_numpy(
            wav_predictions["data_type"]
        )
        # Load configs
        self.preprocess_config = yaml.load(
            open(os.path.join(path, "preprocess.yaml"), "r"),
            Loader=yaml.FullLoader,
        )
        with open(os.path.join(path, "config.json"), "r") as f:
            config = json.load(f)
        config = hifigan.AttrDict(config)

        # Instantiate the PyTorch model
        # self.vocoder = get_vocoder(self.model_config, device)
        self.vocoder = hifigan.Generator(config)
        ckpt = torch.load(
            os.path.join(path, "generator_universal.pth.tar"),
            map_location=device,
        )
        self.vocoder.load_state_dict(ckpt["generator"])
        self.vocoder.eval()
        self.vocoder.remove_weight_norm()
        self.vocoder.to(device)

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

            # Prediction
            with torch.no_grad():
                wavs = self.vocoder(mel_predictions).squeeze(1)

            wavs = (
                wavs.cpu().numpy()
                * self.preprocess_config["preprocessing"]["audio"]["max_wav_value"]
            ).astype("int16")

            wavs = [wav for wav in wavs]

            for i in range(len(mel_predictions)):
                if lengths is not None:
                    wavs[i] = wavs[i][: lengths[i]]

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
