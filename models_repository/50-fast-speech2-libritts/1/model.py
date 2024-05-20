import triton_python_backend_utils as pb_utils
import numpy as np
import json
import yaml
import torch
import os
from .data.model.fastspeech2 import FastSpeech2


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TritonPythonModel:

    def initialize(self, args):

        # You must parse model_config. JSON string is not parsed here
        self.model_config = model_config = json.loads(args["model_config"])

        repository = args["model_repository"]
        version = args["model_version"]
        path = os.path.join(repository, version, "data")
        # Get OUTPUTS configuration

        output_config = pb_utils.get_output_config_by_name(model_config, "output")
        postnet_output_config = pb_utils.get_output_config_by_name(
            model_config, "postnet_output"
        )
        p_predictions_config = pb_utils.get_output_config_by_name(
            model_config, "p_predictions"
        )
        e_predictions_config = pb_utils.get_output_config_by_name(
            model_config, "e_predictions"
        )
        log_d_predictions_config = pb_utils.get_output_config_by_name(
            model_config, "log_d_predictions"
        )
        d_rounded_config = pb_utils.get_output_config_by_name(model_config, "d_rounded")
        src_masks_config = pb_utils.get_output_config_by_name(model_config, "src_masks")
        mel_masks_config = pb_utils.get_output_config_by_name(model_config, "mel_masks")
        src_lens_config = pb_utils.get_output_config_by_name(model_config, "src_lens")
        mel_lens_config = pb_utils.get_output_config_by_name(model_config, "mel_lens")

        # Convert Triton types to numpy types
        self.output_dtype = pb_utils.triton_string_to_numpy(output_config["data_type"])
        self.postnet_output_dtype = pb_utils.triton_string_to_numpy(
            postnet_output_config["data_type"]
        )
        self.postnet_output_dtype = pb_utils.triton_string_to_numpy(
            p_predictions_config["data_type"]
        )

        self.e_predictions_dtype = pb_utils.triton_string_to_numpy(
            e_predictions_config["data_type"]
        )
        self.log_d_predictions_dtype = pb_utils.triton_string_to_numpy(
            log_d_predictions_config["data_type"]
        )
        self.d_rounded_dtype = pb_utils.triton_string_to_numpy(
            d_rounded_config["data_type"]
        )
        self.src_masks_dtype = pb_utils.triton_string_to_numpy(
            src_masks_config["data_type"]
        )
        self.mel_masks_dtype = pb_utils.triton_string_to_numpy(
            mel_masks_config["data_type"]
        )
        self.src_lens_dtype = pb_utils.triton_string_to_numpy(
            src_lens_config["data_type"]
        )
        self.mel_lens_dtype = pb_utils.triton_string_to_numpy(
            mel_lens_config["data_type"]
        )

        # Load configs
        preprocess_config = yaml.load(
            open(os.path.join(path, "preprocess.yaml"), "r"), Loader=yaml.FullLoader
        )
        model_config = yaml.load(
            open(os.path.join(path, "model.yaml"), "r"), Loader=yaml.FullLoader
        )

        # Instantiate the PyTorch model
        self.model = FastSpeech2(
            preprocess_config=preprocess_config, model_config=model_config
        )
        ckpt = torch.load(
            os.path.join(path, "LibriTTS_800000.pth.tar"), map_location=device
        )
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        self.model.requires_grad_ = False

    def execute(self, requests):

        responses = []

        # Every Python backend must iterate over everyone of the requests
        # and create a pb_utils.InferenceResponse for each of them.
        for request in requests:
            # Get Inputs
            speakers = torch.from_numpy(
                pb_utils.get_input_tensor_by_name(request, "speakers").as_numpy()
            )
            texts = torch.from_numpy(
                pb_utils.get_input_tensor_by_name(request, "texts").as_numpy()
            ).long()
            text_lens = torch.from_numpy(
                pb_utils.get_input_tensor_by_name(request, "text_lens").as_numpy()
            ).long()
            max_text_lens = pb_utils.get_input_tensor_by_name(
                request, "max_text_lens"
            ).as_numpy()[0]

            with torch.no_grad():
                (
                    output,
                    postnet_output,
                    p_predictions,
                    e_predictions,
                    log_d_predictions,
                    d_rounded,
                    src_masks,
                    mel_masks,
                    src_lens,
                    mel_lens,
                ) = self.model(speakers, texts, text_lens, max_text_lens)
            # Create output tensors. You need pb_utils.Tensor
            # objects to create pb_utils.InferenceResponse.
            output_tensor = pb_utils.Tensor("output", np.array(output))
            postnet_output_tensor = pb_utils.Tensor(
                "postnet_output", np.array(postnet_output)
            )
            p_predictions_tensor = pb_utils.Tensor(
                "p_predictions", np.array(p_predictions)
            )
            e_predictions_tensor = pb_utils.Tensor(
                "e_predictions", np.array(e_predictions)
            )
            log_d_predictions_tensor = pb_utils.Tensor(
                "log_d_predictions", np.array(log_d_predictions)
            )
            d_rounded_tensor = pb_utils.Tensor("d_rounded", np.array(d_rounded))
            src_masks_tensor = pb_utils.Tensor("src_masks", np.array(src_masks))
            mel_masks_tensor = pb_utils.Tensor("mel_masks", np.array(mel_masks))
            src_lens_tensor = pb_utils.Tensor("src_lens", np.array(src_lens))
            mel_lens_tensor = pb_utils.Tensor("mel_lens", np.array(mel_lens))
            # Create InferenceResponse. You can set an error here in case
            # there was a problem with handling this inference request.
            # Below is an example of how you can set errors in inference
            # response:
            #
            # pb_utils.InferenceResponse(
            #    output_tensors=..., TritonError("An error occurred"))
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[
                    output_tensor,
                    postnet_output_tensor,
                    p_predictions_tensor,
                    e_predictions_tensor,
                    log_d_predictions_tensor,
                    d_rounded_tensor,
                    src_masks_tensor,
                    mel_masks_tensor,
                    src_lens_tensor,
                    mel_lens_tensor,
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
