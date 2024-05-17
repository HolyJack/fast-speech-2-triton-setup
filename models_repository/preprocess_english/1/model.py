import triton_python_backend_utils as pb_utils
import numpy as np
import json
import yaml
import torch
from FastSpeech2.synthesize import preprocess_english


class TritonPythonModel:

    def initialize(self, args):

        # You must parse model_config. JSON string is not parsed here
        self.model_config = model_config = json.loads(args["model_config"])

        # Get OUTPUTS configuration
        speakers_config = pb_utils.get_output_config_by_name(model_config, "speakers")
        texts_config = pb_utils.get_output_config_by_name(model_config, "texts")
        text_lens_config = pb_utils.get_output_config_by_name(model_config, "text_lens")
        max_text_lens_config = pb_utils.get_output_config_by_name(
            model_config, "max_text_lens"
        )

        # Convert Triton types to numpy types
        self.speakers_dtype = pb_utils.triton_string_to_numpy(
            speakers_config["data_type"]
        )
        self.texts_dtype = pb_utils.triton_string_to_numpy(texts_config["data_type"])
        self.text_lens_dtype = pb_utils.triton_string_to_numpy(
            text_lens_config["data_type"]
        )
        self.max_text_lens_dtype = pb_utils.triton_string_to_numpy(
            max_text_lens_config["data_type"]
        )
        # Load configs
        self.preprocess_config = yaml.load(
            open("/preprocess.yaml", "r"), Loader=yaml.FullLoader
        )

        self.speakers = np.array([0])
        # Instantiate the PyTorch model
        # --------- no model ----------

    def execute(self, requests):

        responses = []

        # Every Python backend must iterate over everyone of the requests
        # and create a pb_utils.InferenceResponse for each of them.
        for request in requests:
            # Get Inputs
            text_data = pb_utils.get_input_tensor_by_name(request, "text")
            text = text_data.as_numpy()[0][0].decode("utf-8")
            print(text, flush=True)

            # Preprocessing english
            speakers = self.speakers
            texts = np.array([preprocess_english(text, self.preprocess_config)])
            text_lens = np.array([len(texts[0])])
            max_text_lens = np.array([max(text_lens)])
            # Create output tensors. You need pb_utils.Tensor
            # objects to create pb_utils.InferenceResponse.
            speakers_tensor = pb_utils.Tensor("speakers", speakers)
            texts_tensor = pb_utils.Tensor("texts", texts)
            print("All good x -1", flush=True)
            text_lens_tensor = pb_utils.Tensor("text_lens", text_lens)
            print("All good", flush=True)
            max_text_lens_tensor = pb_utils.Tensor("max_text_lens", max_text_lens)
            print("All good x 2", flush=True)
            # Create InferenceResponse. You can set an error here in case
            # there was a problem with handling this inference request.
            # Below is an example of how you can set errors in inference
            # response:
            #
            # pb_utils.InferenceResponse(
            #    output_tensors=..., TritonError("An error occurred"))
            inference_response = pb_utils.InferenceResponse(
                output_tensors=[
                    speakers_tensor,
                    texts_tensor,
                    text_lens_tensor,
                    max_text_lens_tensor,
                ]
            )
            print("All good x 3", flush=True)
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
