from typing import Final
import os
import io
import base64
import torch
from PIL import Image

from llama_cpp import Llama
from llama_cpp.llama_chat_format import Llava15ChatHandler


class SpaceLLaVA13BQ4Inference:
    """
    Requires 11GB for cuda GPU
    """

    model_name: Final[str] = "remyxai/SpaceLLaVA"
    model: Llama

    mmproj_path = "./mmproj-model-f16.gguf"
    model_path = "./ggml-model-q4_0.gguf"

    def __init__(self, device: torch.device | str):
        if isinstance(device, torch.device):
            device = str(device)
        if device.startswith("cuda"):
            device = device.split(":")[1]
        os.environ["CUDA_VISIBLE_DEVICES"] = device

        self._prepare_model()

        self.chat_handler = Llava15ChatHandler(
            clip_model_path=self.mmproj_path, verbose=True
        )

        self.model = Llama(
            model_path=self.model_path,
            chat_handler=self.chat_handler,
            n_ctx=2048,
            logits_all=True,
            n_gpu_layers=-1,
        )

    def _prepare_model(self):
        if not os.path.exists(self.mmproj_path):
            raise FileNotFoundError(f"Model file {self.mmproj_path} not found")
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"Model file {self.model_path} not found")

    def __call__(self, query: str, image: Image.Image):
        return self.query_for_image(query, image)

    def query_for_image(self, query: str, image: Image.Image) -> str:
        iamge_base64 = self._image_to_base64_data_uri(image)

        messages = [
            {
                "role": "system",
                "content": "You are an assistant who perfectly describes images.",
            },
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": iamge_base64}},
                    {"type": "text", "text": query},
                ],
            },
        ]
        results = self.model.create_chat_completion(messages=messages)
        return results["choices"][0]["message"]["content"]  # type: ignore

    def _image_to_base64_data_uri(self, image_input: str | Image.Image) -> str:

        # Check if the input is a file path (string)
        if isinstance(image_input, str):
            with open(image_input, "rb") as img_file:
                base64_data = base64.b64encode(img_file.read()).decode("utf-8")

        # Check if the input is a PIL Image
        elif isinstance(image_input, Image.Image):
            buffer = io.BytesIO()
            image_input.save(
                buffer, format="PNG"
            )  # You can change the format if needed
            base64_data = base64.b64encode(buffer.getvalue()).decode("utf-8")

        else:
            raise ValueError(
                "Unsupported input type. Input must be a file path or a PIL.Image.Image instance."
            )

        return f"data:image/png;base64,{base64_data}"
