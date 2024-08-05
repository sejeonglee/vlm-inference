from typing import Final
from transformers import (
    LlavaProcessor,
    LlavaForConditionalGeneration,
    ProcessorMixin,
    PreTrainedModel,
    BitsAndBytesConfig,
)
import torch
from PIL import Image


class LLaVA13BQ4Inference:
    """
    Requires 11GB for cuda GPU
    """

    device: torch.device
    model_name: Final[str] = "llava-hf/llava-1.5-13b-hf"
    processor: LlavaProcessor
    model: PreTrainedModel

    def __init__(self, device: torch.device | str):
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device

        processor = LlavaProcessor.from_pretrained(self.model_name)
        self.processor = (
            processor if isinstance(processor, ProcessorMixin) else processor[0]
        )

        self.model = LlavaForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            device_map=self.device,
            low_cpu_mem_usage=True,
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
            ),
        )

    def __call__(self, query: str, image: Image.Image):
        return self.query_for_image(query, image)

    def query_for_image(self, query: str, image: Image.Image) -> str:
        conversation = [
            {
                "role": "user",
                "content": [{"type": "image"}, {"type": "text", "text": query}],
            }
        ]

        prompt = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )
        inputs = self.processor(prompt, image, return_tensors="pt").to(self.device)

        output = self.model.generate(**inputs, max_new_tokens=100)

        return self.processor.decode(output[0], skip_special_tokens=True)
