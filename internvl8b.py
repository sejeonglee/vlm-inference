from typing import Final
from transformers import (
    AutoTokenizer,
    AutoModel,
    Intern
    
    ProcessorMixin,
    PreTrainedModel,
)
import torch
from PIL import Image


class LLaVANext8BInference:
    """
    Requires 19.5GB for cuda GPU
    """

    device: torch.device
    model_name: Final[str]7 = "llava-hf/llama3-llava-next-8b-hf"
    processor: LlavaNextProcessor
    model: PreTrainedModel

    def __init__(self, device: torch.device | str):
        if isinstance(device, str):
            device = torch.device(device)
        self.device = device

        processor = LlavaNextProcessor.from_pretrained(self.model_name)
        self.processor = (
            processor if isinstance(processor, ProcessorMixin) else processor[0]
        )

        self.model = LlavaNextForConditionalGeneration.from_pretrained(
            self.model_name,
            torch_dtype=torch.float16,
            low_cpu_mem_usage=
        ).to(
            self.device  # type: ignore
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
