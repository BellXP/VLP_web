from transformers import CLIPImageProcessor
from .otter.modeling_otter import OtterForConditionalGeneration

CKPT_PATH='/data1/VLP_web_data/otter-9b-hf'


class TestOtter:
    def __init__(self, model_path=CKPT_PATH) -> None:
        self.model = OtterForConditionalGeneration.from_pretrained(model_path)
        self.tokenizer = self.model.text_tokenizer
        self.image_processor = CLIPImageProcessor()
        self.tokenizer.padding_side = "left"

    def generate(self, question, raw_image, device=None):
        try:
            if device is not None and 'cuda' in device.type:
                self.model = self.model.to(device)

            vision_x = (self.image_processor.preprocess([raw_image], return_tensors="pt")["pixel_values"].unsqueeze(1).unsqueeze(0))
            lang_x = self.model.text_tokenizer([f"<image> User: {question} GPT: <answer>"], return_tensors="pt")
            generated_text = self.model.generate(
                vision_x=vision_x.to(self.model.device),
                lang_x=lang_x["input_ids"].to(self.model.device),
                attention_mask=lang_x["attention_mask"].to(self.model.device),
                max_new_tokens=256,
                num_beams=1,
                no_repeat_ngram_size=3,
            )
            output = self.model.text_tokenizer.decode(generated_text[0])
            self.model = self.model.to('cpu')
            output = [x for x in output.split(' ') if not x.startswith('<')]
            out_label = output.index('GPT:')
            output = ' '.join(output[out_label + 1:])
            
            return output
        except Exception as e:
            return getattr(e, 'message', str(e))
    