import torch
from .mplug_owl.interface import do_generate, get_model

TOKENIZER_PATH = '/data1/VLP_web_data/mPLUG-Owl-model/tokenizer.model'
CHECKPOINT_PATH = '/data1/VLP_web_data/mPLUG-Owl-model/pretrained.pth'


class TestMplugOwl:
    def __init__(self, checkpoint_path=CHECKPOINT_PATH, tokenizer_path=TOKENIZER_PATH):
        model, tokenizer, img_processor = get_model(
                checkpoint_path=checkpoint_path, tokenizer_path=tokenizer_path, device='cpu', dtype=torch.float32)
        self.model = model
        self.tokenizer = tokenizer
        self.img_processor = img_processor

    def generate(self, text_input, image, device=None, generate_config=None):
        try:
            prompts = [f'''
                The following is a conversation between a curious human and AI assistant. The assistant gives helpful, detailed, and polite answers to the user's questions.
                Human: <image>
                Human: {text_input}
            ''']

            generate_config = generate_config if generate_config is not None else {
                "top_k": 5, 
                "max_length": 512,
                "do_sample":True
            }

            if device is not None and 'cuda' in device.type:
                self.model = self.model.to(device)
            
            generated_text = do_generate(prompts, [image], self.model, self.tokenizer, self.img_processor, **generate_config, device=device, dtype=torch.float32)
            self.model = self.model.to('cpu').float()

            return generated_text
        except Exception as e:
            return getattr(e, 'message', str(e))