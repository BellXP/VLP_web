import torch
from transformers import AutoTokenizer, AutoConfig
from .llava import conv_templates, LlavaLlamaForCausalLM
from transformers import CLIPImageProcessor, StoppingCriteria

DEFAULT_IMAGE_TOKEN = "<image>"
DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"
DEFAULT_IM_START_TOKEN = "<im_start>"
DEFAULT_IM_END_TOKEN = "<im_end>"


def patch_config(config):
    patch_dict = {
        "use_mm_proj": True,
        "mm_vision_tower": "openai/clip-vit-large-patch14",
        "mm_hidden_size": 1024
    }

    cfg = AutoConfig.from_pretrained(config)
    if not hasattr(cfg, "mm_vision_tower"):
        print(f'`mm_vision_tower` not found in `{config}`, applying patch and save to disk.')
        for k, v in patch_dict.items():
            setattr(cfg, k, v)
        cfg.save_pretrained(config)


class KeywordsStoppingCriteria(StoppingCriteria):
    def __init__(self, keywords, tokenizer, input_ids):
        self.keywords = keywords
        self.tokenizer = tokenizer
        self.start_len = None
        self.input_ids = input_ids

    def __call__(self, output_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        if self.start_len is None:
            self.start_len = self.input_ids.shape[1]
        else:
            outputs = self.tokenizer.batch_decode(output_ids[:, self.start_len:], skip_special_tokens=True)[0]
            for keyword in self.keywords:
                if keyword in outputs:
                    return True
        return False


class TestLLaVA:
    def __init__(self, model_path="liuhaotian/LLaVA-Lightning-MPT-7B-preview"):
        device, dtype = 'cpu', torch.float32

        tokenizer = AutoTokenizer.from_pretrained(model_path)
        patch_config(model_path)
        model = LlavaLlamaForCausalLM.from_pretrained(model_path, torch_dtype=dtype)
        image_processor = CLIPImageProcessor.from_pretrained(model.config.mm_vision_tower, torch_dtype=dtype)

        mm_use_im_start_end = False
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
        if mm_use_im_start_end:
            tokenizer.add_tokens([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True)

        vision_tower = model.model.vision_tower[0]
        vision_tower.to(device=device, dtype=dtype)
        vision_config = vision_tower.config
        vision_config.im_patch_token = tokenizer.convert_tokens_to_ids([DEFAULT_IMAGE_PATCH_TOKEN])[0]
        vision_config.use_im_start_end = mm_use_im_start_end
        if mm_use_im_start_end:
            vision_config.im_start_token, vision_config.im_end_token = tokenizer.convert_tokens_to_ids([DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN])
        image_token_len = (vision_config.image_size // vision_config.patch_size) ** 2

        self.model = model
        self.tokenizer = tokenizer
        self.image_processor = image_processor
        self.mm_use_im_start_end = mm_use_im_start_end
        self.image_token_len = image_token_len # 256
        self.conv_mode = 'mpt_multimodal' # mpt, mpt_text, mpt_multimodal

    def move_to_device(self, device=None):
        if device is not None and 'cuda' in device.type:
            dtype = torch.float16
            self.model = self.model.to(device, dtype=dtype)
            self.model.model.vision_tower[0].to(device, dtype=dtype)
            self.image_processor = CLIPImageProcessor.from_pretrained(self.model.config.mm_vision_tower, torch_dtype=dtype)
        else:
            dtype = torch.float32
            device = 'cpu'
            self.model = self.model.to(device, dtype=dtype)
            self.model.model.vision_tower[0].to(device, dtype=dtype)
            self.image_processor = CLIPImageProcessor.from_pretrained(self.model.config.mm_vision_tower, torch_dtype=dtype)
        
        return device, dtype

    def generate(self, text_input, image=None, device=None, keep_in_device=False):
        try:
            device, dtype = self.move_to_device(device)
            qs = text_input
            if self.mm_use_im_start_end:
                qs = qs + '\n' + DEFAULT_IM_START_TOKEN + DEFAULT_IMAGE_PATCH_TOKEN * self.image_token_len + DEFAULT_IM_END_TOKEN
            else:
                qs = qs + '\n' + DEFAULT_IMAGE_PATCH_TOKEN * self.image_token_len

            conv = conv_templates[self.conv_mode].copy()
            conv.append_message(conv.roles[0], qs)
            prompt = conv.get_prompt()
            inputs = self.tokenizer([prompt])
            input_ids = torch.as_tensor(inputs.input_ids).to(device)

            if image is not None:
                image_tensor = self.image_processor.preprocess(image, return_tensors='pt')['pixel_values'][0]
                images = image_tensor.unsqueeze(0).to(device, dtype=dtype)
            else:
                images = None
            
            keywords = [conv.sep] if conv.sep2 is None else [conv.sep, conv.sep2]
            stopping_criteria = KeywordsStoppingCriteria(keywords, self.tokenizer, input_ids)

            with torch.inference_mode():
                output_ids = self.model.generate(
                    input_ids,
                    images=images,
                    do_sample=True,
                    temperature=0.7,
                    max_new_tokens=1024,
                    stopping_criteria=[stopping_criteria])

            input_token_len = input_ids.shape[1]
            n_diff_input_output = (input_ids != output_ids[:, :input_token_len]).sum().item()
            if n_diff_input_output > 0:
                print(f'[Warning] {n_diff_input_output} output_ids are not the same as the input_ids')
            outputs = self.tokenizer.batch_decode(output_ids[:, input_token_len:], skip_special_tokens=True)[0]

            try:
                index = outputs.index(conv.sep)
            except ValueError:
                outputs += conv.sep
                index = outputs.index(conv.sep)

            outputs = outputs[:index].strip()
            if not keep_in_device:
                self.move_to_device()
            
            return outputs
        except Exception as e:
            return getattr(e, 'message', str(e))
