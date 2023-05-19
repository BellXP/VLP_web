import os
import gradio as gr
import torch
from PIL import Image
from mmgpt.models.builder import create_model_and_transforms

TEMPLATE = "Below is an instruction that describes a task. Write a response that appropriately completes the request."
response_split = "### Response:"

from peng_utils import DATA_DIR
DATA_DIR = '/nvme/data1/VLP_web_data'
llama_path = f'{DATA_DIR}/Multimodel-GPT/llama-7b-hf'
open_flamingo_path = f'{DATA_DIR}/Multimodel-GPT/checkpoint.pt'
finetune_path = f'{DATA_DIR}/Multimodel-GPT/mmgpt-lora-v0-release.pt'


def get_prompt(message):
    prompt_template=TEMPLATE
    ai_prefix="Response"
    user_prefix="Instruction"

    format_dict = dict()
    format_dict["user_prefix"] = user_prefix
    format_dict["ai_prefix"] = ai_prefix
    prompt_template = prompt_template.format(**format_dict)
    ret = prompt_template

    sep="\n\n### "
    role = 'Instruction'
    context = [sep + "Image:\n<image>" + sep + role + ":\n" + message]

    ret += "".join(context[::-1])
    return ret


class TestMultiModelGPT:
    def __init__(self):
        ckpt = torch.load(finetune_path, map_location="cpu")
        if "model_state_dict" in ckpt:
            state_dict = ckpt["model_state_dict"]
            # remove the "module." prefix
            state_dict = {
                k[7:]: v
                for k, v in state_dict.items() if k.startswith("module.")
            }
        else:
            state_dict = ckpt
        tuning_config = ckpt.get("tuning_config")
        if tuning_config is None:
            print("tuning_config not found in checkpoint")
        else:
            print("tuning_config found in checkpoint: ", tuning_config)
        model, image_processor, tokenizer = create_model_and_transforms(
            model_name="open_flamingo",
            clip_vision_encoder_path="ViT-L-14",
            clip_vision_encoder_pretrained="openai",
            lang_encoder_path=llama_path,
            tokenizer_path=llama_path,
            pretrained_model_path=open_flamingo_path,
            tuning_config=tuning_config,
        )
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        tokenizer.padding_side = "left"
        tokenizer.add_eos_token = False
        self.model = model
        self.image_processor = image_processor
        self.tokenizer = tokenizer

    def move_to_device(self, device=None):
        if device is not None and 'cuda' in device.type:
            dtype = torch.float16
            self.model = self.model.half()
        else:
            dtype = torch.float32
            device = torch.device('cpu')
            self.model = self.model.float()
        self.model = self.model.to(device)
        
        return device, dtype

    def do_generate(self, prompt, images, device, max_new_token=512, num_beams=3, temperature=1.0,
                 top_k=20, top_p=1.0, do_sample=True):
        lang_x = self.tokenizer([prompt], return_tensors="pt")
        vision_x = [self.image_processor(im).unsqueeze(0) for im in images]
        vision_x = torch.cat(vision_x, dim=0)
        vision_x = vision_x.unsqueeze(1).unsqueeze(0)
        if 'cuda' in device.type:
            vision_x = vision_x.half()

        output_ids = self.model.generate(
            vision_x=vision_x.to(device),
            lang_x=lang_x["input_ids"].to(device),
            attention_mask=lang_x["attention_mask"].to(device),
            max_new_tokens=max_new_token,
            num_beams=num_beams,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            do_sample=do_sample,
        )[0]

        generated_text = self.tokenizer.decode(
            output_ids, skip_special_tokens=True)
        # print(generated_text)
        result = generated_text.split(response_split)[-1].strip()
        return result

    def generate(self, question, raw_image, device=None, keep_in_device=False):
        # try:
        device, dtype = self.move_to_device(device)
        output = self.do_generate(get_prompt(question), [raw_image], device)

        if not keep_in_device:
            self.move_to_device()
        
        return output
        # except Exception as e:
        #     return getattr(e, 'message', str(e))