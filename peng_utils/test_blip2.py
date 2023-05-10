import torch
from lavis.models import load_model_and_preprocess


class TestBlip2:
    def __init__(self) -> None:
        self.model, self.vis_processors, _ = load_model_and_preprocess(
            name="blip2_t5", model_type="pretrain_flant5xl", is_eval=True, device='cpu'
        )

    def generate(self, question, raw_image, device=None):
        try:
            if device is not None and 'cuda' in device.type:
                self.model = self.model.to(device)
            else:
                device = 'cpu'
            
            image = self.vis_processors["eval"](raw_image).unsqueeze(0).to(self.model.device)
            answer = self.model.generate({
                "image": image, "prompt": f"Question: {question} Answer:"
            })

            self.model = self.model.to('cpu')
            self.model.Qformer = self.model.Qformer.to('cpu')
            
            return answer[0]
        except Exception as e:
            return getattr(e, 'message', str(e))
    