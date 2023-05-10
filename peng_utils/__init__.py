from .test_blip2 import TestBlip2
from .test_llava import TestLLaVA
from .test_minigpt4 import TestMiniGPT4
from .test_mplug_owl import TestMplugOwl
from .test_multimodel_gpt import TestMultimodelGPT
from .test_otter import TestOtter
from .test_flamingo import TestFlamingo


import torch
def skip(*args, **kwargs):
        pass
torch.nn.init.kaiming_uniform_ = skip
torch.nn.init.uniform_ = skip
torch.nn.init.normal_ = skip
