CONTROLLER_HEART_BEAT_EXPIRATION = 90
WORKER_HEART_BEAT_INTERVAL = 30
WORKER_API_TIMEOUT = 2000


LOGDIR = "server_utils/runtime_log"

from peng_utils import DATA_DIR
CONVERSATION_SAVE_DIR = f'{DATA_DIR}/conversation_data'


rules_markdown = """ ### Rules
- Vote for large multi-modality models on VQA.
- Load an image and ask a question or click following examples. 
- Only one question is supported per round.
- Two models are anonymous before your vote.
- Click “Clear history” to start a new round.
- [[GitHub]](https://github.com/OpenGVLab/Multi-modality-Arena)
"""


notice_markdown = """ ### Terms of use
To use this service, users must agree to the following conditions: This service is an experimental research tool for non-commercial purposes only. It has limited safeguards and may generate inappropriate content. It cannot be used for anything illegal, harmful, violent, racist or sexual. The service collects data on user conversations and may distribute this data under a Creative Commons Attribution (CC-BY) license.
"""


license_markdown = """ ### Acknowledgement
The service is built upon [Fastchat](https://chat.lmsys.org/).
### License
The service is intended for research purpose and non-commercial use only. It also subjects to subject to the model license of models we used, Terms of Use of the data generated by OpenAI, and Privacy Practices of ShareGPT. If you suspect any possible violations, please do not hesitate to contact us.
"""