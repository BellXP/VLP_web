"""
The gradio demo server for chatting with a single model.
"""
import os
import time
import json
import uuid
import random
import datetime
import argparse
import requests
import multiprocessing

import gradio as gr
import numpy as np
from PIL import Image

from server_utils.constants import LOGDIR, WORKER_API_TIMEOUT, CONVERSATION_SAVE_DIR
from server_utils.utils import build_logger, server_error_msg

os.makedirs(CONVERSATION_SAVE_DIR, exist_ok=True)
logger = build_logger("web_server", f"{LOGDIR}/web_server.log")
headers = {"User-Agent": "fastchat Client"}
model_list = []
controller_url = None
no_change_btn = gr.Button.update()
enable_btn = gr.Button.update(interactive=True)
disable_btn = gr.Button.update(interactive=False)


def get_model_list(controller_url):
    ret = requests.post(controller_url + "/refresh_all_workers")
    assert ret.status_code == 200
    ret = requests.post(controller_url + "/list_models")
    return list(set(ret.json()["models"]))


def save_vote_data(state, request: gr.Request):
    t = datetime.datetime.now()
    # save image
    img_name = os.path.join(CONVERSATION_SAVE_DIR, 'images', f"{t.year}-{t.month:02d}-{t.day:02d}-{str(uuid.uuid4())}.png")
    while os.path.exists(image):
        img_name = os.path.join(CONVERSATION_SAVE_DIR, 'images', f"{t.year}-{t.month:02d}-{t.day:02d}-{str(uuid.uuid4())}.png")
    image = np.array(state.pop('image'), dtype='uint8')
    image = Image.fromarray(image.astype('uint8')).convert('RGB')
    image.save(img_name)
    # save conversation
    state['image'] = img_name
    log_name = os.path.join(CONVERSATION_SAVE_DIR, f"{t.year}-{t.month:02d}-{t.day:02d}-conversation.json")
    with open(log_name, 'a') as fout:
        state['ip'] = request.client.host
        fout.write(json.dumps(state) + "\n")


def vote_up_model(state, chatbot, request: gr.Request):
    state['user_vote'] = 'up'
    save_vote_data(state, request)
    chatbot.append((
        'Your Vote: Up!',
        f"Up Model: {state['VLP_names'][0]}, Down Model: {state['VLP_names'][1]}"
    ))
    return chatbot, disable_btn, disable_btn, disable_btn, enable_btn


def vote_down_model(state, chatbot, request: gr.Request):
    state['user_vote'] = 'down'
    save_vote_data(state, request)
    chatbot.append((
        'Your Vote: Down!',
        f"Up Model: {state['VLP_names'][0]}, Down Model: {state['VLP_names'][1]}"
    ))
    return chatbot, disable_btn, disable_btn, disable_btn, enable_btn


def vote_model_tie(state, chatbot, request: gr.Request):
    state['user_vote'] = 'tie'
    save_vote_data(state, request)
    chatbot.append((
        'Your Vote: Tie!',
        f"Up Model: {state['VLP_names'][0]}, Down Model: {state['VLP_names'][1]}"
    ))
    return chatbot, disable_btn, disable_btn, disable_btn, enable_btn


def clear_chat(state):
    if state is not None:
        state = {}
    return state, None, gr.update(value=None, interactive=True), gr.update(placeholder="Enter text and press ENTER"), disable_btn, disable_btn, disable_btn, enable_btn


def user_ask(state, chatbot, text_box):
    state['text'] = text_box
    if text_box == '':
        return state, chatbot, '', enable_btn
    chatbot = chatbot + [[text_box, None], [text_box, None]] 
    return state, chatbot, '', disable_btn


def model_worker_stream_iter(worker_addr, state):
    response = requests.post(
        worker_addr + "/worker_generate_stream",
        headers=headers,
        json={"text": state['text'], "image": state['image']},
        stream=True,
        timeout=WORKER_API_TIMEOUT,
    )
    for chunk in response.iter_lines(decode_unicode=False, delimiter=b"\0"):
        if chunk:
            data = json.loads(chunk.decode())
            yield data


def get_model_worker_output(worker_input):
    (worker_addr, state) = worker_input
    stream_iter = model_worker_stream_iter(worker_addr, state)
    try:
        for data in stream_iter:
            if data["error_code"] == 0:
                output = data["text"].strip()
                return output
            elif data["error_code"] == 1:
                output = data["text"] + f" (error_code: {data['error_code']})"
                return output
            time.sleep(5)
    except requests.exceptions.RequestException as e:
        output = server_error_msg + f" (error_code: 4)"
        return output
    except Exception as e:
        output = server_error_msg + f" (error_code: 5, {e})"
        return output


def run_VLP_models(state, chatbot, gr_img):
    def get_model_worker_addr(model_name):
        ret = requests.post(
            controller_url + "/get_worker_address", json={"model": model_name}
        )
        return ret.json()["address"]

    if state['text'] == '' or gr_img is None:
        return state, chatbot, enable_btn, disable_btn, disable_btn, disable_btn, enable_btn
    state['image'] = np.array(gr_img, dtype='uint8').tolist()
    selected_VLP_models = random.sample(model_list, 2)
    model_worker_addrs = [get_model_worker_addr(model_name) for model_name in selected_VLP_models]
    pool = multiprocessing.Pool()
    vlp_outputs = pool.map(get_model_worker_output, [(worker_addr, state) for worker_addr in model_worker_addrs])
    state['VLP_names'] = selected_VLP_models
    state['VLP_outputs'] = vlp_outputs
    chatbot[-2][1] = vlp_outputs[0]
    chatbot[-1][1] = vlp_outputs[1]
    return state, chatbot, enable_btn, enable_btn, enable_btn, enable_btn, enable_btn


def build_demo():
    with gr.Blocks() as demo:
        state = gr.State({})

        with gr.Row():
            with gr.Column(scale=0.5):
                imagebox = gr.Image(type="pil")
                with gr.Row() as button_row:
                    upvote_btn = gr.Button(value="👍  Upvote", interactive=False)
                    downvote_btn = gr.Button(value="👎  Downvote", interactive=False)
                    tie_btn = gr.Button(value="🤝  Tie", interactive=False)
                    clear_btn = gr.Button(value="🗑️  Clear", interactive=False)
            with gr.Column():
                with gr.Row():
                    chatbot = gr.Chatbot(label='ChatBox')
                    # chatbot2 = gr.Chatbot(label='ChatBox')
                with gr.Row():
                    with gr.Column(scale=8):
                        textbox = gr.Textbox(placeholder="Enter text and press ENTER")
                    with gr.Column(scale=1, min_width=60):
                        submit_btn = gr.Button(value="Submit")
        
        btn_list = [upvote_btn, downvote_btn, tie_btn, clear_btn]
        textbox.submit(user_ask, [state, chatbot, textbox], [state, chatbot, textbox, submit_btn]).then(run_VLP_models, [state, chatbot, imagebox], [state, chatbot, submit_btn] + btn_list)
        submit_btn.click(user_ask, [state, chatbot, textbox], [state, chatbot, textbox, submit_btn]).then(run_VLP_models, [state, chatbot, imagebox], [state, chatbot, submit_btn] + btn_list)
        clear_btn.click(clear_chat, [state], [state, chatbot, imagebox, textbox] + btn_list)
        upvote_btn.click(vote_up_model, [state, chatbot], [chatbot] + btn_list)
        downvote_btn.click(vote_down_model, [state, chatbot], [chatbot] + btn_list)
        tie_btn.click(vote_model_tie, [state, chatbot], [chatbot] + btn_list)
    
    return demo


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, default="0.0.0.0")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--controller-url", type=str, default="http://localhost:21001")
    parser.add_argument("--concurrency-count", type=int, default=10)
    parser.add_argument("--share", action="store_true")
    args = parser.parse_args()
    logger.info(f"args: {args}")

    controller_url = args.controller_url
    model_list = get_model_list(controller_url)
    print(f"Available model: {', '.join(model_list)}")
    demo = build_demo()
    demo.queue(
        concurrency_count=args.concurrency_count, status_update_rate=10, api_open=False
    ).launch(
        server_name=args.host, server_port=args.port, share=args.share, max_threads=200
    )