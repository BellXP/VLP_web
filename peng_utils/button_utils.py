import os
import time
import pickle
import datetime

import gradio as gr
from .conversation import default_conversation

no_change_btn = gr.Button.update()
enable_btn = gr.Button.update(interactive=True)
disable_btn = gr.Button.update(interactive=False)
LOGDIR = ''


def get_conv_log_filename():
    t = datetime.datetime.now()
    name = os.path.join(LOGDIR, f"{t.year}-{t.month:02d}-{t.day:02d}-conv.pkl")
    return name


def vote_last_response(state, vote_type, model_selector, request: gr.Request):
    log_name = get_conv_log_filename()
    new_log_data = {
        "tstamp": round(time.time(), 4),
        "type": vote_type,
        "model": model_selector,
        "state": state.dict(),
        "ip": request.client.host,
    }
    try:
        log_data = pickle.load(open(log_name, 'rb'))
    except Exception as e:
        log_data = []
    log_data.append(new_log_data)
    with open(log_name, 'wb') as f:
        pickle.dump(log_data, f)


def upvote_last_response(state, model_selector, request: gr.Request):
    vote_last_response(state, "upvote", model_selector, request)
    return ("",) + (disable_btn,) * 3


def downvote_last_response(state, model_selector, request: gr.Request):
    vote_last_response(state, "downvote", model_selector, request)
    return ("",) + (disable_btn,) * 3


def tie_last_response(state, model_selector, request: gr.Request):
    vote_last_response(state, "tie", model_selector, request)
    return ("",) + (disable_btn,) * 3


def clear_vqa_history(request: gr.Request):
    state = default_conversation.copy()
    return (state, state.to_gradio_chatbot(), "", None) + (disable_btn,) * 4


def clear_chat_history(request: gr.Request):
    state = default_conversation.copy()
    return (state, state.to_gradio_chatbot(), "") + (enable_btn, disable_btn)