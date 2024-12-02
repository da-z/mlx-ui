import re
import time

import mlx.core as mx
import streamlit as st
from mlx_lm.utils import load, generate_step
import argparse

title = "MLX Chat"
ver = "0.7.26"
debug = False


def generate(the_prompt, the_model):
    tokens = []
    skip = 0

    for (token, prob), n in zip(generate_step(mx.array(tokenizer.encode(the_prompt)), the_model, temp=temperature),
                                range(context_length)):

        if token == tokenizer.eos_token_id:
            break

        tokens.append(token)
        text = tokenizer.decode(tokens)

        trim = None

        for sw in stop_words:
            if text[-len(sw):].lower() == sw:
                # definitely ends with a stop word. stop generating
                return
            else:
                # if text ends with start of an end word, accumulate tokens and wait for the full word
                for i, _ in enumerate(sw, start=1):
                    if text[-i:].lower() == sw[:i]:
                        trim = -i

        # flush text up till trim point (beginning of stop word)
        yield text[skip:trim]
        skip = len(text)


def show_chat(the_prompt, previous=""):
    if debug:
        print(the_prompt)
        print("-" * 80)

    with ((st.chat_message("assistant"))):
        message_placeholder = st.empty()
        response = previous

        for chunk in generate(the_prompt, model):
            response = response + chunk

            if not previous:
                # begin neural-beagle-14 fixes
                response = re.sub(r"^/\*+/", "", response)
                response = re.sub(r"^:+", "", response)
                # end neural-beagle-14 fixes

            response = response.replace('�', '')
            message_placeholder.markdown(response + "▌")

        message_placeholder.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})


def remove_last_occurrence(array, criteria_fn):
    for i in reversed(range(len(array))):
        if criteria_fn(array[i]):
            del array[i]
            break


def build_memory():
    if len(st.session_state.messages) > 2:
        return st.session_state.messages[1:-1]
    return []


def queue_chat(the_prompt, continuation=""):
    # workaround because the chat boxes are not really replaced until a rerun
    st.session_state["prompt"] = the_prompt
    st.session_state["continuation"] = continuation
    st.rerun()


# tx @cocktailpeanut
parser = argparse.ArgumentParser(description="mlx-ui")
parser.add_argument("--models", type=str, help="the txt file that contains the models list", default="models.txt")
args = parser.parse_args()
models_file = args.models

assistant_greeting = "How may I help you?"

with open(models_file, 'r') as file:
    model_refs = [line.strip() for line in file.readlines() if not line.startswith('#')]

model_refs = {k.strip(): v.strip() for k, v in [line.split("|") for line in model_refs]}

st.set_page_config(
    page_title=title,
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.title(title)

st.markdown(r"<style>.stDeployButton{display:none}</style>", unsafe_allow_html=True)


@st.cache_resource(show_spinner=True)
def load_model_and_cache(ref):
    return load(ref, {"trust_remote_code": True})


model = None

model_ref = st.sidebar.selectbox("model", model_refs.keys(), format_func=lambda value: model_refs[value],
                                 help="See https://huggingface.co/mlx-community for more models. Add your favorites "
                                      "to models.txt")

if model_ref.strip() != "-":
    model, tokenizer = load_model_and_cache(model_ref)

    chat_template = tokenizer.chat_template or (
        "{% for message in messages %}"
        "{{'<|im_start|>' + message['role'] + '\n' + message['content'] + '<|im_end|>' + '\n'}}"
        "{% endfor %}"
        "{% if add_generation_prompt %}"
        "{{ '<|im_start|>assistant\n' }}"
        "{% endif %}"
    )
    supports_system_role = "system role not supported" not in chat_template.lower()

    system_prompt = st.sidebar.text_area("system prompt", "You are a helpful AI assistant trained on a vast amount of "
                                                          "human knowledge. Answer as concisely as possible.",
                                         disabled=not supports_system_role)

    context_length = st.sidebar.number_input('context length', value=400, min_value=100, step=100, max_value=32000,
                                             help="how many maximum words to print, roughly")

    temperature = st.sidebar.slider('temperature', min_value=0., max_value=1., step=.10, value=.5,
                                    help="lower means less creative but more accurate")

    st.sidebar.markdown("---")
    actions = st.sidebar.columns(2)

    # give a bit of time for sidebar widgets to render
    time.sleep(0.05)

    if "messages" not in st.session_state:
        st.session_state["messages"] = [{"role": "assistant", "content": assistant_greeting}]

    stop_words = ["<|im_start|>", "<|im_end|>", "<s>", "</s>"]

    if actions[0].button("😶‍🌫️ Forget", use_container_width=True,
                         help="Forget the previous conversations."):
        st.session_state.messages = [{"role": "assistant", "content": assistant_greeting}]
        if "prompt" in st.session_state and st.session_state["prompt"]:
            st.session_state["prompt"] = None
            st.session_state["continuation"] = None
        st.rerun()

    if actions[1].button("🔂 Continue", use_container_width=True,
                         help="Continue the generation."):

        user_prompts = [msg["content"] for msg in st.session_state.messages if msg["role"] == "user"]

        if user_prompts:

            last_user_prompt = user_prompts[-1]

            assistant_responses = [msg["content"] for msg in st.session_state.messages
                                   if msg["role"] == "assistant" and msg["content"] != assistant_greeting]
            last_assistant_response = assistant_responses[-1] if assistant_responses else ""

            # remove last line completely, so it is regenerated correctly (in case it stopped mid-word or mid-number)
            last_assistant_response_lines = last_assistant_response.split('\n')
            if len(last_assistant_response_lines) > 1:
                last_assistant_response_lines.pop()
                last_assistant_response = "\n".join(last_assistant_response_lines)

            messages = [
                {"role": "user", "content": last_user_prompt},
                {"role": "assistant", "content": last_assistant_response},
            ]
            if supports_system_role:
                messages.insert(0, {"role": "system", "content": system_prompt})

            full_prompt = tokenizer.apply_chat_template(messages,
                                                        tokenize=False,
                                                        add_generation_prompt=False,
                                                        chat_template=chat_template)
            full_prompt = full_prompt.rstrip("\n")

            # remove last assistant response from state, as it will be replaced with a continued one
            remove_last_occurrence(st.session_state.messages,
                                   lambda msg: msg["role"] == "assistant" and msg["content"] != assistant_greeting)

            queue_chat(full_prompt, last_assistant_response)

    if prompt := st.chat_input():
        st.session_state.messages.append({"role": "user", "content": prompt})

        messages = []
        if supports_system_role:
            messages += [{"role": "system", "content": system_prompt}]
        messages += build_memory()
        messages += [{"role": "user", "content": prompt}]

        full_prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                                                    chat_template=chat_template)
        full_prompt = full_prompt.rstrip("\n")

        queue_chat(full_prompt)

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).write(msg["content"])

    # give a bit of time for messages to render
    time.sleep(0.05)

    if "prompt" in st.session_state and st.session_state["prompt"]:
        show_chat(st.session_state["prompt"], st.session_state["continuation"])
        st.session_state["prompt"] = None
        st.session_state["continuation"] = None

st.sidebar.markdown("---")
st.sidebar.markdown(f"v{ver} / st {st.__version__}")
