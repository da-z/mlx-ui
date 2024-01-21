import time

import mlx.core as mx
import streamlit as st
from mlx_lm import load
from mlx_lm.utils import generate_step

title = "MLX Chat"
ver = "0.7.5"
debug = False

with open('models.txt', 'r') as file:
    model_refs = file.readlines()
model_refs = [line.strip() for line in model_refs]

st.set_page_config(
    page_title=title,
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.title(title)

assistant_greeting = "How may I help you?"

model_ref = st.sidebar.selectbox("model", model_refs,
                                 help="See https://huggingface.co/mlx-community for more models. Add your favorites "
                                      "to models.txt")

system_prompt = st.sidebar.text_area("system prompt", "You are a helpful AI assistant trained on a vast amount of "
                                                      "human knowledge. Answer as concisely as possible.")

context_length = st.sidebar.number_input('context length', value=400, min_value=100, step=100, max_value=32000,
                                         help="how many maximum words to print, roughly")

temperature = st.sidebar.slider('temperature', min_value=0., max_value=1., step=.10, value=.7,
                                help="lower means less creative but more accurate")

st.sidebar.markdown("---")
actions = st.sidebar.columns(2)

st.sidebar.markdown("---")
st.sidebar.markdown(f"v{ver} / st {st.__version__}")

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": assistant_greeting}]


@st.cache_resource(show_spinner=True)
def load_model(ref):
    return load(ref)


model, tokenizer = load_model(model_ref)


def generate(the_prompt, the_model):
    tokens = []
    skip = 0
    for token, _ in zip(generate_step(mx.array(tokenizer.encode(the_prompt)), the_model, temperature),
                        range(context_length)):
        if token == tokenizer.eos_token_id:
            break
        tokens.append(token.item())
        s = tokenizer.decode(tokens)
        yield s[skip:]
        skip = len(s)


def show_chat(the_prompt, previous=""):
    # hack. give a bit of time to draw the UI before going into this long-running process
    time.sleep(0.05)

    if debug:
        print(the_prompt)
        print("-" * 80)

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        response = previous

        for chunk in generate(the_prompt, model):
            response = (response + chunk).replace('�', '')
            message_placeholder.markdown(response + "▌")

        message_placeholder.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})


def remove_last_occurrence_in_array(array_of_dicts, criteria):
    for i in reversed(range(len(array_of_dicts))):
        if criteria(array_of_dicts[i]):
            del array_of_dicts[i]
            break


def build_memory_prompt():
    mem = ""
    if len(st.session_state.messages) > 2:
        mem += "\n\n##START_PREVIOUS_DISCUSSION## (do not repeat it in chat but use it as context)"
        for msg in st.session_state.messages[1:-1]:
            mem += f"\n{'ME' if msg['role'] == 'assistant' else 'USER'}:\n{msg['content'].strip()}\n"
        mem += "##END_PREVIOUS_DISCUSSION##\n"
    return mem


for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    prompt_sys_with_memory = system_prompt + build_memory_prompt()

    full_prompt = tokenizer.apply_chat_template([
        {"role": "system", "content": prompt_sys_with_memory},
        {"role": "user", "content": prompt},
    ], tokenize=False, add_generation_prompt=True)
    full_prompt = full_prompt.rstrip("\n")

    last_chat_element = st.empty()
    show_chat(full_prompt)

if st.session_state.messages and sum(msg["role"] == "assistant" for msg in st.session_state.messages) > 1:
    if actions[0].button("😶‍🌫️ Forget", use_container_width=True,
                         help="Forget the previous conversations."):
        st.session_state.messages = [{"role": "assistant", "content": assistant_greeting}]
        # st.rerun()

if st.session_state.messages and sum(msg["role"] == "assistant" for msg in st.session_state.messages) > 1:
    if actions[1].button("🔂 Continue", use_container_width=True,
                         help="Continue the generation."):

        user_prompts = [msg["content"] for msg in st.session_state.messages if msg["role"] == "user"]
        last_prompt = user_prompts[-1] or "Please continue your response."

        assistant_responses = [msg["content"] for msg in st.session_state.messages if msg["role"] == "assistant"]
        last_assistant_response = assistant_responses[-1] if assistant_responses else ""

        # remove last line completely, so it is regenerated correctly (in case it stopped mid-word or mid-number)
        last_assistant_response_lines = last_assistant_response.split('\n')
        if len(last_assistant_response_lines) > 1:
            last_assistant_response_lines.pop()
            last_assistant_response = "\n".join(last_assistant_response_lines)

        full_prompt = tokenizer.apply_chat_template([
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": last_prompt},
            {"role": "assistant", "content": last_assistant_response},
        ], tokenize=False, add_generation_prompt=True)
        full_prompt = full_prompt.rstrip("\n")

        # replace last assistant response from state, as it will be replaced with a continued one
        # strangely, the chat messages are not refreshed - workaround: click on +/- on the 'context length' field
        remove_last_occurrence_in_array(st.session_state.messages, lambda msg: msg["role"] == "assistant")

        show_chat(full_prompt, last_assistant_response)
