from mlx_lm import load
from mlx_lm.utils import generate_step
import mlx.core as mx
import streamlit as st
from tqdm import tqdm

tqdm(disable=True, total=0)  # initialise internal lock

title = "MLX Chat"

st.set_page_config(
    page_title=title,
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)
st.title(title)

model_ref = st.sidebar.text_input("model", "mlx-community/Nous-Hermes-2-Mixtral-8x7B-DPO-4bit")
prompt_sys = st.sidebar.text_input("system prompt", "You are an AI assistant, a large language model trained by awesome data scientists. Answer as concisely as possible.")
n_ctx = st.sidebar.number_input('n_ctx', 100)


@st.cache_resource(show_spinner=True, hash_funcs={str: lambda x: None})
def load_model(model_ref):
    return load(model_ref)


model, tokenizer = load_model(model_ref)


def generate(prompt, model):
    tokens = []
    skip = 0
    for token, _ in zip(generate_step(mx.array(tokenizer.encode(prompt)), model), range(n_ctx)):
        if token == tokenizer.eos_token_id:
            break
        tokens.append(token.item())
        s = tokenizer.decode(tokens)
        yield s[skip:]
        skip = len(s)


# with st.sidebar:
#     pass

if "messages" not in st.session_state:
    st.session_state["messages"] = [{"role": "assistant", "content": "How can I help you?"}]

for msg in st.session_state.messages:
    st.chat_message(msg["role"]).write(msg["content"])

if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})
    st.chat_message("user").write(prompt)

    full_prompt = (f"<|im_start|>system\n{prompt_sys}<|im_end|>"
                   f"<|im_start|>user\n{prompt}<|im_end|>"
                   f"<|im_start|>assistant\n")

    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        response = ""

        for chunk in generate(full_prompt, model):
            response += chunk
            message_placeholder.markdown(response + "▌")

        message_placeholder.markdown(response)

    st.session_state.messages.append({"role": "assistant", "content": response})
