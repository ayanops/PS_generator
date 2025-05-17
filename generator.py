# Webアプリ用：Markov連鎖でペン回しオーダーを生成するStreamlitアプリ
# ローカルで動かすなら：streamlit run generator.py

import streamlit as st
import markovify
import random
import re
import pandas as pd
from collections import Counter
import altair as alt

# --- テキスト読み込み ---
@st.cache_data
def load_text(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return f.read()

# --- Markovify 用クラス ---
class TrickText(markovify.Text):
    def sentence_split(self, text):
        return text.split("\n")

    def word_split(self, sentence):
        return re.split(r"[ 　]+", sentence.strip())

    def word_join(self, words):
        return " ".join(words)

# --- モデル構築 ---
corpus_text = load_text("Sangkm13th_simplified.txt")
text_model = TrickText(corpus_text, state_size=1)

# --- UI：タイトル ---
st.title("Penspinning Order Generator from Sangkm 13th")
st.write("Generate a random pen spinning combo using Markov chain trained on Sangkm 13th.")

# --- Trick候補の取得 ---
state_keys = list(text_model.chain.model.keys())
trick_list = sorted(set("".join(k) for k in state_keys))

# --- First / Last Trick Selection ---
first = st.selectbox("First trick", options=[""] + trick_list, index=0)
last = st.selectbox("Last trick", options=[""] + trick_list, index=0)

# --- 長さの選択 ---
length = st.slider("Maximum number of tricks", min_value=5, max_value=30, value=15)

# --- 生成関数 ---
def generate_order(first_word, last_word="", n=15, max_attempts=200):
    state_keys = list(text_model.chain.model.keys())
    str_keys = ["".join(k) for k in state_keys]

    for _ in range(max_attempts):
        if first_word in str_keys:
            idx = str_keys.index(first_word)
            init_state = state_keys[idx]
        else:
            init_state = random.choice(state_keys)
        seq = list(init_state) + text_model.chain.walk(init_state)
        seq = seq[:n]
        if not last_word or seq[-1] == last_word:
            return " → ".join(seq)
    return None

# --- ボタンが押されたら生成 ---
if st.button("Generate"):
    if first and first not in trick_list:
        st.warning(f"The selected first trick '{first}' is not in the model.")
    order = generate_order(first, last, length)
    if order:
        st.success("Generated Order:")
        st.write(order)
    else:
        st.error("No valid order could be generated with the given conditions.")

# --- 出現頻度分析 ---
st.subheader("Trick Frequency")

top_n = st.slider("Top N tricks to show", min_value=5, max_value=50, value=20, step=5)
sort_order = st.radio("Sort order", options=["High to Low", "Low to High"])

tokens = []
for line in corpus_text.splitlines():
    tokens.extend(re.split(r"[ 　]+", line.strip()))
freq = Counter(tokens)

df = pd.DataFrame(freq.items(), columns=["Trick", "Frequency"])
ascending = True if sort_order == "Low to High" else False
df = df.sort_values("Frequency", ascending=ascending).head(top_n)

chart = alt.Chart(df).mark_bar().encode(
    x=alt.X("Frequency:Q"),
    y=alt.Y("Trick:N", sort=None)
).properties(
    height=alt.Step(20)
)

st.altair_chart(chart, use_container_width=True)

