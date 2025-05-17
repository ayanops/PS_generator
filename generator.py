# app.py
# Webアプリ用：Markov連鎖でペン回しオーダーを生成するStreamlitアプリ
# streamlit run generator.py
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

# --- Streamlit UI ---
st.title("Penspinning Order Generator from Sangkm 13th")
st.write("Markov連鎖を使ってSangkm 13thからペン回しオーダーを自動生成します。")

# 入力フォーム
first = st.text_input("開始トリック (必須) / First trick (required)", value="Example: 34-23TwistedSonic", key="first_trick")
last = st.text_input("最終トリック (任意) / Last trick（optional）", value="", key="last_trick")
length = st.slider("トリック最大数 / Maximum number of tricks", min_value=5, max_value=30, value=15)

# --- 生成関数 ---
def generate_order(first_word, last_word="", n=15, max_attempts=100):
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
    state_keys = list(text_model.chain.model.keys())
    str_keys = ["".join(k) for k in state_keys]
    if first not in str_keys:
        st.warning(f"指定された開始トリック「{first}」は学習データに存在しません。")
    order = generate_order(first, last, length)
    if order:
        st.success("生成されたオーダー:")
        st.write(order)
    else:
        st.error("条件に一致するオーダーが見つかりませんでした。")

# --- Available tricks ---
st.subheader("Exmaples of tricks")
state_keys = list(text_model.chain.model.keys())
str_keys = ["".join(k) for k in state_keys]
st.write(str_keys[:10])

# --- トークン頻度分析 ---
st.subheader("出現頻度")

# 表示数の選択
top_n = st.slider("表示する上位トリック数", min_value=5, max_value=50, value=20, step=5)

# 並び順の選択
sort_order = st.radio("並び順", options=["頻度が高い順", "頻度が低い順"])

# トークンの抽出と頻度計算
tokens = []
for line in corpus_text.splitlines():
    tokens.extend(re.split(r"[ 　]+", line.strip()))
freq = Counter(tokens)

# データフレームの作成と並び替え
df = pd.DataFrame(freq.items(), columns=["トリック", "出現回数"])
ascending = True if sort_order == "頻度が低い順" else False
df = df.sort_values("出現回数", ascending=ascending).head(top_n)

# Altairで棒グラフを作成（高さを動的に調整）
chart = alt.Chart(df).mark_bar().encode(
    x=alt.X("出現回数:Q"),
    y=alt.Y("トリック:N", sort=None)
).properties(
    height=alt.Step(20)  # 各トリックに対するステップサイズを指定
)


st.altair_chart(chart, use_container_width=True)
