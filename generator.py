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


# --- モード選択 ---
st.sidebar.title("Language Mode")
lang_mode = st.sidebar.radio("Choose language", options=["English", "日本語"], index=0)

# --- テキストファイルの選択 ---
file_path = "Sangkm13th_simplified.txt" if lang_mode == "English" else "Sangkm13th_japanese.txt"
corpus_text = load_text(file_path)

# --- モデル構築 ---
text_model = TrickText(corpus_text, state_size=1)

# --- UI：タイトル ---
st.title("Penspinning Order Generator from Sangkm 13th")
st.write("Generate a random pen spinning combo using Markov chain trained on Sangkm 13th.")
st.video("https://www.youtube.com/watch?v=r4qq_tkwH1E")

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


# --- 追加機能：ランダム生成ボタン ---
st.markdown("---")
st.subheader("Alternative Generation")

# あなたが指定したい last trick 候補を定義
preset_last_tricks = ["11Sp","12Sp","122Sp","1212Sp","121Sp","1211Sp","22Sp","222Sp",
                      "2BackSA33","BackaroundFall","FLTA","NeoSA233","RayGun","Thumbaround"]

# ボタン押下時にランダム生成
if st.button("Generate (Random First & random finish trick)"):
    random_first = random.choice(trick_list)
    random_last = random.choice(preset_last_tricks)

    order = generate_order(random_first, random_last, length)
    if order:
        st.success(f"Generated Order from **{random_first}** to **{random_last}**:")
        st.write(order)
    else:
        st.error(f"No valid order could be generated from {random_first} to {random_last}.")


# --- 学習元オーダーの表示 ---
st.subheader("Original Orders used for Training")

order_labels = [
    "KTH", "WhiteTiger", "Woojung", "Angmaramyon_a", "Uriel",
    "Nagi", "Fresh-gel-_-v", "Nory", "Ferrari", "taeryong",
    "syugen", "Morse", "Raply", "Sound", "Vision", "Flip",
    "Xien", "Nanna", "Shahell", "chunhwang", "CloudTraveller",
    "Biee", "Outsider", "Pashas"
]

# 行ごとに分割（空行を除外）
corpus_lines = [line.strip() for line in corpus_text.strip().splitlines() if line.strip()]

# ラベルと行数の整合性をチェック
if len(order_labels) != len(corpus_lines):
    st.error(f"ラベル数（{len(order_labels)}）とデータ行数（{len(corpus_lines)}）が一致していません。")
else:
    labeled_orders = pd.DataFrame({
        "Spinner": order_labels,
        "Trick Sequence": corpus_lines
    })
    st.dataframe(labeled_orders, use_container_width=True, width=0)


import streamlit.components.v1 as components

# --- 出現頻度分析 ---
st.subheader("Trick Frequency")

# ソート順
sort_order = st.radio("Sort order", options=["High to Low", "Low to High"])

# 頻度カウント
tokens = []
for line in corpus_text.splitlines():
    tokens.extend(re.split(r"[ 　]+", line.strip()))
freq = Counter(tokens)
df = pd.DataFrame(freq.items(), columns=["Trick", "Frequency"])

# ソート
ascending = True if sort_order == "Low to High" else False
df = df.sort_values("Frequency", ascending=ascending)

# Altairグラフ（縦長でもOK）
chart = alt.Chart(df).mark_bar().encode(
    x=alt.X("Frequency:Q", title="Frequency"),
    y=alt.Y("Trick:N", sort=None, title="Trick")
).properties(
    height=20 * len(df),  # 全トリック分の高さで生成
    width='container'
)

# HTMLに変換してスクロール付きdivに埋め込み
html = f"""
<div style="height:500px; overflow-y:auto; border:1px solid #ccc; padding:10px">
  {chart.to_html()}
</div>
"""

components.html(html, height=520, scrolling=True)
