# Webアプリ用：Markov連鎖でペン回しオーダーを生成するStreamlitアプリ
# ローカルで動かすなら：streamlit run generator.py

import streamlit as st
import markovify
import random
import re
import pandas as pd
from collections import Counter
import altair as alt
from pyvis.network import Network
import streamlit.components.v1 as components


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
lang_mode = st.sidebar.radio("Choose language", options=["日本語","English"], index=0)

# --- テキストファイルの選択 ---
file_path = "Sangkm13th_simplified.txt" if lang_mode == "English" else "Sangkm13th_japanese.txt"
corpus_text = load_text(file_path)

# --- モデル構築 ---
text_model = TrickText(corpus_text, state_size=1)

# --- UI：タイトル ---
desc = "Generate a random pen spinning combo using Markov chain trained on Sangkm 13th." if lang_mode == "English" else "Sangkm 13thからマルコフ連鎖を用いて学習したペン回しオーダーを生成します。"
st.title("Penspinning Combo Generator from Sangkm 13th")
st.write(desc)
st.video("https://www.youtube.com/watch?v=r4qq_tkwH1E")

# --- Trick候補の取得 ---
state_keys = list(text_model.chain.model.keys())
trick_list = sorted(set("".join(k) for k in state_keys))

# --- First / Last Trick Selection ---
first_label = "First trick" if lang_mode == "English" else "開始トリック"
last_label = "Last trick" if lang_mode == "English" else "終了トリック"
first = st.selectbox(first_label, options=[""] + trick_list, index=0)
last = st.selectbox(last_label, options=[""] + trick_list, index=0)

# --- 長さの選択 ---
length_label = "Maximum number of tricks" if lang_mode == "English" else "最大トリック数"
length = st.slider(length_label, min_value=5, max_value=30, value=15)

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


# --- ランダム生成ボタン ---

# last trick 候補を定義
preset_last_tricks = ["11Sp","12Sp","122Sp","1212Sp","121Sp","1211Sp","22Sp","222Sp",
                      "2BackSA33","BackaroundFall","FLTA","NeoSA233","RayGun","Thumbaround"]

# ボタン押下時にランダム生成
alt_title = "Alternative Generation" if lang_mode == "English" else "ランダム生成"
random_button = "Generate (Random First & random finish trick)" if lang_mode == "English" else "ランダム生成（ランダム開始・締めトリックまで）"
st.markdown("---")
st.subheader(alt_title)
if st.button(random_button):
    random_first = random.choice(trick_list)
    random_last = random.choice(preset_last_tricks)
    order = generate_order(random_first, random_last, length)
    if order:
        msg = f"Generated Order from **{random_first}** to **{random_last}**:" if lang_mode == "English" else f"**{random_first}** から **{random_last}** までの生成コンボ："
        st.success(msg)
        st.write(order)
    else:
        msg = f"No valid order could be generated from {random_first} to {random_last}." if lang_mode == "English" else f"{random_first} から {random_last} の間では有効なコンボを生成できませんでした。"
        st.error(msg)


# --- 学習元オーダーの表示 ---
corpus_header = "Original Orders used for Training" if lang_mode == "English" else "学習に使用されたオーダー一覧"
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
freq_header = "Trick Frequency" if lang_mode == "English" else "トリック出現頻度"
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



# --- ネットワークの可視化---
net_header = "Trick Transition Network (Filtered by Frequency)" if lang_mode == "English" else "トリック遷移ネットワーク（頻度でフィルター）"
st.subheader(net_header)

# トリック出現回数カウント
token_counts = Counter()
for line in corpus_text.splitlines():
    tricks = re.split(r"[ 　]+", line.strip())
    token_counts.update(tricks)

# 遷移ペアの出現回数カウント
transition_counts = Counter()
for line in corpus_text.splitlines():
    tricks = re.split(r"[ 　]+", line.strip())
    for i in range(len(tricks) - 1):
        transition_counts[(tricks[i], tricks[i+1])] += 1

# --- 出現回数でフィルタ ---
min_count_label = "Minimum trick frequency to include in graph" if lang_mode == "English" else "グラフに含める最小出現回数"
min_count = st.slider(min_count_label, min_value=1, max_value=20, value=3, step=1)

# 有効トリック = 出現回数がmin_count以上のもの
valid_tricks = {trick for trick, count in token_counts.items() if count >= min_count}

# Pyvis グラフ作成
net = Network(height="600px", width="100%", directed=True)
added_nodes = set()

for (a, b), count in transition_counts.items():
    if a in valid_tricks and b in valid_tricks:
        # ノード追加（出現回数に応じてサイズ指定）
        for trick in (a, b):
            if trick not in added_nodes:
                size = token_counts[trick] * 1.5  # ノードサイズ調整
                net.add_node(trick, label=trick, size=size)
                added_nodes.add(trick)

        # エッジ追加（遷移頻度に応じて太さ調整）
        net.add_edge(a, b, value=count, title=f"{a} → {b}: {count}")

# 保存して読み込む
net.save_graph("filtered_trick_graph.html")
with open("filtered_trick_graph.html", "r", encoding="utf-8") as f:
    components.html(f.read(), height=620, scrolling=True)



# --- 特定の trick に注目---
st.subheader("Trick Relationship Explorer" if lang_mode == "English" else "トリックごとの探索")
focus_header = "Select a Trick to Explore" if lang_mode == "English" else "トリックを選択"
focus_trick = st.selectbox(focus_header, options=sorted(set(tokens)))

# 前後関係をカウント
pair_counts = Counter()
for line in corpus_text.splitlines():
    tricks = re.split(r"[ 　]+", line.strip())
    for i, t in enumerate(tricks):
        if t == focus_trick:
            if i > 0:
                pair_counts[(tricks[i-1], t)] += 1
            if i < len(tricks) - 1:
                pair_counts[(t, tricks[i+1])] += 1

# グラフ構築
net = Network(height="500px", width="100%", directed=True)
net.add_node(focus_trick, color="red")

for (a, b), count in pair_counts.items():
    net.add_node(a)
    net.add_node(b)
    net.add_edge(a, b, value=count, title=f"{a} → {b}: {count}")

# 表示
net.save_graph("trick_graph.html")
HtmlFile = open("trick_graph.html", 'r', encoding='utf-8')
components.html(HtmlFile.read(), height=550)


