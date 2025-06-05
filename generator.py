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


# --- slidebarレイアウト作成、モード選択 ---
lang_mode = st.sidebar.radio("Choose language", options=["日本語","English"], index=0)

with st.sidebar:
    st.markdown("## Information")
    st.markdown("""
    Version: 1.0.0  
    Last update: May 2025  
    [GitHub Repository](https://github.com/ayanops/PS_generator)  
      
    Creater: ayaNo  
    Contact: [X](https://x.com/ayanops), [Youtube](https://www.youtube.com/channel/UCeuUf2nRyGRir2mtmn9By6g?app=desktop)
    """)


# --- テキストファイルの選択 ---
file_path = "Sangkm13th_simplified.txt" if lang_mode == "English" else "Sangkm13th_japanese.txt"
corpus_text = load_text(file_path)

# --- モデル構築 ---
text_model = TrickText(corpus_text, state_size=1)

# --- UI：タイトル ---
desc = "Generate a random pen spinning combo using Markov chain trained on Sangkm 13th." if lang_mode == "English" else "Sangkm 13thからマルコフ連鎖を用いて学習したペン回しオーダーを生成します。"
st.title("Penspinning Combo Generator")
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
def generate_order(first_word, last_word="", n=15, max_attempts=500):
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
generate_button = "Generate" if lang_mode == "English" else "生成"
if st.button(generate_button):
    if first and first not in trick_list:
        st.warning(f"The selected first trick '{first}' is not in the model.")
    order = generate_order(first, last, length)
    if order:
        msg = f"Generated Order from **{first}** to **{last}**:" if lang_mode == "English" else f"**{first}** から **{last}** までの生成コンボ："
        st.success(msg)
        st.write(order)
    else:
        msg = f"No valid order could be generated from {first} to {last}." if lang_mode == "English" else f"{first} から {last} の間では有効なコンボを生成できませんでした。"
        st.error(msg)

# --- ランダム生成ボタン ---

# last trick 候補を定義
preset_last_tricks = (["11Sp","12Sp","122Sp","1212Sp","121Sp","1211Sp","22Sp","222Sp",
                      "2BackSA33","BackaroundFall","FLTA","NeoSA233","RayGun","Thumbaround"]
                      if lang_mode == "English" 
                      else ["11スプレッド","12スプレッド","122スプレッド","1212スプレッド","121スプレッド","1211スプレッド","22スプレッド","222スプレッド",
                      "2バクアラSA33","シャフィーボ","フィンガーレスノーマル","NeoSA233","レイガン","ノーマル"])

# ボタン押下時にランダム生成
random_button = "Generate random combo (finish with spread or around)" if lang_mode == "English" else "ランダム生成 (スプレッドかアラウンドで締め)"
st.markdown("---")
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
st.markdown("---")



# --- 学習元オーダーの表示 ---
st.markdown("### Original data used for Training" if lang_mode == "English" else "### 学習データ")
st.markdown("#### Combos" if lang_mode == "English" else "#### フリースタイル")

order_labels = [
    "KTH", "WhiteTiger", "Woojung", "Angmaramyon_a", "Uriel",
    "Nagi", "Fresh-gel-_-v", "Nory", "Ferrari", "taeryong",
    "syugen", "Morse", "Raply", "Sound", "Vision", "Flip",
    "Xien", "Nanna", "Shahell", "chunhwang", "CloudTraveller",
    "Biee", "Outsider", "Pashas"
]

FS_check = "Show" if lang_mode == "English" else "表示"
show_FS = st.checkbox(FS_check, value=False, key="show_FS_checkbox")

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

    # チェックボックスがオンの場合のみ表示
    if show_FS:
        st.dataframe(labeled_orders, use_container_width=True, width=0)


# --- 出現頻度分析 ---
st.markdown("#### Trick Frequency" if lang_mode == "English" else "#### トリック出現数")

# 先に tokens を準備（表示の有無に関係なく）
tokens = []
for line in corpus_text.splitlines():
    tokens.extend(re.split(r"[ 　]+", line.strip()))
freq = Counter(tokens)
df = pd.DataFrame(freq.items(), columns=["Trick", "Frequency"])

# 表示するかどうかのチェックボックス
freq_check = "Show" if lang_mode == "English" else "表示"
show_freq = st.checkbox(freq_check, value=False, key="show_freq_checkbox")

if show_freq:
    # 高頻度順にソート
    df = df.sort_values("Frequency", ascending=False)

    # Altairグラフ
    chart = alt.Chart(df).mark_bar().encode(
        x=alt.X("Frequency:Q", title="Frequency"),
        y=alt.Y("Trick:N", sort=None, title="Trick")
    ).properties(
        height=20 * len(df),
        width='container'
    )

    # HTMLに変換してスクロール付きdivに埋め込み
    html = f"""
    <div style="height:500px; overflow-y:auto; border:1px solid #ccc; padding:10px">
      {chart.to_html()}
    </div>
    """

    components.html(html, height=520, scrolling=True)
st.markdown("---")



# --- ネットワークの可視化 ---
net_header = "Network analysis" if lang_mode == "English" else "ネットワーク分析"
st.subheader(net_header)

# ネットワークを表示するかどうか
show_check = "Show full network" if lang_mode == "English" else "全体のネットワークを表示"
show_network = st.checkbox(show_check, value=False)

if show_network:
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
    min_count_label = "Minimum trick frequency" if lang_mode == "English" else "最小出現回数"
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
st.markdown("#### Explore by Trick" if lang_mode == "English" else "#### トリック別分析")
focus_header = "Select a Trick to Explore" if lang_mode == "English" else "トリックを選択"
focus_trick = st.selectbox(focus_header, options=sorted(set(tokens)))

# 前後関係 + 出現頻度の取得
pair_counts = Counter()
neighbor_counts = Counter()

for line in corpus_text.splitlines():
    tricks = re.split(r"[ 　]+", line.strip())
    for i, t in enumerate(tricks):
        if t == focus_trick:
            if i > 0:
                prev = tricks[i - 1]
                pair_counts[(prev, t)] += 1
                neighbor_counts[prev] += 1
            if i < len(tricks) - 1:
                nxt = tricks[i + 1]
                pair_counts[(t, nxt)] += 1
                neighbor_counts[nxt] += 1

# --- グラフ構築 ---
net = Network(height="500px", width="100%", directed=True)
net.add_node(focus_trick, color="#FF6347", size=20)

# ノード追加（前後トリック）
for node, freq in neighbor_counts.items():
    size = 10 + freq * 2
    color = "#1f77b4" if (node, focus_trick) in pair_counts else "#2ca02c"  # 前:青 / 後:緑
    net.add_node(node, label=node, size=size, color=color)

# エッジ追加
for (a, b), count in pair_counts.items():
    net.add_edge(a, b, title=f"{a} → {b}: {count} times")

# --- 表示 ---
net.save_graph("trick_graph.html")
with open("trick_graph.html", "r", encoding="utf-8") as HtmlFile:
    components.html(HtmlFile.read(), height=550)


# --- 出現行の逆引き ---
st.markdown("#### Spinner")
lines_with_focus = []

# 出現行のリスト作成
html_lines = []
for idx, line in enumerate(corpus_text.splitlines()):
    label = order_labels[idx]
    if focus_trick in line:
        pattern = re.escape(focus_trick)
        highlighted = re.sub(
            pattern,
            f"<span style='color:red; font-weight:bold'>{focus_trick}</span>",
            line
        )
        html_lines.append(f"<b>{label}</b>: {highlighted}")

if html_lines:
    full_html = """
    <div style='max-height: 300px; overflow-y: auto; padding: 10px;
                border: 1px solid #ddd; border-radius: 6px;
                background-color: #f9f9f9; font-family: monospace;'>
    """ + "<br>".join(html_lines) + "</div>"

    st.markdown(full_html, unsafe_allow_html=True)
else:
    st.info("このトリックはどの行にも含まれていません。")
