# anaconda运行
# streamlit run "D:\软件\四级词汇比例-频率-缺失率\003-English_Reader_018.py"
import streamlit as st
import streamlit.components.v1 as components
import pandas as pd
from pathlib import Path
from collections import Counter
import plotly.express as px
import plotly.graph_objects as go
import numpy as np
import re
from nltk.stem import WordNetLemmatizer
import nltk
import requests
from io import BytesIO
import json
import html
import hashlib
import secrets
import asyncio
import edge_tts
import tempfile

# ------------------- 依存标签英文注释 -------------------
DEPREL_LABELS = {
    "nsubj": "subject",
    "nsubj:pass": "passive subject",
    "obj": "object",
    "iobj": "indirect object",
    "csubj": "clausal subject",
    "csubj:pass": "passive clausal subject",
    "ccomp": "clausal complement",
    "xcomp": "open clausal complement",
    "obl": "oblique nominal",
    "obl:npmod": "noun phrase modifier",
    "obl:tmod": "temporal modifier",
    "advmod": "adverbial modifier",
    "amod": "adjectival modifier",
    "nmod": "nominal modifier",
    "nmod:poss": "possessive modifier",
    "appos": "appositional modifier",
    "nummod": "numeric modifier",
    "acl": "adjectival clause",
    "acl:relcl": "relative clause",
    "advcl": "adverbial clause",
    "det": "determiner",
    "det:predet": "predeterminer",
    "aux": "auxiliary",
    "aux:pass": "passive auxiliary",
    "cop": "copula",
    "mark": "marker",
    "case": "case marker",
    "cc": "coordinating conjunction",
    "conj": "conjunct",
    "fixed": "fixed multiword",
    "flat": "flat multiword",
    "compound": "compound",
    "compound:prt": "phrasal verb particle",
    "expl": "expletive",
    "parataxis": "parataxis",
    "discourse": "discourse element",
    "vocative": "vocative",
    "dep": "unspecified dependency",
    "root": "root",
    "ROOT": "root",
    "punct": "punctuation",
    "list": "list",
}

def label_deprel(rel: str) -> str:
    """返回 'English note (original)' 格式，未知标签直接返回原始值。"""
    note = DEPREL_LABELS.get(rel) or DEPREL_LABELS.get(rel.lower())
    if note:
        return f"{note} ({rel})"
    return rel


try:
    import graphviz
    GRAPHVIZ_AVAILABLE = True
except ImportError:
    GRAPHVIZ_AVAILABLE = False

# ------------------- NLTK 初始化 -------------------
nltk.download('wordnet', quiet=True)
nltk.download('omw-1.4', quiet=True)
lemmatizer = WordNetLemmatizer()

# ------------------- 路径配置 -------------------
# TEXTS_DIR = Path(r"D:\软件\四级词汇比例-频率-缺失率\Modern Library Top 100 Novels_AMAZON")
# DP_RESULTS_DIR = Path(r"D:\软件\四级词汇比例-频率-缺失率\Modern Library Top 100 Novels_AMAZON_DP_Analysis_Results")
# SINGLE_DIR = Path(r"D:\软件\四级词汇比例-频率-缺失率\Modern Library Top 100 Novels_AMAZON_Vocabulary_single")
# COMBINED_DIR = Path(r"D:\软件\四级词汇比例-频率-缺失率\Modern Library Top 100 Novels_AMAZON_Vocabulary_combined")
# WORDLISTS_DIR = Path(r"D:\软件\四级词汇比例-频率-缺失率\wordlists")

BASE_DIR = Path(__file__).parent
USERS_FILE = BASE_DIR / "users.json"
TEXTS_DIR = BASE_DIR / "data" / "novels"
DP_RESULTS_DIR = BASE_DIR / "data" / "dp_results"
SINGLE_DIR = BASE_DIR / "data" / "vocab_single"
COMBINED_DIR = BASE_DIR / "data" / "vocab_combined"
WORDLISTS_DIR = BASE_DIR / "data" / "wordlists"

PROGRESS_FILE = Path("progress.json")
# Base directory for user-specific data (progress, users, etc.)



# 内置书籍列表（按前缀编号进行数字排序）
def get_sorted_books():
    if not TEXTS_DIR.exists():
        return []

    def sort_key(stem):
        m = re.match(r"(\d+)_", stem)
        if m:
            return (int(m.group(1)), stem.lower())
        return (10_000, stem.lower())

    stems = [p.stem for p in TEXTS_DIR.glob("*.txt")]
    return sorted(stems, key=sort_key)


BUILT_IN_BOOKS = get_sorted_books()


# ------------------- 颜色与样式函数 -------------------
def get_color(freq):
    if freq <= 1:
        return "#8B00FF"
    elif 2 <= freq <= 10:
        return "#00BFFF"
    elif 11 <= freq <= 20:
        return "#FF8C00"
    else:
        return "#023020"


def get_font_style_by_frequency(freq):
    if freq <= 1:
        return "13px", "Cardo"
    elif 2 <= freq <= 10:
        return "14px", "Crimson Text"
    elif 11 <= freq <= 20:
        return "15px", "Lora"
    else:
        return "16px", "Merriweather"


# ------------------- 标准化字符串 -------------------
def normalize_string(s):
    s = re.sub(r'^\d+_', '', s)
    s = s.replace("_", " ")
    s = s.replace("'", "").replace('"', "")
    s = re.sub(r"[^\w\s]", "", s)
    s = " ".join(s.lower().split())
    return s


# ------------------- 查找DP文件夹 -------------------
def find_dp_folder(book_stem):
    if not DP_RESULTS_DIR.exists():
        return None

    clean_stem = normalize_string(book_stem)
    best_match = None
    best_match_score = 0

    for folder in DP_RESULTS_DIR.iterdir():
        if not folder.is_dir():
            continue

        clean_folder = normalize_string(folder.name)

        if clean_stem == clean_folder:
            return folder

        if clean_stem in clean_folder or clean_folder in clean_stem:
            match_score = len(clean_stem) if clean_stem in clean_folder else len(clean_folder)
            if match_score > best_match_score:
                best_match = folder
                best_match_score = match_score

    if best_match:
        return best_match

    for folder in DP_RESULTS_DIR.iterdir():
        if not folder.is_dir():
            continue

        stem_keywords = set(clean_stem.split()) - {'the', 'a', 'an', 'of', 'and', 'in', 'to'}
        folder_keywords = set(normalize_string(folder.name).split()) - {'the', 'a', 'an', 'of', 'and', 'in', 'to'}

        if stem_keywords and folder_keywords:
            overlap = len(stem_keywords & folder_keywords)
            if overlap >= len(stem_keywords) * 0.7:
                return folder

    return None


# ------------------- 加载词表 -------------------
@st.cache_data
def load_standard_wordlists():
    wordlists = {}
    if not WORDLISTS_DIR.exists():
        return wordlists

    file_names = ["COCA_20000_part1.txt", "COCA_20000_part2.txt", "COCA_20000_part3.txt", "COCA_20000_part4.txt"]

    for i, file_name in enumerate(file_names, 1):
        wordlist_path = WORDLISTS_DIR / file_name
        if wordlist_path.exists():
            with open(wordlist_path, 'r', encoding='utf-8') as f:
                lemmas = [line.strip().lower() for line in f if line.strip()]
            df = pd.DataFrame({'lemma': lemmas, 'order': range(len(lemmas))})
            wordlists[i] = df

    return wordlists


def load_wordlist_data(book_stem, dp_folder):
    single_path = SINGLE_DIR / f"{book_stem}_vocabulary_analysis.csv"
    combined_path = COMBINED_DIR / f"{book_stem}_vocabulary_analysis.csv"

    if single_path.exists():
        wordlist_df = pd.read_csv(single_path)
    elif combined_path.exists():
        wordlist_df = pd.read_csv(combined_path)
    else:
        freq_path = dp_folder / "lemma_frequency.csv"
        if freq_path.exists():
            wordlist_df = pd.read_csv(freq_path)
            if 'wordlist' not in wordlist_df.columns:
                standard_wordlists = load_standard_wordlists()
                wordlist_mapping = []
                for _, row in wordlist_df.iterrows():
                    lemma = str(row.get('lemma', '')).lower()
                    source = '未知'
                    for level, wl_df in standard_wordlists.items():
                        if lemma in wl_df['lemma'].values:
                            source = f'COCA_{level * 5000}'
                            break
                    wordlist_mapping.append(source)
                wordlist_df['wordlist'] = wordlist_mapping
        else:
            return pd.DataFrame()

    return wordlist_df


# ------------------- lemmatize 缓存（避免重复调用）-------------------
@st.cache_data(show_spinner=False)
def cached_lemmatize(word: str) -> str:
    return lemmatizer.lemmatize(word)


# ------------------- 加载书籍数据（带完整缓存）-------------------
@st.cache_data(show_spinner="Loading book data…")
def load_book_data(book_stem):
    dp_folder = find_dp_folder(book_stem)

    if dp_folder is None:
        st.error(f"Could not find a matching dependency-analysis folder (searching for '{book_stem}')")
        if DP_RESULTS_DIR.exists():
            st.write("Available folders (first 10):")
            for folder in list(DP_RESULTS_DIR.iterdir())[:10]:
                if folder.is_dir():
                    st.write(f"  - {folder.name}")
        st.stop()

    sentences_path = dp_folder / "sentences.csv"
    if not sentences_path.exists():
        st.error(f"未找到 sentences.csv：{sentences_path}")
        st.stop()
    sentences_df = pd.read_csv(sentences_path)

    if "sentence_id" in sentences_df.columns:
        sentences_df["sentence_id"] = pd.to_numeric(
            sentences_df["sentence_id"], errors="coerce"
        ).astype("Int64")

    dep_path = dp_folder / "dependencies.csv"
    dep_df = pd.read_csv(dep_path) if dep_path.exists() else pd.DataFrame()

    if not dep_df.empty and "sentence_id" in dep_df.columns:
        dep_df["sentence_id"] = pd.to_numeric(
            dep_df["sentence_id"], errors="coerce"
        ).astype("Int64")
        dep_df = dep_df[dep_df["sentence_id"].notna()]

    freq_path = dp_folder / "lemma_frequency.csv"
    global_freq_dict = {}
    if freq_path.exists():
        freq_df = pd.read_csv(freq_path)
        global_freq_dict = dict(zip(freq_df['lemma'].str.lower(), freq_df['frequency']))

    metrics_path = dp_folder / "metrics.csv"
    most_deprel_path = dp_folder / "most_common_deprels.csv"
    metrics_df = pd.read_csv(metrics_path) if metrics_path.exists() else pd.DataFrame()
    most_deprel_df = pd.read_csv(most_deprel_path) if most_deprel_path.exists() else pd.DataFrame()

    all_sentence_lemmas = []

    if "tokenized_sentence" in sentences_df.columns:
        for tokenized in sentences_df["tokenized_sentence"]:
            if pd.isna(tokenized):
                all_sentence_lemmas.append([])
                continue
            words = re.findall(r'\b[a-zA-Z]+\b', str(tokenized).lower())
            words = [w for w in words if len(w) > 1 and '-' not in w]
            lemmas = [cached_lemmatize(w) for w in words]
            all_sentence_lemmas.append(lemmas)
    else:
        all_sentence_lemmas = [[] for _ in range(len(sentences_df))]

    # ── 预计算累计频率前缀和（prefix_counters[i] = 前 i 句的累计 Counter）──
    sentence_deltas = []
    for lemmas in all_sentence_lemmas:
        sentence_deltas.append(Counter(lemmas))

    # ── 预建 dep_df 的 sentence_id 索引（dict: id -> sub-DataFrame）──
    dep_index = {}
    if not dep_df.empty and "sentence_id" in dep_df.columns:
        for sid, group in dep_df.groupby("sentence_id", sort=False):
            dep_index[sid] = group

    wordlist_df = load_wordlist_data(book_stem, dp_folder)

    return {
        "sentences": sentences_df,
        "all_sentence_lemmas": all_sentence_lemmas,
        "sentence_deltas": sentence_deltas,
        "global_freq_dict": global_freq_dict,
        "dep_df": dep_df,
        "dep_index": dep_index,
        "metrics": metrics_df,
        "most_deprel": most_deprel_df,
        "wordlist": wordlist_df
    }


# ------------------- 获取单词的句子列表 -------------------
def get_word_sentences(lemma, sentences_df, all_sentence_lemmas):
    matching_sentences = []
    for idx, sentence_lemmas in enumerate(all_sentence_lemmas):
        if lemma in sentence_lemmas:
            matching_sentences.append({
                "sentence_id": sentences_df.iloc[idx]["sentence_id"],
                "text": sentences_df.iloc[idx].get("tokenized_sentence", "")
            })
    return matching_sentences


# ------------------- 获取单词的依存关系 -------------------
def get_word_dependencies(lemma, dep_df):
    if dep_df.empty:
        return []

    lemma = str(lemma).lower()
    dep_pairs = []

    required_cols = ['dependent_text', 'head_text', 'deprel', 'sentence_id']
    if not all(col in dep_df.columns for col in required_cols):
        return []

    for idx, row in dep_df.iterrows():
        dep_text = str(row.get('dependent_text', ''))
        head_text = str(row.get('head_text', ''))
        deprel = str(row.get('deprel', ''))
        sent_id = row.get('sentence_id', 0)

        dep_clean = re.sub(r'[^\w]', '', dep_text.lower())
        head_clean = re.sub(r'[^\w]', '', head_text.lower())

        if not dep_clean or not head_clean:
            continue

        dep_lemma = cached_lemmatize(dep_clean)
        head_lemma = cached_lemmatize(head_clean)

        if dep_lemma == lemma or head_lemma == lemma:
            dep_pairs.append({

                'relation': f"{head_text} ──{label_deprel(deprel)}──> {dep_text}",
                'sentence_id': sent_id,
                'head_text': head_text,
                'dependent_text': dep_text,
                'deprel': deprel
            })

    seen = set()
    unique_pairs = []
    for pair in dep_pairs:
        key = (pair['relation'], pair['sentence_id'])
        if key not in seen:
            seen.add(key)
            unique_pairs.append(pair)

    unique_pairs.sort(key=lambda x: x['sentence_id'])
    return unique_pairs


# ------------------- 生成依存树 -------------------
def generate_dependency_tree(dep_df, sentence_id):
    if not GRAPHVIZ_AVAILABLE:
        return None

    try:
        sent_deps = dep_df[dep_df["sentence_id"] == sentence_id].copy()

        if sent_deps.empty:
            return None

        dot = graphviz.Digraph(comment='Dependency Tree')
        dot.attr(rankdir='TB')
        dot.attr('node', shape='box', style='rounded,filled', fillcolor='lightblue', fontname='Arial')
        dot.attr('edge', fontname='Arial')

        node_map = {}

        for idx, row in sent_deps.iterrows():
            head_text = str(row.get('head_text', ''))
            dep_text = str(row.get('dependent_text', ''))
            head_id = str(row.get('head_id', '0'))
            dep_id = str(row.get('dependent_id', ''))
            deprel = str(row.get('deprel', ''))

            if dep_text in ['', 'nan'] or head_text in ['', 'nan']:
                continue

            node_id = f"{dep_id}_{dep_text}"
            if node_id not in node_map:
                dot.node(node_id, dep_text)
                node_map[node_id] = dep_text

            if deprel.upper() == 'ROOT' or head_id == '0':
                dot.node(node_id, dep_text, fillcolor='lightgreen')
            else:
                head_node_id = f"{head_id}_{head_text}"
                if head_node_id not in node_map:
                    dot.node(head_node_id, head_text)
                    node_map[head_node_id] = head_text

                dot.edge(head_node_id, node_id, label=deprel)

        return dot
    except Exception as e:
        st.error(f"生成依存树时出错: {e}")
        return None


# ------------------- 持久化与用户管理 -------------------
def get_progress_file():
    """
    Return the current user's progress file.
    If no user is logged in, fall back to a shared progress.json.
    """
    username = st.session_state.get("current_user")
    if username:
        return BASE_DIR / f"progress_{username}.json"
    return PROGRESS_FILE


def load_progress():
    """从磁盘加载当前用户的进度到 session state。"""
    progress_path = get_progress_file()
    if progress_path.exists():
        try:
            with open(progress_path, 'r') as f:
                data = json.load(f)
                for k, v in data.items():
                    if k == 'global_progress':
                        gp = v
                        gp['global_counter'] = Counter(gp['global_counter'])
                        st.session_state[k] = gp
                    elif isinstance(v, dict) and 'cumulative_counter' in v:
                        book_prog = v
                        book_prog['cumulative_counter'] = Counter(book_prog['cumulative_counter'])
                        st.session_state[k] = book_prog
                    else:
                        st.session_state[k] = v
        except Exception:
            pass


def save_progress():
    """将当前 session state 中的进度保存到磁盘。"""
    # 只有登录状态下才保存，防止匿名数据污染用户文件
    if not st.session_state.get("current_user"):
        return
    progress_path = get_progress_file()
    try:
        data = {}
        if 'global_progress' in st.session_state:
            gp = st.session_state['global_progress']
            data['global_progress'] = {
                'completed_books': gp['completed_books'],
                'current_book_index': gp['current_book_index'],
                'global_counter': dict(gp['global_counter'])
            }
        for k in list(st.session_state.keys()):
            if k in BUILT_IN_BOOKS:
                if isinstance(st.session_state[k], dict) and 'cumulative_counter' in st.session_state[k]:
                    book_prog = st.session_state[k]
                    data[k] = {
                        'current_sentence': book_prog['current_sentence'],
                        'cumulative_counter': dict(book_prog['cumulative_counter'])
                    }
            elif k.startswith('cumul_sentence_') or k.startswith('view_sentence_'):
                data[k] = st.session_state[k]
            elif k in ['user_wordbook', 'daily_plan', 'cumulative_mode']:
                data[k] = st.session_state[k]
        with open(progress_path, 'w') as f:
            json.dump(data, f)
    except Exception:
        pass


def clear_session_progress():
    """
    切换用户时清除所有进度相关的 session state，
    防止旧用户的数据残留到新用户会话中。

    修复说明：
    1. 新增清除 '_progress_loaded'，确保 rerun 后会重新从磁盘加载新用户数据。
    2. 不在此函数内修改 current_user，调用方负责在 clear 之后再赋值。
    """
    fixed_keys = {
        'user_wordbook', 'daily_plan', 'cumulative_mode',
        'global_progress', 'simplify_mode', 'clicked_word',
        '_progress_loaded',   # ← 新增：确保新用户会重新 load
    }
    dynamic_prefixes = (
        'cumul_sentence_', 'view_sentence_', 'slider_', '_pending_',
        'show_dep_', 'show_tree_'
    )

    keys_to_delete = []
    for k in list(st.session_state.keys()):
        if k in fixed_keys:
            keys_to_delete.append(k)
        elif k in BUILT_IN_BOOKS:
            keys_to_delete.append(k)
        elif any(k.startswith(prefix) for prefix in dynamic_prefixes):
            keys_to_delete.append(k)

    for k in keys_to_delete:
        del st.session_state[k]


def reset_default_session_state():
    """登入/登出后重置为默认值，再由 load_progress 覆盖已保存的数据。"""
    if "cumulative_mode" not in st.session_state:
        st.session_state.cumulative_mode = False
    if "user_wordbook" not in st.session_state:
        st.session_state.user_wordbook = []
    if "daily_plan" not in st.session_state:
        st.session_state.daily_plan = {"sentences_per_day": 10, "current_day_progress": 0}
    if "clicked_word" not in st.session_state:
        st.session_state.clicked_word = None
    if "simplify_mode" not in st.session_state:
        st.session_state.simplify_mode = None


def load_users():
    if USERS_FILE.exists():
        try:
            with open(USERS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}
    return {}


def save_users(users):
    try:
        with open(USERS_FILE, "w", encoding="utf-8") as f:
            json.dump(users, f)
    except Exception:
        pass


def hash_password(password: str, salt: str) -> str:
    return hashlib.sha256((salt + password).encode("utf-8")).hexdigest()


def register_user(username: str, password: str) -> bool:
    username = username.strip()
    if not username or not password:
        return False
    users = load_users()
    if username in users:
        return False
    salt = secrets.token_hex(16)
    users[username] = {
        "salt": salt,
        "password_hash": hash_password(password, salt)
    }
    save_users(users)
    return True


def authenticate_user(username: str, password: str) -> bool:
    username = username.strip()
    users = load_users()
    user = users.get(username)
    if not user:
        return False
    salt = user.get("salt", "")
    expected_hash = user.get("password_hash", "")
    return hash_password(password, salt) == expected_hash


# ------------------- 初始化状态 -------------------
# ------------------- TTS 语音合成 -------------------
def text_to_speech_bytes(text: str, voice: str = "en-US-JennyNeural") -> bytes | None:
    """用 edge-tts 将文本合成为 mp3，返回字节流；失败则返回 None。"""
    async def _run():
        communicate = edge_tts.Communicate(text, voice)
        with tempfile.NamedTemporaryFile(delete=False, suffix=".mp3") as f:
            tmp_path = f.name
        await communicate.save(tmp_path)
        with open(tmp_path, "rb") as f:
            return f.read()
    try:
        return asyncio.run(_run())
    except Exception:
        return None

# current_user 必须最先初始化，其他状态依赖它
if "current_user" not in st.session_state:
    st.session_state.current_user = None

# 第一次运行时设置默认值
reset_default_session_state()

# 初始化 global_progress（仅累计模式下需要）
if st.session_state.get("cumulative_mode"):
    if "global_progress" not in st.session_state:
        st.session_state.global_progress = {
            "completed_books": [],
            "current_book_index": 0,
            "global_counter": Counter()
        }

# 首次加载（已登录用户）的进度
if st.session_state.current_user:
    # 仅在第一次（session 刚创建）时加载，避免每次 rerun 都重置
    if "_progress_loaded" not in st.session_state:
        load_progress()
        st.session_state["_progress_loaded"] = True


# ------------------- 词典API -------------------
def get_word_info(word):
    try:
        response = requests.get(f"https://api.dictionaryapi.dev/api/v2/entries/en/{word}", timeout=5)
        if response.status_code == 200:
            data = response.json()[0]
            phonetics = data.get('phonetics', [{}])[0]
            phonetic = phonetics.get('text', '')
            audio_url = phonetics.get('audio', '')
            meanings = data.get('meanings', [])
            definitions = []
            for meaning in meanings:
                for defn in meaning.get('definitions', []):
                    definitions.append(defn.get('definition', ''))
            explanation = "; ".join(definitions[:3])
            return phonetic, audio_url, explanation
        else:
            return '', '', 'No definition found'
    except:
        return '', '', 'Error fetching definition'


# ------------------- 生成带交互的句子HTML -------------------
def generate_interactive_sentence_html(words, dep_map_by_position, dep_roles_by_position,
                                       sentence_id, core_lemmas, modifier_lemmas,
                                       book_name, display_sentence, simplify_mode=None):
    def should_show_word(word_idx):
        if simplify_mode is None:
            return True

        as_dependent_rels = set()
        if word_idx in dep_roles_by_position:
            as_dependent_rels = dep_roles_by_position[word_idx]['as_dependent']

        if simplify_mode == 'svo_only':
            core_rels = {'nsubj', 'nsubj:pass', 'obj', 'iobj', 'root', 'ROOT',
                         'csubj', 'csubj:pass', 'ccomp', 'xcomp'}
            if any(rel in core_rels for rel in as_dependent_rels) or len(as_dependent_rels) == 0:
                return True
            return False

        elif simplify_mode == 'no_amod':
            return 'amod' not in as_dependent_rels

        elif simplify_mode == 'no_advmod':
            return 'advmod' not in as_dependent_rels and 'obl' not in as_dependent_rels

        elif simplify_mode == 'no_complement':
            complement_rels = {'ccomp', 'xcomp', 'advcl'}
            return not any(rel in complement_rels for rel in as_dependent_rels)

        return True

    html_parts = []
    word_data_json = []

    for idx, word_data in enumerate(words):
        word = word_data['display']
        lemma = word_data.get('lemma')
        freq = word_data.get('freq', 0)

        if lemma and not should_show_word(idx):
            continue

        if lemma:
            color = get_color(freq)
            size_str, font = get_font_style_by_frequency(freq)
            size = int(size_str.replace('px', ''))

            styles = [f"color:{color}", f"font-size:{size}px", f"font-family:{font}", "cursor:pointer"]
            if lemma in core_lemmas:
                styles.append("font-weight:bold")
            if lemma in modifier_lemmas:
                styles.append("font-style:italic")

            style_str = "; ".join(styles)

            word_escaped = html.escape(word)
            lemma_escaped = html.escape(lemma)

            dep_data = []
            if idx in dep_map_by_position:
                for related_idx, deprel, related_lemma in dep_map_by_position[idx]:
                    dep_data.append({
                        'position': related_idx,
                        'lemma': related_lemma,
                        'deprel': deprel
                    })

            word_data_json.append({
                'idx': idx,
                'lemma': lemma,
                'word': word,
                'deps': dep_data
            })

            html_parts.append(
                f'<span class="word" data-idx="{idx}" data-lemma="{lemma_escaped}" '
                f'style="{style_str}">{word_escaped}</span> '
            )
        else:
            html_parts.append(f'<span style="color:#000000; font-size:12px">{html.escape(word)}</span> ')

    sentence_html = ''.join(html_parts)

    full_html = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <style>
            body {{
                margin: 0;
                padding: 20px;
                font-family: Arial, sans-serif;
                background: #F5E6C8;
            }}
            .sentence-container {{
                font-size: 28px;
                line-height: 2.5;
                padding: 20px;
                background: #F5E6C8;
                border-radius: 10px;
                cursor: default;
                user-select: none;
            }}
            .word {{
                transition: background-color 0.2s;
                padding: 2px 4px;
                border-radius: 3px;
            }}
            .word:hover {{
                background-color: #f0f0f0;
            }}
            .dep-relation {{
                margin-top: 15px;
                padding: 15px;
                background: #e3f2fd;
                border-radius: 8px;
                font-size: 16px;
                border-left: 4px solid #2196F3;
                animation: fadeIn 0.3s;
            }}
            @keyframes fadeIn {{
                from {{ opacity: 0; transform: translateY(-10px); }}
                to {{ opacity: 1; transform: translateY(0); }}
            }}
            .relation-title {{
                font-weight: bold;
                margin-bottom: 8px;
                color: #1976D2;
            }}
            .relation-item {{
                margin: 5px 0;
                padding: 5px;
                background: white;
                border-radius: 4px;
            }}
            .flash-green {{
                animation: flashGreen 0.5s;
            }}
            @keyframes flashGreen {{
                0%, 100% {{ background-color: transparent; }}
                50% {{ background-color: #90EE90; }}
            }}
        </style>
    </head>
    <body>
        <div class="sentence-container" id="sentenceContainer">
            {sentence_html}
        </div>
        <div id="depRelationContainer"></div>

        <script>
            const wordData = {json.dumps(word_data_json)};

            const idxToElement = new Map();
            wordData.forEach(data => {{
                const element = document.querySelector(`.word[data-idx="${{data.idx}}"]`);
                if (element) {{
                    idxToElement.set(data.idx, element);
                }}
            }});

            document.querySelectorAll('.word').forEach(element => {{
                element.addEventListener('click', function(e) {{
                    e.stopPropagation();
                    handleSingleClick(this);
                }});
            }});

            function handleSingleClick(element) {{
                clearHighlights();

                const idx = parseInt(element.dataset.idx);
                const data = wordData.find(d => d.idx === idx);

                if (!data || data.deps.length === 0) return;


                
                element.style.outline = '3px solid #39FF14';

                data.deps.forEach(dep => {{
                    const relatedElement = idxToElement.get(dep.position);
                    if (relatedElement) {{
                        
                        
                        
                        relatedElement.style.outline = '3px dashed #39FF14';
                    }}
                }});

                showDependencies(data);
            }}

            function showDependencies(data) {{
                const container = document.getElementById('depRelationContainer');
                container.innerHTML = '';

                const relationDiv = document.createElement('div');
                relationDiv.className = 'dep-relation';

                let html = '<div class="relation-title">Dependency relations:</div>';
                data.deps.forEach(dep => {{
                    const deprelLabels = {json.dumps(DEPREL_LABELS)};
                    const relNote = deprelLabels[dep.deprel] || dep.deprel;
                    const relDisplay = relNote !== dep.deprel ? relNote + ' (' + dep.deprel + ')' : dep.deprel;
                    html += `<div class="relation-item"><strong>${{data.lemma}}</strong> ──${{relDisplay}}──&gt; <strong>${{dep.lemma}}</strong></div>`;
                    
                }});

                relationDiv.innerHTML = html;
                container.appendChild(relationDiv);
            }}

            function clearHighlights() {{
                document.querySelectorAll('.word').forEach(w => {{
                
                    w.style.outline = '';
                    
                }});
                document.getElementById('depRelationContainer').innerHTML = '';
            }}

            document.addEventListener('click', function(e) {{
                if (!e.target.classList.contains('word')) {{
                    clearHighlights();
                }}
            }});
        </script>
    </body>
    </html>
    """

    return full_html


# =============== 主界面 ===============
st.sidebar.title("📚 Reading mode & selection")

cumulative_mode = st.sidebar.checkbox("Enable cumulative reading mode", value=st.session_state.cumulative_mode)

if cumulative_mode != st.session_state.cumulative_mode:
    st.session_state.cumulative_mode = cumulative_mode
    if cumulative_mode:
        st.session_state.global_progress = {
            "completed_books": [],
            "current_book_index": 0,
            "global_counter": Counter()
        }
    save_progress()
    st.rerun()

st.sidebar.markdown("---")

# ------------------- 用户认证（登录 / 注册） -------------------
st.sidebar.markdown("### 👤 Account")

if st.session_state.current_user:
    st.sidebar.success(f"Logged in as {st.session_state.current_user}")
    if st.sidebar.button("Log out"):
        # ① 先保存当前用户数据（此时 current_user 还是旧用户）
        save_progress()
        # ② 切换用户身份为 None
        st.session_state.current_user = None
        # ③ 清除所有进度相关 session state（包含 _progress_loaded）
        clear_session_progress()
        # ④ 恢复默认值
        reset_default_session_state()
        st.rerun()
else:
    auth_mode = st.sidebar.radio("Authentication mode", ["Log in", "Register"])
    username_input = st.sidebar.text_input("Username")
    password_input = st.sidebar.text_input("Password", type="password")

    if auth_mode == "Register":
        password_confirm = st.sidebar.text_input("Confirm password", type="password")
        if st.sidebar.button("Register"):
            if not username_input or not password_input:
                st.sidebar.error("Username and password cannot be empty.")
            elif password_input != password_confirm:
                st.sidebar.error("Passwords do not match.")
            else:
                ok = register_user(username_input, password_input)
                if not ok:
                    st.sidebar.error("Username already exists or invalid.")
                else:
                    st.sidebar.success("Registration successful. You are now logged in.")
                    # ① 如果之前有登录用户，先保存他的数据
                    if st.session_state.get("current_user"):
                        save_progress()
                    # ② 清除旧 session 数据（包含 _progress_loaded）
                    clear_session_progress()
                    # ③ 设置新用户身份
                    st.session_state.current_user = username_input.strip()
                    # ④ 恢复默认值，再加载新用户进度（新用户无文件，load_progress 无副作用）
                    reset_default_session_state()
                    load_progress()
                    st.session_state["_progress_loaded"] = True
                    st.rerun()
    else:
        if st.sidebar.button("Log in"):
            if authenticate_user(username_input, password_input):
                st.sidebar.success("Login successful.")
                # ① 如果之前有登录用户，先保存他的数据
                if st.session_state.get("current_user"):
                    save_progress()
                # ② 清除旧 session 数据（包含 _progress_loaded）
                clear_session_progress()
                # ③ 设置新用户身份
                st.session_state.current_user = username_input.strip()
                # ④ 恢复默认值，再加载新用户进度
                reset_default_session_state()
                load_progress()
                st.session_state["_progress_loaded"] = True
                st.rerun()
            else:
                st.sidebar.error("Invalid username or password.")

if not st.session_state.current_user:
    st.info("Please log in or register in the sidebar to start reading.")
    st.stop()

if cumulative_mode:
    st.sidebar.subheader("Cumulative reading mode")
    progress = st.session_state.global_progress
    unlocked_books = BUILT_IN_BOOKS[:progress["current_book_index"] + 1]
    completed_set = set(progress["completed_books"])

    st.sidebar.write(f"Completed {len(progress['completed_books'])} / {len(BUILT_IN_BOOKS)} books")
    available_choices = ["None"] + unlocked_books
    book_choice = st.sidebar.radio("Current novel", available_choices)
else:
    st.sidebar.subheader("Single-book mode")
    book_choice = st.sidebar.radio("Built-in novels", ["None"] + BUILT_IN_BOOKS)

st.sidebar.markdown("### 📅 Daily plan")
daily_sentences = st.sidebar.number_input("Sentences per day", min_value=1,
                                          value=st.session_state.daily_plan["sentences_per_day"])
if daily_sentences != st.session_state.daily_plan["sentences_per_day"]:
    st.session_state.daily_plan["sentences_per_day"] = daily_sentences
    st.session_state.daily_plan["current_day_progress"] = 0
    save_progress()

if book_choice == "None":
    st.info("👋 Please choose a novel to start reading.")
    st.stop()

book_name = book_choice
data = load_book_data(book_name)

tab1, tab2, tab3 = st.tabs(["📖 Reading", "📚 Wordbook", "🌌 Vocabulary universe"])

with tab1:
    st.title(f"📖 Reading: {book_name}")

    with st.expander("🌈 Color & frequency (historical snapshot)", expanded=False):
        st.markdown("""
        **Color represents the historical frequency when you reached this sentence**:

        - **Dark purple** (#8B00FF): frequency = 1 ✨
        - **Sky blue** (#00BFFF): frequency 2–10 💧
        - **Dark orange** (#FF8C00): frequency 11–20 🔥
        - **Dark green** (#023020): frequency >20 👑

        **Key ideas**:
        - When revisiting, you see the frequency **at the time you read that sentence**
        - Colors do not change later (historical snapshot)
        - In the same sentence, the 1st occurrence is purple, 2nd is blue, etc.

        **Interaction**:
        - **Single-click** a word: highlight words that have dependency relations with it
        - **Add to wordbook**: use the word buttons below the sentence
        """)

    if cumulative_mode:
        progress = st.session_state.global_progress
        cumulative_counter = progress["global_counter"]
        book_progress_key = f"cumul_sentence_{book_name}"
        current_sentence = st.session_state.get(book_progress_key, 0)
    else:
        if book_name not in st.session_state:
            st.session_state[book_name] = {"current_sentence": 0, "cumulative_counter": Counter()}
        book_progress = st.session_state[book_name]
        cumulative_counter = book_progress["cumulative_counter"]
        current_sentence = book_progress["current_sentence"]

    sentences_df = data["sentences"]
    all_sentence_lemmas = data["all_sentence_lemmas"]
    global_freq_dict = data["global_freq_dict"]
    total_sentences = len(sentences_df)

    max_view = current_sentence if current_sentence < total_sentences else total_sentences - 1

    # view_key：记录当前查看位置（纯数据 key，不绑定 widget）
    view_key = f"view_sentence_{book_name}"
    if view_key not in st.session_state:
        st.session_state[view_key] = 0

    # slider_key：专门绑定滑块 widget（Streamlit 锁定后不可在代码中修改）
    slider_key = f"slider_{book_name}"

    # _pending_key：按钮点击时写入期望跳转的句子编号，
    # 在滑块渲染之前读取并同步给 slider_key，避免"widget 已锁定"报错。
    pending_key = f"_pending_{book_name}"

    # ── 在渲染任何 widget 之前处理按钮的跳转意图 ──
    if pending_key in st.session_state:
        target = st.session_state.pop(pending_key)
        target = max(0, min(target, max_view))
        st.session_state[view_key] = target
        st.session_state[slider_key] = target

    # 确保 view_key / slider_key 不超出可查看范围
    if st.session_state[view_key] > max_view:
        st.session_state[view_key] = max_view
    if st.session_state.get(slider_key, 0) > max_view:
        st.session_state[slider_key] = st.session_state[view_key]

    if max_view == 0:
        display_sentence = 0
    else:
        # 初始化 slider_key（首次）
        if slider_key not in st.session_state:
            st.session_state[slider_key] = st.session_state[view_key]

        # 渲染滑块；用户拖动时 Streamlit 自动更新 slider_key
        st.slider(
            "📖 Sentence selector",
            min_value=0,
            max_value=max_view,
            step=1,
            key=slider_key,
        )
        # 让 view_key 跟随滑块（用户拖动场景）
        st.session_state[view_key] = st.session_state[slider_key]
        display_sentence = st.session_state[view_key]
        save_progress()

    if current_sentence < total_sentences:
        if st.button("⏭️ Next sentence", use_container_width=True):
            next_lemmas = all_sentence_lemmas[current_sentence]
            cumulative_counter.update(next_lemmas)
            if cumulative_mode:
                progress["global_counter"].update(next_lemmas)
                st.session_state[book_progress_key] = current_sentence + 1
                if current_sentence + 1 >= total_sentences:
                    if book_name not in progress["completed_books"]:
                        progress["completed_books"].append(book_name)
                    if book_name in BUILT_IN_BOOKS:
                        progress["current_book_index"] = BUILT_IN_BOOKS.index(book_name) + 1
            else:
                book_progress["current_sentence"] += 1
            # 写入 pending_key，下一次 rerun 开头读取并同步给滑块
            st.session_state[pending_key] = min(current_sentence + 1, max_view)
            st.session_state.daily_plan["current_day_progress"] += 1
            save_progress()
            st.rerun()

    row = sentences_df.iloc[display_sentence]
    sentence_text = row.get("tokenized_sentence", "")
    sentence_id = row["sentence_id"]

    # 句子简化控制
    st.markdown("### ✂️ Sentence simplification")

    if 'simplify_mode' not in st.session_state:
        st.session_state.simplify_mode = None

    simplify_cols = st.columns(5)

    with simplify_cols[0]:
        if st.button("🔄 Show full sentence", use_container_width=True, key=f"show_all_{book_name}_{display_sentence}"):
            st.session_state.simplify_mode = None
            st.rerun()
    with simplify_cols[1]:
        if st.button("📌 Subject–Verb–Object only", use_container_width=True, key=f"svo_{book_name}_{display_sentence}"):
            st.session_state.simplify_mode = 'svo_only'
            st.rerun()
    with simplify_cols[2]:
        if st.button("🚫 Remove attributive modifiers", use_container_width=True, key=f"no_amod_{book_name}_{display_sentence}"):
            st.session_state.simplify_mode = 'no_amod'
            st.rerun()
    with simplify_cols[3]:
        if st.button("🚫 Remove adverbials", use_container_width=True, key=f"no_advmod_{book_name}_{display_sentence}"):
            st.session_state.simplify_mode = 'no_advmod'
            st.rerun()
    with simplify_cols[4]:
        if st.button("🚫 Remove complements", use_container_width=True, key=f"no_comp_{book_name}_{display_sentence}"):
            st.session_state.simplify_mode = 'no_complement'
            st.rerun()

    mode_names = {
        None: "Full sentence",
        'svo_only': "Subject–Verb–Object only",
        'no_amod': "Remove attributive modifiers",
        'no_advmod': "Remove adverbials",
        'no_complement': "Remove complements"
    }
    st.caption(f"Current mode: {mode_names.get(st.session_state.simplify_mode, 'Full sentence')}")

    sent_dep_df = data["dep_df"]
    dep_index   = data["dep_index"]
    core_lemmas = set()
    modifier_lemmas = set()

    dep_roles_by_position = {}
    dep_map_by_position = {}

    # ── 用预建索引直接取当前句的依存行，O(1) 而非全表扫描 ──
    sent_deps_df = dep_index.get(sentence_id, pd.DataFrame())

    if not sent_deps_df.empty:
        core_rels = {
            "nsubj", "nsubj:pass", "obj", "iobj",
            "csubj", "csubj:pass", "ccomp", "xcomp",
            "root", "ROOT"
        }
        modifier_rels = {"amod", "advmod", "obl", "nmod", "appos", "acl", "acl:relcl"}

        for _, r in sent_deps_df.iterrows():
            rel = str(r.get("deprel", "")).lower()
            dep_text = str(r.get("dependent_text", ""))
            head_text = str(r.get("head_text", ""))
            dep_id = r.get("dependent_id", "")
            head_id = r.get("head_id", "")

            dep_clean = re.sub(r"[^\w]", "", dep_text.lower())
            head_clean = re.sub(r"[^\w]", "", head_text.lower())

            if not dep_clean and not head_clean:
                continue

            dep_lemma = cached_lemmatize(dep_clean) if dep_clean else ""
            head_lemma = cached_lemmatize(head_clean) if head_clean else ""

            try:
                dep_pos = int(dep_id) - 1
                head_pos = int(head_id) - 1 if head_id != '0' else -1

                if dep_pos >= 0:
                    if dep_pos not in dep_roles_by_position:
                        dep_roles_by_position[dep_pos] = {'as_dependent': set(), 'as_head': set()}
                    if dep_pos not in dep_map_by_position:
                        dep_map_by_position[dep_pos] = []

                if head_pos >= 0:
                    if head_pos not in dep_roles_by_position:
                        dep_roles_by_position[head_pos] = {'as_dependent': set(), 'as_head': set()}
                    if head_pos not in dep_map_by_position:
                        dep_map_by_position[head_pos] = []

                if dep_pos >= 0:
                    dep_roles_by_position[dep_pos]['as_dependent'].add(rel)
                if head_pos >= 0:
                    dep_roles_by_position[head_pos]['as_head'].add(rel)

                if dep_pos >= 0 and head_pos >= 0:
                    dep_map_by_position[dep_pos].append((head_pos, rel, head_lemma))
                    dep_map_by_position[head_pos].append((dep_pos, rel, dep_lemma))

            except (ValueError, TypeError):
                pass

            if rel in core_rels:
                if dep_lemma:
                    core_lemmas.add(dep_lemma)
                if head_lemma:
                    core_lemmas.add(head_lemma)
            elif rel in modifier_rels:
                if dep_lemma:
                    modifier_lemmas.add(dep_lemma)

    sentence_tokens = []
    if sentence_text:
        words = sentence_text.split()

        # ── 用差分前缀和 O(n) 替代原来每次重算的 O(n) 循环 ──
        sentence_deltas = data["sentence_deltas"]
        freq_before_this_sentence = sum(
            sentence_deltas[:display_sentence], Counter()
        )

        running_counter = Counter(freq_before_this_sentence)

        for idx, word in enumerate(words):
            cleaned = re.sub(r'[^\w-]', '', word.lower())
            lemma = None
            freq = 0

            if cleaned and cleaned.isalpha() and len(cleaned) > 1:
                lemma = cached_lemmatize(cleaned)
                running_counter[lemma] += 1
                freq = running_counter[lemma]

            sentence_tokens.append({
                "display": word,
                "lemma": lemma,
                "freq": freq
            })

    interactive_html = generate_interactive_sentence_html(
        sentence_tokens,
        dep_map_by_position,
        dep_roles_by_position,
        sentence_id,
        core_lemmas,
        modifier_lemmas,
        book_name,
        display_sentence,
        st.session_state.get('simplify_mode')
    )

    components.html(interactive_html, height=400, scrolling=True)
    # ── 语音播放 ──
    tts_col1, tts_col2 = st.columns([1, 4])
    with tts_col1:
        if st.button("🔊 Play sentence", key=f"tts_{book_name}_{display_sentence}"):
            if sentence_text:
                audio_bytes = text_to_speech_bytes(sentence_text)
                if audio_bytes:
                    st.session_state[f"tts_audio_{book_name}_{display_sentence}"] = audio_bytes
                else:
                    st.warning("Speech synthesis failed.")
    tts_audio_key = f"tts_audio_{book_name}_{display_sentence}"
    if tts_audio_key in st.session_state:
        st.audio(st.session_state[tts_audio_key], format="audio/mp3")
################################################################################
    if sentence_tokens:
        st.markdown("---")
        st.markdown("**💾 Add to wordbook (click words):**")

        valid_tokens = [t for t in sentence_tokens if t.get('lemma')]

        if valid_tokens:
            cols_per_row = 6
            for i in range(0, len(valid_tokens), cols_per_row):
                cols = st.columns(cols_per_row)
                for j, token in enumerate(valid_tokens[i:i + cols_per_row]):
                    with cols[j]:
                        btn_key = f"add_{book_name}_{display_sentence}_{token['lemma']}_{i + j}"
                        if st.button(
                                token['display'],
                                key=btn_key,
                                use_container_width=True,
                                help=f"Click to add '{token['display']}' to the wordbook"
                        ):
                            entry = {
                                "book": book_name,
                                "lemma": token['lemma'],
                                "word": token['display']
                            }
                            if entry not in st.session_state.user_wordbook:
                                st.session_state.user_wordbook.append(entry)
                                st.success(f"✅ Added '{token['display']}' to the wordbook.")
                                save_progress()
                            else:
                                st.info(f"'{token['display']}' is already in the wordbook.")

    st.caption(f"Viewing sentence {display_sentence + 1} / {total_sentences} | Progressed to sentence {current_sentence + 1}")


########################################################################################
    query_word = st.text_input("🔍 Enter a word to look up")
    if query_word:
        lemma_query = lemmatizer.lemmatize(query_word.lower())

        phonetic, audio_url, explanation = get_word_info(query_word)
        st.write(f"**Word:** {query_word}")
        st.write(f"**Lemma:** {lemma_query}")
        st.write(f"**Phonetic:** {phonetic}")
        if audio_url:
            try:
                st.audio(BytesIO(requests.get(audio_url, timeout=5).content), format="audio/mp3")
            except:
                pass
        st.write(f"**Definition:** {explanation}")

    dep_key = f"show_dep_{book_name}_{display_sentence}"
    if dep_key not in st.session_state:
        st.session_state[dep_key] = False

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("🔍 Show dependency analysis" if not st.session_state[dep_key] else "❌ Hide dependency analysis"):
            st.session_state[dep_key] = not st.session_state[dep_key]
            st.rerun()

    if st.session_state[dep_key]:
        sent_deps = data["dep_df"][data["dep_df"]["sentence_id"] == sentence_id]
        if not sent_deps.empty:
            st.subheader("Dependency relations")
            for _, r in sent_deps.iterrows():
                head = str(r.get('head_text', ''))
                dep = str(r.get('dependent_text', ''))
                rel = str(r.get('deprel', ''))
                if head not in ['', 'nan'] and dep not in ['', 'nan']:
                    st.text(f"{head} ──{rel}──> {dep}")
        else:
            st.info("No dependency records for this sentence.")

    if GRAPHVIZ_AVAILABLE:
        tree_key = f"show_tree_{book_name}_{display_sentence}"
        if tree_key not in st.session_state:
            st.session_state[tree_key] = False

        with col2:
            if st.button("🌳 Show dependency tree" if not st.session_state[tree_key] else "❌ Hide dependency tree"):
                st.session_state[tree_key] = not st.session_state[tree_key]
                st.rerun()

        if st.session_state[tree_key]:
            tree = generate_dependency_tree(data["dep_df"], sentence_id)
            if tree:
                st.graphviz_chart(tree)
            else:
                st.info("Unable to generate dependency tree.")

    col_prev, col_next = st.columns([1, 1])
    with col_prev:
        if display_sentence > 0:
            if st.button("← Previous sentence", key=f"prev_btn_{book_name}_{display_sentence}"):
                st.session_state[pending_key] = display_sentence - 1
                save_progress()
                st.rerun()

    with col_next:
        if display_sentence < max_view:
            if st.button("Next sentence →", key=f"next_btn_{book_name}_{display_sentence}"):
                st.session_state[pending_key] = display_sentence + 1
                save_progress()
                st.rerun()

with tab2:
    st.title("📚 Wordbook")

    wordbook = st.session_state.user_wordbook

    normalized_wordbook = []
    for item in wordbook:
        if isinstance(item, dict):
            normalized_wordbook.append(item)
        elif isinstance(item, str):
            normalized_wordbook.append(
                {"book": "unknown", "lemma": item.lower(), "word": item}
            )
    if normalized_wordbook != wordbook:
        st.session_state.user_wordbook = normalized_wordbook
    wordbook = normalized_wordbook

    if not wordbook:
        st.info("The wordbook is empty. Add words from the reading tab.")
    else:
        current_book_words = [w for w in wordbook if w.get("book") == book_name]
        if not current_book_words:
            st.info("This book has no saved words yet. Add words from the reading tab.")
        else:
            lemma_to_word = {}
            for w in current_book_words:
                lemma = w.get("lemma")
                if lemma and lemma not in lemma_to_word:
                    lemma_to_word[lemma] = w.get("word", lemma)

            sorted_lemmas = sorted(lemma_to_word.keys())
            selected_lemma = st.selectbox(
                "Select a word to view example sentences in this book",
                options=sorted_lemmas,
                format_func=lambda x: f"{lemma_to_word.get(x, x)} ({x})"
            )

            if selected_lemma:
                sentences_for_word = get_word_sentences(
                    selected_lemma, sentences_df, all_sentence_lemmas
                )

                if not sentences_for_word:
                    st.info("No sentences containing this word were found in this book.")
                else:
                    sentences_for_word.sort(key=lambda x: x["sentence_id"])
                    total_hits = len(sentences_for_word)

                    st.markdown(f"**Found {total_hits} sentences containing this word (ordered by sentence index)**")

                    top_n = 5
                    for item in sentences_for_word[:top_n]:
                        st.write(f"{item['sentence_id']}: {item['text']}")

                    if total_hits > top_n:
                        with st.expander(f"Show remaining {total_hits - top_n} sentences"):
                            for item in sentences_for_word[top_n:]:
                                st.write(f"{item['sentence_id']}: {item['text']}")

                st.markdown("---")
                st.subheader("🔗 Word dependencies and definitions")

                default_query = lemma_to_word.get(selected_lemma, selected_lemma)
                dep_query = st.text_input(
                    "Enter a word (view dependencies & definition)",
                    value=default_query,
                    key=f"dep_query_input_{book_name}"
                )
                if st.button("🔍 Show word dependencies", key=f"dep_btn_{book_name}"):
                    if not dep_query:
                        st.warning("Please enter a word first.")
                    else:
                        lemma_dep = lemmatizer.lemmatize(dep_query.lower())

                        # phonetic, audio_url, explanation = get_word_info(dep_query)
                        # st.write(f"**Word:** {dep_query}")
                        # st.write(f"**Lemma:** {lemma_dep}")
                        # st.write(f"**Phonetic:** {phonetic}")
                        # if audio_url:
                        #     try:
                        #         st.audio(BytesIO(requests.get(audio_url, timeout=5).content), format="audio/mp3")
                        #     except:
                        #         pass
                        # st.write(f"**Definition:** {explanation}")


                        # dep_pairs = get_word_dependencies(lemma_dep, data["dep_df"])
                        # if dep_pairs:
                        #     st.subheader("Dependency relations (ordered by sentence index)")
                        #     for pair in dep_pairs[:200]:
                        #         st.text(f"[句子 {pair['sentence_id']}] {pair['relation']}")
                        #     if len(dep_pairs) > 200:
                        #         st.info(f"{len(dep_pairs)} relations in total, showing the first 200.")
                        # else:
                        #     st.info("No dependency relations found for this word.")
                        dep_pairs = get_word_dependencies(lemma_dep, data["dep_df"])
                        if dep_pairs:
                            st.subheader("Dependency relations (ordered by sentence index)")

                            # 当前阅读句子的 sentence_id，用于高亮
                            current_sid = sentences_df.iloc[display_sentence]["sentence_id"]

                            shown = 0
                            for pair in dep_pairs:
                                if shown >= 200:
                                    break
                                sid = pair['sentence_id']
                                relation = pair['relation']
                                if sid == current_sid:
                                    # 当前句子：加 ★ 标记并用 markdown 高亮
                                    st.markdown(
                                        f"**★ \[句子 {sid}\]** &nbsp; `{relation}`",
                                        unsafe_allow_html=False
                                    )
                                else:
                                    st.text(f"[句子 {sid}] {relation}")
                                shown += 1

                            if len(dep_pairs) > 200:
                                st.info(f"{len(dep_pairs)} relations in total, showing the first 200.")
                        else:
                            st.info("No dependency relations found for this word.")

with tab3:
    st.title("🌌 Vocabulary universe")

    counter_for_stats = progress["global_counter"] if cumulative_mode else cumulative_counter

    standard_wordlists = load_standard_wordlists()

    wordlist_df = data["wordlist"]

    if not wordlist_df.empty and standard_wordlists:
        text_lemmas = set()
        if 'lemma' in wordlist_df.columns:
            text_lemmas = set(wordlist_df['lemma'].str.lower())

        st.markdown("### 📈 Vocabulary coverage statistics")
        stats_cols = st.columns(4)
        for idx, (level, wl_df) in enumerate(sorted(standard_wordlists.items())):
            if idx >= 4:
                break
            wordlist_lemmas = set(wl_df['lemma'].str.lower())
            intersection = text_lemmas & wordlist_lemmas
            coverage = len(intersection) / len(wordlist_lemmas) * 100 if len(wordlist_lemmas) > 0 else 0

            with stats_cols[idx]:
                st.metric(
                    label=f"COCA {(level - 1) * 5000 + 1}-{level * 5000}",
                    value=f"{len(intersection)} words",
                    delta=f"{coverage:.1f}% coverage"
                )

    col1, col2 = st.columns([1, 1])
    with col1:
        if st.button("📊 Show dependency relation types"):
            if not data["dep_df"].empty:
                counts = data["dep_df"]["deprel"].value_counts()
                fig = px.bar(counts, x=counts.index, y=counts.values, title="Dependency relation types")
                st.plotly_chart(fig, use_container_width=True)

    with col2:
        if st.button("📋 Show word frequency table"):
            if not data["wordlist"].empty:
                display_df = data["wordlist"].copy()
                if 'wordlist' not in display_df.columns:
                    display_df['wordlist'] = 'Unknown'

                def get_wordlist_order(wl):
                    if 'COCA' in str(wl):
                        try:
                            num = int(str(wl).split('_')[1])
                            return num
                        except:
                            pass
                    return 999999

                display_df['_sort_key'] = display_df['wordlist'].apply(get_wordlist_order)
                display_df = display_df.sort_values(['_sort_key', 'frequency'], ascending=[True, False])
                display_df = display_df.drop('_sort_key', axis=1)

                st.dataframe(display_df[['lemma', 'frequency', 'wordlist']])