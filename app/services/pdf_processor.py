from __future__ import annotations
import io
import re
import string
import hashlib
try:
    import numpy as np
except Exception:
    np = None
from typing import List, Dict, Optional, Tuple

# Optional: better PDF extractor (strongly recommended)
try:
    import fitz  # PyMuPDF
    HAS_PYMUPDF = True
except Exception:
    HAS_PYMUPDF = False

# Fallback extractor
try:
    import PyPDF2
    HAS_PYPDF2 = True
except Exception:
    HAS_PYPDF2 = False

# Optional: TF-IDF (fallback implemented if not available)
try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics.pairwise import cosine_similarity
    HAS_SKLEARN = True
except Exception:
    HAS_SKLEARN = False


# -------------------- PDF TEXT EXTRACTION --------------------

def extract_text_from_pdf(filelike) -> str:
    """Extract text from PDF using PyMuPDF first, then PyPDF2 as fallback"""
    # 1) PyMuPDF
    if HAS_PYMUPDF:
        try:
            with fitz.open(stream=filelike.read(), filetype="pdf") as doc:
                text_parts = []
                for page in doc:
                    txt = page.get_text("text") or ""
                    text_parts.append(txt)
                raw = "\n".join(text_parts)
                if len(raw.strip()) >= 200:
                    return raw
        except Exception:
            pass
        finally:
            try:
                filelike.seek(0)
            except Exception:
                pass

    # 2) PyPDF2
    if HAS_PYPDF2:
        try:
            reader = PyPDF2.PdfReader(filelike)
            if getattr(reader, "is_encrypted", False):
                try:
                    reader.decrypt("")
                except Exception:
                    return ""
            text_parts = []
            for page in reader.pages:
                page_text = page.extract_text() or ""
                text_parts.append(page_text)
            return "\n".join(text_parts)
        except Exception:
            return ""

    return ""


# -------------------- CLEANING / NORMALIZATION --------------------

UNWANTED_LINE_PATTERNS = re.compile(
    r"(?i)\b("
    r"publisher|author|edition|college|university|institute|department|faculty|"
    r"copyright|isbn|doi|website|index|contents|table of contents|"
    r"acknowledg(e)?ments|foreword|preface|biography|about the author|"
    r"figure\s*\d+|fig\.\s*\d+|table\s*\d+|page\s*\d+|pp\.\s*\d+|"
    r"references?|bibliography|appendix|supplement|abstract"
    r")\b"
)
UNWANTED_INLINE = ["\u00ad", "\uf0b7", "\u2022", "\u200b", "\u200c", "\u200d"]
BULLET_PREFIX_RE = re.compile(r"^\s*([\-–—•·●◦\*]|\d+[\.)])\s+")
ONLY_DECOR_RE = re.compile(r"^\s*[-–—=~_+*#]+\s*$")
REF_MARKER_RE = re.compile(r"\[\s*\d+\s*\]|\(\s*\d+\s*\)")
MULTI_SPACE_RE = re.compile(r"[ \t]+")

STOPWORDS = set("""
a an the and or but if while with without within into onto from to of in on at by for as that this these those there here
is are was were be been being have has had do does did can could should would may might will shall it its itself himself herself themselves
about above below under over between among per via etc such than then so not no nor also more most less least very much many few each either neither both
""".split())

def normalize_text(raw_text: str) -> str:
    for ch in UNWANTED_INLINE:
        raw_text = raw_text.replace(ch, " ")
    raw_text = re.sub(r"-\s*\n\s*(?=\w)", "", raw_text)  # de-hyphenate across linebreaks
    raw_text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    raw_text = re.sub(r"[ \t]+\n", "\n", raw_text)
    raw_text = re.sub(r"\n{3,}", "\n\n", raw_text)
    return raw_text

def is_noise_line(line: str) -> bool:
    if not line:
        return True
    if ONLY_DECOR_RE.match(line):
        return True
    if UNWANTED_LINE_PATTERNS.search(line):
        return True
    tokens = line.split()
    if len(tokens) <= 3:
        return True
    letters = sum(c.isalpha() for c in line)
    digits = sum(c.isdigit() for c in line)
    if letters == 0:
        return True
    alpha_ratio = letters / max(1, len(line))
    digit_ratio = digits / max(1, len(line))
    if alpha_ratio < 0.55 or digit_ratio > 0.25:
        return True
    if len(line) >= 12 and line.upper() == line and any(c.isalpha() for c in line):
        # Only treat SHORT all-caps lines as headers/footers ("CHAPTER 5 FUELS").
        # Long all-caps lines are real content (common in scanned notes) and
        # must survive cleaning.
        if len(tokens) <= 8:
            return True
    return False

def remove_repeating_headers_footers(lines):
    freq = {}
    for ln in lines:
        key = ln.lower()
        freq[key] = freq.get(key, 0) + 1
    max_allowed = max(2, len(lines) // 40)
    return [ln for ln in lines if freq.get(ln.lower(), 0) <= max_allowed]

def clean_text(raw_text: str) -> str:
    text = normalize_text(raw_text)
    if not text.strip():
        return ""
    
    # Remove code blocks and programming artifacts
    text = re.sub(r'```[\s\S]*?```', '', text)  # Remove code blocks
    text = re.sub(r'`[^`]+`', '', text)  # Remove inline code
    text = re.sub(r'#include\s*<[^>]+>', '', text)  # Remove #include statements
    text = re.sub(r'printf\s*\([^)]*\);?', '', text)  # Remove printf statements
    text = re.sub(r'//.*$', '', text, flags=re.MULTILINE)  # Remove single line comments
    text = re.sub(r'/\*[\s\S]*?\*/', '', text)  # Remove multi-line comments
    text = re.sub(r'unsigned\s+long\s+long\s+\w+\s*=\s*\d+;', '', text)  # Remove variable declarations
    text = re.sub(r'for\s*\([^)]*\)\s*\{[^}]*\}', '', text)  # Remove for loops
    text = re.sub(r'while\s*\([^)]*\)\s*\{[^}]*\}', '', text)  # Remove while loops
    text = re.sub(r'if\s*\([^)]*\)\s*\{[^}]*\}', '', text)  # Remove if statements
    text = re.sub(r'return\s+[^;]+;', '', text)  # Remove return statements
    text = re.sub(r'#define\s+\w+\s+[^\n]+', '', text)  # Remove #define statements
    
    lines = [MULTI_SPACE_RE.sub(" ", ln.strip()) for ln in text.split("\n")]

    cleaned = []
    for ln in lines:
        if not ln:
            continue
        ln = BULLET_PREFIX_RE.sub("", ln)
        ln = REF_MARKER_RE.sub("", ln)
        ln = re.sub(r"\b\S+@\S+\b", "", ln)
        ln = re.sub(r"\bhttps?://\S+\b", "", ln)
        ln = re.sub(r"\bwww\.\S+\b", "", ln)
        ln = ln.strip(string.punctuation + " ")
        ln = MULTI_SPACE_RE.sub(" ", ln).strip()
        if ln:
            cleaned.append(ln)

    cleaned = remove_repeating_headers_footers(cleaned)
    cleaned = [ln for ln in cleaned if not is_noise_line(ln)]

    def _is_continuation(prev: str, nxt: str) -> bool:
        """Heuristic: should line `nxt` be joined to line `prev`?

        Join only for hyphenated line-wraps or when the previous line has no
        terminal punctuation and the next starts lowercase (wrapped prose).
        Standalone lines/bullets (starting uppercase, no punctuation) are kept
        as separate sentences/paragraphs so bullet-style notes work well.
        """
        if prev.endswith("-"):
            return True
        return bool(prev) and prev[-1] not in ".?!" and nxt[:1].islower()

    paragraphs = []
    current: List[str] = []
    for ln in cleaned:
        if not ln:
            if current:
                paragraphs.append(" ".join(current))
                current = []
            continue
        if current and not _is_continuation(current[-1], ln):
            paragraphs.append(" ".join(current))
            current = []
        current.append(ln)
    if current:
        paragraphs.append(" ".join(current))

    final_text = "\n\n".join(p.strip() for p in paragraphs if p.strip())
    final_text = MULTI_SPACE_RE.sub(" ", final_text).strip()
    return final_text


# -------------------- CHUNKING / SENTENCES --------------------

SENT_SPLIT = re.compile(r"(?<=[.?!])\s+(?=[A-Z(])")

def split_into_paragraphs(text: str) -> List[str]:
    paras = [p.strip() for p in text.split("\n\n") if p.strip()]
    # Prefer substantial paragraphs, but degrade gracefully for line-based
    # notes (bullets/slides) where paragraphs are short.
    for min_words in (20, 8, 1):
        kept = [p for p in paras if len(p.split()) >= min_words]
        if kept:
            return kept
    return paras

def split_into_sentences(text: str) -> List[str]:
    # Split on newlines first so line-based notes (each bullet/line = one
    # sentence even without terminal punctuation) don't get merged, then
    # apply the sentence-boundary regex within each line/paragraph.
    parts: List[str] = []
    for chunk in re.split(r"\n+", text):
        parts.extend(s.strip() for s in SENT_SPLIT.split(chunk))
    sents = [s for s in parts if s]
    out = []
    for s in sents:
        if REF_MARKER_RE.search(s):
            s = REF_MARKER_RE.sub("", s).strip()
        if len(s.split()) < 6:
            continue
        
        # Skip code-like sentences
        if re.search(r'[{}();]', s) and len(s.split()) < 15:
            continue
        if re.search(r'printf|#include|unsigned|long long', s, re.I):
            continue
        if re.search(r'^\s*[A-Z_][A-Z0-9_]*\s*=', s):
            continue
        
        letters = sum(c.isalpha() for c in s)
        digits = sum(c.isdigit() for c in s)
        if letters == 0:
            continue
        alpha_ratio = letters / max(1, len(s))
        digit_ratio = digits / max(1, len(s))
        if alpha_ratio < 0.55 or digit_ratio > 0.25:
            continue
        if len(s) >= 12 and s.upper() == s and any(c.isalpha() for c in s):
            # Only drop SHORT all-caps fragments (headers/footers). Long
            # all-caps sentences are real content (scanned notes) — keep them.
            if len(s.split()) <= 8:
                continue
        if not s.endswith((".", "?", "!")):
            s = s + "."
        out.append(s)
    return out


# -------------------- VECTORS --------------------

def build_vectorizer(texts: List[str]):
    if HAS_SKLEARN:
        vec = TfidfVectorizer(ngram_range=(1,2), max_df=0.95, min_df=1, stop_words="english")
        mat = vec.fit_transform(texts)
        return vec, mat
    vocab = {}
    rows = []
    for txt in texts:
        tokens = [w for w in re.findall(r"[A-Za-z]{2,}", txt.lower()) if w not in STOPWORDS]
        counts = {}
        for t in tokens:
            counts[t] = counts.get(t, 0) + 1
            if t not in vocab:
                vocab[t] = len(vocab)
        rows.append(counts)
    if np is None:
        # Minimal pure-Python vector (dense lists)
        mat = [[0.0 for _ in range(len(vocab))] for _ in range(len(texts))]
    else:
        mat = np.zeros((len(texts), len(vocab)), dtype=float)
    for i, counts in enumerate(rows):
        for t, c in counts.items():
            j = vocab[t]
            if np is None:
                mat[i][j] = float(c)
            else:
                mat[i, j] = c
        if np is None:
            s = sum(abs(x) for x in mat[i])
            if s > 0:
                mat[i] = [x / (s or 1e-9) for x in mat[i]]
        else:
            if mat[i].sum() > 0:
                mat[i] /= (np.linalg.norm(mat[i]) + 1e-9)
    return (vocab, None), mat

def vectorize(vec, texts: List[str]):
    if HAS_SKLEARN and isinstance(vec, TfidfVectorizer):
        return vec.transform(texts)
    vocab, _ = vec
    if np is None:
        mat = [[0.0 for _ in range(len(vocab))] for _ in range(len(texts))]
    else:
        mat = np.zeros((len(texts), len(vocab)), dtype=float)
    for i, txt in enumerate(texts):
        tokens = [w for w in re.findall(r"[A-Za-z]{2,}", txt.lower()) if w not in STOPWORDS]
        counts = {}
        for t in tokens:
            counts[t] = counts.get(t, 0) + 1
        for t, c in counts.items():
            if t in vocab:
                j = vocab[t]
                if np is None:
                    mat[i][j] = float(c)
                else:
                    mat[i, j] = c
        if np is None:
            s = sum(abs(x) for x in mat[i])
            if s > 0:
                mat[i] = [x / (s or 1e-9) for x in mat[i]]
        else:
            if mat[i].sum() > 0:
                mat[i] /= (np.linalg.norm(mat[i]) + 1e-9)
    return mat

def cos_sim(A, B):
    if HAS_SKLEARN and "csr_matrix" in str(type(A)):
        return cosine_similarity(A, B)
    if np is None:
        # Simple cosine for lists
        def norm(v):
            return sum(x*x for x in v) ** 0.5
        if isinstance(A[0], (int, float)):
            A = [A]
        if isinstance(B[0], (int, float)):
            B = [B]
        out = []
        for a in A:
            row = []
            na = norm(a) + 1e-9
            for b in B:
                nb = norm(b) + 1e-9
                dot = sum(x*y for x,y in zip(a,b))
                row.append(dot / (na * nb))
            out.append(row)
        return out
    else:
        if A.ndim == 1: A = A.reshape(1, -1)
        if B.ndim == 1: B = B.reshape(1, -1)
        denom = (np.linalg.norm(A, axis=1, keepdims=True) + 1e-9) * (np.linalg.norm(B, axis=1, keepdims=True) + 1e-9)
        return (A @ B.T) / denom


# -------------------- PURE-PYTHON VECTOR HELPERS (numpy-free fallbacks) --------------------
# These helpers make every downstream feature (summary, flashcards, QA, quiz)
# work even when numpy and scikit-learn are not installed (e.g. Vercel/free tier).

def _vector_centroid(mat):
    """Column-mean (centroid) of a document matrix.

    Supports: sklearn csr_matrix, numpy 2D array, and pure-Python list-of-rows.
    """
    if HAS_SKLEARN and "csr_matrix" in str(type(mat)):
        return mat.mean(axis=0)
    if np is not None and not isinstance(mat, list):
        return mat.mean(axis=0, keepdims=True)
    cols = len(mat[0]) if mat and isinstance(mat[0], list) else 0
    n = max(1, len(mat))
    return [sum(row[j] for row in mat) / n for j in range(cols)]


def _centroid_scores(mat, centroid):
    """Similarity of each row to the centroid vector (higher = more central)."""
    if HAS_SKLEARN and "csr_matrix" in str(type(mat)):
        return (mat @ centroid.T).A.ravel()
    if np is not None and not isinstance(mat, list):
        return (mat @ centroid.T).ravel()
    cols = len(centroid)
    return [sum(row[j] * centroid[j] for j in range(cols)) for row in mat]


def _argsort_desc(scores):
    """Indices of scores sorted descending (works with numpy arrays and lists)."""
    if np is not None and not isinstance(scores, list):
        return np.argsort(-scores).tolist()
    return sorted(range(len(scores)), key=lambda i: -scores[i])


def _ravel(x):
    """Flatten a matrix (numpy array, csr, or nested lists) to a 1-D sequence."""
    if isinstance(x, list):
        out = []
        for row in x:
            if isinstance(row, list):
                out.extend(row)
            else:
                out.append(row)
        return out
    if HAS_SKLEARN and "csr_matrix" in str(type(x)):
        return x.toarray().ravel()
    return np.asarray(x).ravel()


def _sim_scalar(a, b) -> float:
    """Scalar cosine similarity between two vectors/matrix-rows."""
    s = cos_sim(a, b)
    if isinstance(s, list):
        return float(s[0][0])
    return float(s[0, 0])


# -------------------- SUMMARY (GLOBAL MAIN POINTS) --------------------

def summarize_main_points(all_sentences: List[str], sentence_count: int = 10) -> List[str]:
    if not all_sentences:
        return []
    vec, mat = build_vectorizer(all_sentences)
    centroid = _vector_centroid(mat)
    scores = _centroid_scores(mat, centroid)
    idxs_sorted = _argsort_desc(scores)
    selected = []
    selected_vecs = []
    for idx in idxs_sorted:
        if len(selected) >= sentence_count:
            break
        cand_vec = vectorize(vec, [all_sentences[idx]])
        redundant = False
        for sv in selected_vecs:
            if _sim_scalar(cand_vec, sv) > 0.75:
                redundant = True
                break
        if not redundant:
            selected.append(all_sentences[idx])
            selected_vecs.append(cand_vec)
    return selected


# -------------------- FLASHCARDS --------------------

def sentence_is_similar_to_any(target: str, others: List[str], threshold: float = 0.8) -> bool:
    if not others:
        return False
    vec, mat = build_vectorizer(others + [target])
    q = vectorize(vec, [target])
    pool = vectorize(vec, others)
    if hasattr(pool, "shape"):
        if pool.shape[0] == 0:
            return False
    elif not pool:
        return False
    sims = _ravel(cos_sim(pool, q))
    if np is not None and not isinstance(sims, list):
        return float(np.max(sims)) >= threshold
    return max(sims, default=0.0) >= threshold

def make_short_explanation_from_context(paragraph: str, anchor_sentence: str,
                                        min_words: int = 25, max_words: int = 55) -> str:
    sents = split_into_sentences(paragraph)
    if not sents:
        return anchor_sentence
    try:
        idx = sents.index(anchor_sentence)
    except ValueError:
        idx = 0
    window = sents[max(0, idx-1): min(len(sents), idx+2)]
    text = " ".join(window).strip()
    words = text.split()
    if len(words) > max_words:
        text = " ".join(words[:max_words]).rstrip(",;: ") + "."
    elif len(words) < min_words and idx+2 < len(sents):
        text = (text + " " + sents[min(len(sents)-1, idx+2)]).strip()
    if not text.endswith((".", "?", "!")):
        text += "."
    return text

def generate_paragraph_flashcards(paragraph: str, banned_sentences: List[str],
                                  limit: int, explain_min: int, explain_max: int) -> List[Dict]:
    sentences = split_into_sentences(paragraph)
    if not sentences:
        return []
    vec, mat = build_vectorizer(sentences)
    centroid = _vector_centroid(mat)
    scores = _centroid_scores(mat, centroid)
    order = _argsort_desc(scores)

    selected = []
    selected_vecs = []
    for idx in order:
        if len(selected) >= limit:
            break
        s = sentences[idx]
        if banned_sentences and sentence_is_similar_to_any(s, banned_sentences, threshold=0.8):
            continue
        cand_vec = vectorize(vec, [s])
        redundant = False
        for sv in selected_vecs:
            if _sim_scalar(cand_vec, sv) > 0.7:
                redundant = True
                break
        if redundant:
            continue
        selected.append(s)
        selected_vecs.append(cand_vec)

    cards = []
    for s in selected:
        title = s  # full sentence
        explanation = make_short_explanation_from_context(paragraph, s, explain_min, explain_max)
        cards.append({"summary": title, "explanation": explanation})
    return cards

def generate_global_fallback_flashcards(all_text: str, banned_sentences: List[str],
                                        desired_count: int, explain_min: int, explain_max: int) -> List[Dict]:
    # Fallback: sample flashcards from the whole document when paragraph-level fails
    sentences = split_into_sentences(all_text)
    if not sentences:
        return []
    vec, mat = build_vectorizer(sentences)
    centroid = _vector_centroid(mat)
    scores = _centroid_scores(mat, centroid)
    order = _argsort_desc(scores)

    selected = []
    selected_vecs = []
    for idx in order:
        if len(selected) >= desired_count:
            break
        s = sentences[idx]
        if banned_sentences and sentence_is_similar_to_any(s, banned_sentences, threshold=0.8):
            continue
        cand_vec = vectorize(vec, [s])
        redundant = False
        for sv in selected_vecs:
            if _sim_scalar(cand_vec, sv) > 0.7:
                redundant = True
                break
        if redundant:
            continue
        selected.append(s)
        selected_vecs.append(cand_vec)

    cards = []
    for s in selected:
        title = s
        # For fallback, explanation = the same sentence (already concise)
        explanation = s if len(s.split()) >= explain_min else (s + " " + s)
        words = explanation.split()
        if len(words) > explain_max:
            explanation = " ".join(words[:explain_max]).rstrip(",;: ") + "."
        if not explanation.endswith((".", "?", "!")):
            explanation += "."
        cards.append({"summary": title, "explanation": explanation})
    return cards


# -------------------- QA (RETRIEVAL, NO LLM) --------------------

def build_qa_index(paragraphs: List[str]):
    if not paragraphs:
        return None, None
    vec, mat = build_vectorizer(paragraphs)
    return vec, mat

def answer_question(query: str, paragraphs: List[str], vec, mat, top_k: int = 3) -> Dict:
    if not query.strip() or not paragraphs:
        return {"answer": "No content available.", "sources": []}
    qv = vectorize(vec, [query])
    sims = _ravel(cos_sim(mat, qv))
    top_idx = _argsort_desc(sims)[:top_k]
    chosen = [paragraphs[i] for i in top_idx if i < len(paragraphs) and sims[i] > 0.05]

    sent_pool = []
    for p in chosen:
        sent_pool.extend(split_into_sentences(p))
    if not sent_pool:
        return {"answer": "I could not find a relevant answer in the PDF.", "sources": []}

    svec, smat = build_vectorizer(sent_pool)
    qsv = vectorize(svec, [query])
    ssims = _ravel(cos_sim(smat, qsv))
    best_sent_idx = _argsort_desc(ssims)[:5]
    best_sents = [sent_pool[i] for i in best_sent_idx if ssims[i] > 0.05]

    answer = " ".join(best_sents) if best_sents else "I could not find a clear answer in the PDF."
    return {"answer": answer, "sources": chosen}


# -------------------- QUIZ GENERATION --------------------

def generate_moderate_questions(text: str, num_questions: int = 10) -> List[Dict]:
    """Generate intelligent MCQ questions from text using the exact repository algorithm"""
    import random
    import re
    
    # Clean text first
    cleaned_text = clean_text(text)
    sentences = sentence_split(cleaned_text)
    
    if len(sentences) < 1:
        return []
    
    # Extract top terms with phrases (exact repository algorithm)
    top_terms = extract_top_terms_with_phrases(cleaned_text, 400)
    top_pool = [t['term'] for t in top_terms[:250]]
    
    # Choose targets (key terms to create questions about)
    raw_targets = choose_targets(sentences, top_terms, num_questions * 3)
    if not raw_targets:
        return []
    
    questions = []
    used_prompts = set()
    
    # Mix of question types (exact repository ratios)
    want_cloze = max(1, int(num_questions * 0.4))
    want_def = max(1, int(num_questions * 0.3))
    want_true = max(1, int(num_questions * 0.3))
    
    random.shuffle(raw_targets)
    
    def try_add(q):
        if not q:
            return False
        if q['prompt'] in used_prompts:
            return False
        if not q.get('options') or len(q['options']) != 4:
            return False
        used_prompts.add(q['prompt'])
        questions.append(q)
        return True
    
    # Generate cloze questions (fill in the blank)
    for target in raw_targets:
        if len([q for q in questions if q.get('type') == 'cloze']) >= want_cloze:
            break
        try_add(make_cloze_question(target, sentences, top_pool))
    
    # Generate definition questions
    for target in raw_targets:
        if len([q for q in questions if q.get('type') == 'definition']) >= want_def:
            break
        try_add(make_definition_question(target, sentences, top_pool))
    
    # Generate true/false questions
    for target in raw_targets:
        if len([q for q in questions if q.get('type') == 'trueabout']) >= want_true:
            break
        try_add(make_true_about_question(target, sentences, top_pool))
    
    # Fill remaining questions
    for target in raw_targets:
        if len(questions) >= num_questions:
            break
        attempts = [
            make_cloze_question(target, sentences, top_pool),
            make_true_about_question(target, sentences, top_pool),
            make_definition_question(target, sentences, top_pool)
        ]
        for q in attempts:
            if len(questions) >= num_questions:
                break
            try_add(q)
    
    # Fallback: permissive generator for hard texts (bullet notes, short
    # lines, unusual formatting) so a valid text PDF never returns 0 questions.
    if len(questions) < num_questions:
        existing = set(used_prompts)
        for q in generate_fallback_questions(cleaned_text, num_questions - len(questions)):
            if q['prompt'] not in existing:
                questions.append(q)
                existing.add(q['prompt'])
    
    return questions[:num_questions]


def generate_fallback_questions(text: str, num_questions: int) -> List[Dict]:
    """Permissive fallback MCQ generator.

    Works on short/bullet-style sentences (>= 6 words) that the main
    algorithm rejects, using relaxed distractor selection. Guarantees
    questions for any PDF that has real (non-image) text.
    """
    import random
    import re
    
    sentences = sentence_split(text)
    if len(sentences) < 2:
        return []
    
    top = extract_top_terms_with_phrases(text, 150)
    pool = [t['term'] for t in top
            if len(t['term']) >= 4 and not re.match(r'^\d+$', t['term'])]
    if len(pool) < 4:
        return []
    
    def title_case(s):
        return re.sub(r'\b([a-z])', lambda m: m.group(1).upper(), s)
    
    def jaccard(a, b):
        sa, sb = set(a.split()), set(b.split())
        return len(sa & sb) / max(1, len(sa | sb))
    
    questions = []
    used_prompts = set()
    used_terms = set()
    
    order = list(range(len(sentences)))
    random.shuffle(order)
    for idx in order:
        if len(questions) >= num_questions:
            break
        s = sentences[idx]
        if len(s.split()) < 6:
            continue
        s_lc = s.lower()
        
        candidates = [t for t in pool
                      if t not in used_terms
                      and re.search(r'\b' + re.escape(t) + r'\b', s, re.I)]
        if not candidates:
            continue
        term = random.choice(candidates)
        
        rx = re.compile(r'\b' + re.escape(term) + r'\b', re.I)
        prompt = rx.sub('_____', s)
        if '_____' not in prompt or prompt in used_prompts:
            continue
        
        dis_pool = [t for t in pool
                    if t != term and jaccard(t.lower(), term.lower()) < 0.5]
        if len(dis_pool) < 3:
            continue
        random.shuffle(dis_pool)
        
        answer = title_case(term)
        opts = [answer] + [title_case(t) for t in dis_pool[:3]]
        random.shuffle(opts)
        correct = opts.index(answer)
        
        questions.append({
            'type': 'cloze',
            'prompt': prompt,
            'options': [f"{chr(65 + i)}) {o}" for i, o in enumerate(opts)],
            'correct_answer': correct
        })
        used_prompts.add(prompt)
        used_terms.add(term)
    
    return questions


def sentence_split(text: str) -> List[str]:
    """Split text into sentences (exact repository algorithm).

    Also splits on newlines so line-based notes (bullets/slides) don't merge
    into one giant 'sentence' when terminal punctuation is missing.
    """
    import re
    parts = re.split(r"\n+|(?<=[.?!])\s+(?=[A-Z(])", text)
    return [p.strip() for p in parts if p.strip()]


def extract_top_terms_with_phrases(text: str, max_terms: int = 400) -> List[Dict]:
    """Extract top terms and phrases from text (exact repository algorithm)"""
    import re
    
    # Tokenize text (exact repository algorithm)
    tokens = re.sub(r'[^a-z0-9\s\-]', ' ', text.lower()).split()
    tokens = [t for t in tokens if t]
    
    # Stop words (exact repository set)
    stopwords = {
        'the', 'a', 'an', 'and', 'or', 'but', 'if', 'then', 'else', 'for', 'to', 'of', 'in', 'on', 'by', 'with', 'as', 'at', 'from', 'that', 'this', 'these', 'those', 'it', 'its', 'is', 'are', 'was', 'were', 'be', 'been', 'being', 'which', 'who', 'whom', 'whose', 'what', 'when', 'where', 'why', 'how', 'not', 'so', 'into', 'we', 'you', 'they', 'he', 'she', 'i', 'their', 'our', 'your', 'his', 'her', 'them', 'us', 'me', 'can', 'could', 'should', 'would', 'may', 'might', 'will', 'shall', 'do', 'does', 'did', 'done', 'have', 'has', 'had', 'more', 'most', 'many', 'much', 'very', 'such', 'other', 'than', 'also', 'because', 'however', 'therefore', 'thus', 'hence', 'consequently', 'in contrast', 'for example', 'whereas', 'while'
    }
    
    # Count unigrams, bigrams, trigrams (exact repository algorithm)
    unigrams = {}
    bigrams = {}
    trigrams = {}
    
    for i, token in enumerate(tokens):
        if token not in stopwords and len(token) >= 3 and re.search(r'[a-z]', token):
            unigrams[token] = unigrams.get(token, 0) + 1
            
            # Bigrams
            if i < len(tokens) - 1:
                next_token = tokens[i + 1]
                if next_token not in stopwords and len(next_token) >= 3 and re.search(r'[a-z]', next_token):
                    bigram = f"{token} {next_token}"
                    bigrams[bigram] = bigrams.get(bigram, 0) + 1
                    
                    # Trigrams
                    if i < len(tokens) - 2:
                        next_next_token = tokens[i + 2]
                        if next_next_token not in stopwords and len(next_next_token) >= 3 and re.search(r'[a-z]', next_next_token):
                            trigram = f"{token} {next_token} {next_next_token}"
                            trigrams[trigram] = trigrams.get(trigram, 0) + 1
    
    # Score and rank terms (exact repository scoring)
    scored = []
    for term, count in unigrams.items():
        scored.append({'term': term, 'count': count, 'len': 1, 'score': count})
    for term, count in bigrams.items():
        scored.append({'term': term, 'count': count, 'len': 2, 'score': count * 1.6})
    for term, count in trigrams.items():
        scored.append({'term': term, 'count': count, 'len': 3, 'score': count * 2.2})
    
    scored.sort(key=lambda x: x['score'], reverse=True)
    
    # Remove overlapping terms (exact repository algorithm)
    taken = set()
    result = []
    for term_data in scored:
        term = term_data['term']
        if not any(taken_term in term or term in taken_term for taken_term in taken):
            taken.add(term)
            result.append(term_data)
            if len(result) >= max_terms:
                break
    
    return result


def choose_targets(sentences: List[str], top_terms: List[Dict], want: int) -> List[Dict]:
    """Choose target terms for question generation (exact repository algorithm)"""
    import re
    
    # Score sentences for informativeness (exact repository algorithm)
    informative_sentences = []
    for i, sentence in enumerate(sentences):
        word_count = len(sentence.split())
        has_comma = ',' in sentence
        has_link = bool(re.search(r'\b(therefore|however|because|whereas|although|while|thus|hence|consequently|in contrast|for example)\b', sentence, re.I))
        
        score = 0
        if 12 <= word_count <= 35:
            score += 2
        elif 8 <= word_count <= 45:
            score += 1
        if has_comma:
            score += 1
        if has_link:
            score += 1.2
        if re.search(r'[;:]', sentence):
            score += 0.5
            
        informative_sentences.append({'sentence': sentence, 'idx': i, 'score': score})
    
    informative_sentences.sort(key=lambda x: x['score'], reverse=True)
    
    # Extract targets (exact repository algorithm)
    term_set = {t['term'] for t in top_terms}
    selected = []
    used_sent_idx = set()
    used_terms = set()
    
    for item in informative_sentences:
        if len(selected) >= want * 2:
            break
        if item['idx'] in used_sent_idx:
            continue
            
        sentence = item['sentence']
        tokens = re.sub(r'[^a-z0-9\s\-]', ' ', sentence.lower()).split()
        tokens = [t for t in tokens if t]
        candidates = []
        
        # Look for 3-word phrases first, then 2-word, then 1-word (exact repository algorithm)
        for length in [3, 2, 1]:
            for i in range(len(tokens) - length + 1):
                phrase = ' '.join(tokens[i:i+length])
                if phrase in term_set and len(phrase) >= 4 and not re.match(r'^\d+$', phrase):
                    candidates.append({'phrase': phrase, 'len': length})
            if candidates:
                break
        
        if not candidates:
            continue
            
        # Choose first unused candidate
        chosen = None
        for cand in candidates:
            if cand['phrase'] not in used_terms:
                chosen = cand
                break
                
        if not chosen:
            continue
            
        selected.append({
            'idx': item['idx'],
            'sentence': sentence,
            'answer': chosen['phrase']
        })
        used_sent_idx.add(item['idx'])
        used_terms.add(chosen['phrase'])
    
    return selected[:want]


def make_cloze_question(target: Dict, sentences: List[str], top_pool: List[str]) -> Dict:
    """Create a fill-in-the-blank question (exact repository algorithm)"""
    import re
    
    answer = target['answer']
    sentence = target['sentence']
    
    # Replace the answer with blank (exact repository algorithm)
    pattern = re.compile(r'\b' + re.escape(answer) + r'\b', re.I)
    if not pattern.search(sentence):
        return None
        
    prompt = pattern.sub('_____', sentence)
    if len(prompt.split()) < 10:
        return None
    
    # Build distractors
    distractors = build_distractors(answer, top_pool, sentences, target['idx'], 3)
    if len(distractors) < 3:
        return None
    
    # Create options (exact repository algorithm)
    options_raw = shuffle_and_label([title_case(answer)] + distractors)
    correct_index = next(i for i, opt in enumerate(options_raw) if opt['text'].lower() == title_case(answer).lower())
    
    return {
        'type': 'cloze',
        'prompt': prompt,
        'options': [f"{opt['label']}) {opt['text']}" for opt in options_raw],
        'correct_answer': correct_index
    }


def make_definition_question(target: Dict, sentences: List[str], top_pool: List[str]) -> Dict:
    """Create a definition question (exact repository algorithm)"""
    import re
    
    answer = target['answer'].title()
    sentence = target['sentence']
    
    # Look for definition pattern (exact repository algorithm)
    def_pattern = re.compile(r'\b' + re.escape(target['answer']) + r'\b\s+(is|are|refers to|means|denotes|involves|consists of|can be defined as)\s+([^.;:]+)', re.I)
    match = def_pattern.search(sentence)
    
    if not match:
        return None
        
    def_text = f"{answer} {match.group(1)} {match.group(2).strip()}."
    
    # Build distractors
    distractors = build_distractors(target['answer'], top_pool, sentences, target['idx'], 3)
    if len(distractors) < 3:
        return None
    
    # Create options (exact repository algorithm)
    options_raw = shuffle_and_label([def_text] + [f"{d} is {generate_generic_definition_tail(d)}" for d in distractors])
    correct_index = next(i for i, opt in enumerate(options_raw) if opt['text'] == def_text)
    
    return {
        'type': 'definition',
        'prompt': f"Which option best defines {answer}?",
        'options': [f"{opt['label']}) {opt['text']}" for opt in options_raw],
        'correct_answer': correct_index
    }


def make_true_about_question(target: Dict, sentences: List[str], top_pool: List[str]) -> Dict:
    """Create a true/false question (exact repository algorithm)"""
    import re
    
    answer = target['answer'].title()
    sentence = target['sentence']
    
    if len(sentence.split()) < 10:
        return None
    
    # Create true statement (exact repository algorithm)
    true_stmt = sentence
    if not re.search(r'\b' + re.escape(answer) + r'\b', sentence, re.I):
        true_stmt = re.sub(r'\b' + re.escape(target['answer']) + r'\b', answer, sentence, flags=re.I)
    
    # Create false statements
    distractors = build_distractors(target['answer'], top_pool, sentences, target['idx'], 6)
    false_stmts = []
    
    for distractor in distractors[:6]:
        false_stmt = re.sub(r'\b' + re.escape(answer) + r'\b', distractor, true_stmt, flags=re.I)
        if false_stmt != true_stmt and len(false_stmt.split()) >= 10:
            false_stmts.append(false_stmt)
        if len(false_stmts) >= 3:
            break
    
    if len(false_stmts) < 3:
        return None
    
    # Create options (exact repository algorithm)
    options_raw = shuffle_and_label([true_stmt] + false_stmts)
    correct_index = next(i for i, opt in enumerate(options_raw) if opt['text'] == true_stmt)
    
    return {
        'type': 'trueabout',
        'prompt': f"Which of the following is true about {answer}?",
        'options': [f"{opt['label']}) {opt['text']}" for opt in options_raw],
        'correct_answer': correct_index
    }


def build_distractors(answer: str, pool: List[str], sentences: List[str], anchor_idx: int, k: int) -> List[str]:
    """Build distractors for multiple choice questions (exact repository algorithm)"""
    import re
    
    answer_lower = answer.lower()
    answer_len = len(answer_lower)
    
    # Get nearby text for context (exact repository algorithm)
    nearby_sentences = []
    for i in range(max(0, anchor_idx - 2), min(len(sentences), anchor_idx + 3)):
        if i != anchor_idx:
            nearby_sentences.append(sentences[i])
    
    nearby_text = ' '.join(nearby_sentences)
    nearby_terms = [t['term'] for t in extract_top_terms_with_phrases(nearby_text, 60)]
    
    # Combine all candidates
    all_candidates = list(set(nearby_terms + pool))
    
    # Filter candidates (exact repository algorithm)
    candidates = []
    for term in all_candidates:
        term_lower = term.lower()
        if (term_lower != answer_lower and
            abs(len(term) - answer_len) <= 5 and
            jaccard_characters(term_lower, answer_lower) < 0.65 and
            term_lower not in {'the', 'a', 'an', 'and', 'or', 'but', 'if', 'then', 'else', 'for', 'to', 'of', 'in', 'on', 'by', 'with', 'as', 'at', 'from'} and
            not re.match(r'^\d+$', term)):
            candidates.append(term)
    
    random.shuffle(candidates)
    
    # Remove similar candidates (exact repository algorithm)
    unique = []
    for candidate in candidates:
        if len(unique) >= k:
            break
        if not any(jaccard_characters(unique_term.lower(), candidate.lower()) >= 0.65 for unique_term in unique):
            unique.append(candidate)
    
    return [title_case(term) for term in unique]


def jaccard_characters(a: str, b: str) -> float:
    """Calculate Jaccard similarity based on characters (exact repository algorithm)"""
    set_a = set(a)
    set_b = set(b)
    intersection = set_a & set_b
    union = set_a | set_b
    return len(intersection) / max(1, len(union))


def title_case(s: str) -> str:
    """Convert to title case (exact repository algorithm)"""
    import re
    return re.sub(r'\b([a-z])', lambda m: m.group(1).upper(), s)


def shuffle_and_label(arr: List[str]) -> List[Dict]:
    """Shuffle and label options (exact repository algorithm)"""
    import random
    copy = arr[:]
    random.shuffle(copy)
    return [{'label': chr(65 + i), 'text': text} for i, text in enumerate(copy)]


def generate_generic_definition_tail(term: str) -> str:
    """Generate generic definition tail (exact repository algorithm)"""
    import random
    tails = [
        'primarily concerned with peripheral aspects unrelated to core concepts',
        'a general approach focusing on adjacent but distinct principles',
        'characterized by features that are context-dependent rather than essential',
        'commonly associated with outcomes rather than underlying mechanisms'
    ]
    return random.choice(tails)


# NOTE: an older, stricter generate_fallback_questions() used to be defined
# here and silently overrode the permissive version above (later definitions
# win in Python). It has been removed so generate_moderate_questions() gets
# the permissive fallback that works on bullet/short-sentence notes.


def extract_key_concepts(paragraph: str) -> Dict[str, str]:
    """Extract key concepts and their definitions from a paragraph"""
    concepts = {}
    
    # Look for definition patterns
    definition_patterns = [
        r'(\w+(?:\s+\w+){0,3})\s+is\s+(?:a\s+)?(?:an\s+)?(?:the\s+)?([^.!?]+)',
        r'(\w+(?:\s+\w+){0,3})\s+refers\s+to\s+([^.!?]+)',
        r'(\w+(?:\s+\w+){0,3})\s+means\s+([^.!?]+)',
        r'(\w+(?:\s+\w+){0,3})\s+can\s+be\s+defined\s+as\s+([^.!?]+)',
        r'(\w+(?:\s+\w+){0,3})\s+are\s+([^.!?]+)',
    ]
    
    for pattern in definition_patterns:
        matches = re.finditer(pattern, paragraph, re.IGNORECASE)
        for match in matches:
            concept = match.group(1).strip()
            definition = match.group(2).strip()
            
            # Clean up the concept and definition
            concept = re.sub(r'[^\w\s]', '', concept).strip()
            definition = re.sub(r'[^\w\s.,;]', '', definition).strip()
            
            if len(concept) > 2 and len(definition) > 10 and concept not in concepts:
                concepts[concept] = definition
    
    return concepts


def generate_definition_question(concept: str, definition: str, context: str) -> Dict:
    """Generate a definition-based question"""
    import random
    
    # Create a fill-in-the-blank question
    question_text = definition.replace(concept, "_____")
    
    # Generate plausible wrong options
    wrong_options = generate_wrong_options(concept, context)
    
    # Create options
    options = [concept] + wrong_options[:3]
    random.shuffle(options)
    
    correct_index = options.index(concept)
    
    return {
        "question": f"What is {concept}?",
        "options": options,
        "correct_answer": correct_index
    }


def generate_application_question(concept: str, definition: str, context: str) -> Dict:
    """Generate an application-based question"""
    import random
    
    # Create application scenarios
    applications = [
        f"When would you use {concept}?",
        f"What is the purpose of {concept}?",
        f"How does {concept} work?",
        f"What are the benefits of {concept}?"
    ]
    
    question_text = random.choice(applications)
    
    # Generate options based on the definition and context
    correct_answer = extract_key_phrase(definition)
    wrong_options = generate_wrong_options(concept, context)
    
    options = [correct_answer] + wrong_options[:3]
    random.shuffle(options)
    
    correct_index = options.index(correct_answer)
    
    return {
        "question": question_text,
        "options": options,
        "correct_answer": correct_index
    }


def generate_example_question(concept: str, definition: str, context: str) -> Dict:
    """Generate an example-based question"""
    import random
    
    # Create example scenarios
    examples = [
        f"Which of the following is an example of {concept}?",
        f"Which scenario demonstrates {concept}?",
        f"What would be a good example of {concept}?"
    ]
    
    question_text = random.choice(examples)
    
    # Generate options
    correct_answer = f"Using {concept} in practice"
    wrong_options = [
        "Something unrelated to the concept",
        "A different programming concept",
        "An outdated method"
    ]
    
    options = [correct_answer] + wrong_options
    random.shuffle(options)
    
    correct_index = options.index(correct_answer)
    
    return {
        "question": question_text,
        "options": options,
        "correct_answer": correct_index
    }


def generate_wrong_options(concept: str, context: str) -> List[str]:
    """Generate plausible wrong options"""
    import random
    
    # Extract other concepts from context
    other_concepts = []
    words = context.split()
    
    # Look for capitalized words (potential concepts)
    for word in words:
        if word[0].isupper() and len(word) > 3 and word.lower() != concept.lower():
            other_concepts.append(word)
    
    # Generate wrong options
    wrong_options = []
    
    # Add other concepts
    wrong_options.extend(other_concepts[:2])
    
    # Add generic wrong options
    generic_wrong = [
        "None of the above",
        "All of the above", 
        "Cannot be determined",
        "Depends on the context"
    ]
    
    wrong_options.extend(generic_wrong)
    
    return wrong_options[:4]


def extract_key_phrase(definition: str) -> str:
    """Extract the most important phrase from a definition"""
    # Take the first meaningful part of the definition
    sentences = definition.split('.')
    if sentences:
        first_sentence = sentences[0].strip()
        if len(first_sentence) > 50:
            first_sentence = first_sentence[:50] + "..."
        return first_sentence
    return definition[:50] + "..." if len(definition) > 50 else definition


# -------------------- PROPER QUIZ GENERATION FUNCTIONS FROM REPOSITORY --------------------

def extract_top_terms_with_phrases(text: str, max_terms: int = 400) -> List[Dict]:
    """Extract top terms with phrases from text"""
    import re
    
    def tokenize(text):
        return re.sub(r'[^a-z0-9\s\-]', ' ', text.lower()).split()
    
    def is_good_token(t):
        return t not in STOPWORDS and re.search(r'[a-z]', t) and len(t) >= 3
    
    tokens = tokenize(text)
    unigrams = {}
    bigrams = {}
    trigrams = {}
    
    for i in range(len(tokens)):
        t1 = tokens[i]
        if is_good_token(t1):
            unigrams[t1] = unigrams.get(t1, 0) + 1
        
        if i + 1 < len(tokens):
            t2 = tokens[i + 1]
            if is_good_token(t1) and is_good_token(t2):
                bigram = f"{t1} {t2}"
                bigrams[bigram] = bigrams.get(bigram, 0) + 1
        
        if i + 2 < len(tokens):
            t2, t3 = tokens[i + 1], tokens[i + 2]
            if is_good_token(t1) and is_good_token(t2) and is_good_token(t3):
                trigram = f"{t1} {t2} {t3}"
                trigrams[trigram] = trigrams.get(trigram, 0) + 1
    
    scored = []
    for term, count in unigrams.items():
        scored.append({'term': term, 'count': count, 'len': 1, 'score': count})
    for term, count in bigrams.items():
        scored.append({'term': term, 'count': count, 'len': 2, 'score': count * 1.6})
    for term, count in trigrams.items():
        scored.append({'term': term, 'count': count, 'len': 3, 'score': count * 2.2})
    
    scored.sort(key=lambda x: x['score'], reverse=True)
    
    taken = set()
    result = []
    for t in scored:
        key = t['term']
        if any(key in x or x in key for x in taken):
            continue
        taken.add(key)
        result.append(t)
        if len(result) >= max_terms:
            break
    
    return result


def choose_targets(sentences: List[str], top_terms: List[Dict], want: int) -> List[Dict]:
    """Choose targets for question generation"""
    import random
    import re
    
    def tokenize(text):
        return re.sub(r'[^a-z0-9\s\-]', ' ', text.lower()).split()
    
    def word_count(s):
        return len((s or '').strip().split())
    
    def pick_informative_sentences(sentences):
        scored = []
        for idx, s in enumerate(sentences):
            wc = word_count(s)
            has_comma = ',' in s
            has_link = bool(re.search(r'\b(therefore|however|because|whereas|although|while|thus|hence|consequently|in contrast|for example)\b', s, re.I))
            
            score = 0
            if 12 <= wc <= 35:
                score += 2
            elif 8 <= wc <= 45:
                score += 1
            if has_comma:
                score += 1
            if has_link:
                score += 1.2
            if re.search(r'[;:]', s):
                score += 0.5
            
            scored.append({'s': s, 'idx': idx, 'wc': wc, 'score': score})
        
        scored.sort(key=lambda x: x['score'], reverse=True)
        return scored
    
    informative = pick_informative_sentences(sentences)
    term_set = set(t['term'] for t in top_terms)
    selected = []
    used_sent_idx = set()
    used_terms = set()
    
    for item in informative:
        if len(selected) >= want * 2:
            break
        if item['idx'] in used_sent_idx:
            continue
        
        s = item['s']
        tokens = tokenize(s)
        candidates = []
        
        for length in range(3, 0, -1):
            for i in range(len(tokens) - length + 1):
                phrase = ' '.join(tokens[i:i + length])
                if phrase in term_set and len(phrase) >= 4 and not re.match(r'^\d+$', phrase):
                    candidates.append({'phrase': phrase, 'len': length})
            if candidates:
                break
        
        if not candidates:
            continue
        
        chosen = None
        for cand in candidates:
            if cand['phrase'] not in used_terms:
                chosen = cand
                break
        
        if not chosen:
            continue
        
        selected.append({
            'idx': item['idx'],
            'sentence': s,
            'answer': chosen['phrase']
        })
        used_sent_idx.add(item['idx'])
        used_terms.add(chosen['phrase'])
    
    return selected[:want]


def make_cloze_question(target: Dict, sentences: List[str], top_pool: List[str]) -> Dict:
    """Make a cloze (fill-in-the-blank) question"""
    import random
    import re
    
    def word_count(s):
        return len((s or '').strip().split())
    
    def title_case(s):
        return re.sub(r'\b([a-z])', lambda m: m.group(1).upper(), s)
    
    def escape_regex(s):
        return re.sub(r'[.*+?^${}()|[\]\\]', r'\\\g<0>', s)
    
    def jaccard_characters(a, b):
        sa = set(a.split())
        sb = set(b.split())
        inter = sa.intersection(sb)
        union = sa.union(sb)
        return len(inter) / max(1, len(union))
    
    def build_distractors(answer, pool, sentences, anchor_idx, k):
        answer_lc = answer.lower()
        answer_len = len(answer_lc)
        
        # Get nearby text
        nearby_text = ' '.join(sentences[max(0, anchor_idx-2):min(len(sentences), anchor_idx+3)])
        nearby_terms = [t['term'] for t in extract_top_terms_with_phrases(nearby_text, 60)]
        all_candidates = list(set(nearby_terms + pool))
        
        candidates = [
            t for t in all_candidates
            if t.lower() != answer_lc
            and abs(len(t) - answer_len) <= 5
            and jaccard_characters(t.lower(), answer_lc) < 0.65
            and t.lower() not in STOPWORDS
            and not re.match(r'^\d+$', t)
        ]
        
        random.shuffle(candidates)
        
        unique = []
        for c in candidates:
            if len(unique) >= k:
                break
            if not any(jaccard_characters(u.lower(), c.lower()) >= 0.65 for u in unique):
                unique.append(c)
        
        return [title_case(u) for u in unique]
    
    def shuffle_and_label(arr):
        copy = arr[:]
        random.shuffle(copy)
        return [{'label': chr(65 + i), 'text': text} for i, text in enumerate(copy)]
    
    answer = target['answer']
    rx = re.compile(r'\b' + escape_regex(answer) + r'\b', re.I)
    
    if not rx.search(target['sentence']):
        return None
    
    prompt = rx.sub('_____', target['sentence'])
    if word_count(prompt) < 10:
        return None
    
    distractors = build_distractors(answer, top_pool, sentences, target['idx'], 3)
    if len(distractors) < 3:
        return None
    
    options_raw = shuffle_and_label([title_case(answer)] + distractors)
    correct_index = next(i for i, o in enumerate(options_raw) if o['text'].lower() == title_case(answer).lower())
    
    return {
        'type': 'cloze',
        'prompt': prompt,
        'options': [f"{o['label']}) {o['text']}" for o in options_raw],
        'correct_answer': correct_index
    }


def make_definition_question(target: Dict, sentences: List[str], top_pool: List[str]) -> Dict:
    """Make a definition question"""
    import random
    import re
    
    def title_case(s):
        return re.sub(r'\b([a-z])', lambda m: m.group(1).upper(), s)
    
    def escape_regex(s):
        return re.sub(r'[.*+?^${}()|[\]\\]', r'\\\g<0>', s)
    
    def definition_pattern(sentence, answer):
        rx = re.compile(rf'\b{escape_regex(answer)}\b\s+(is|are|refers to|means|denotes|involves|consists of|can be defined as)\s+([^.;:]+)', re.I)
        m = rx.search(sentence)
        if m:
            return f"{title_case(answer)} {m.group(1)} {m.group(2).strip()}."
        return None
    
    def nearest_sentences(sentences, idx, radius=1):
        res = []
        for i in range(max(0, idx - radius), min(len(sentences), idx + radius + 1)):
            if i != idx:
                res.append(sentences[i])
        return res
    
    def build_distractors(answer, pool, sentences, anchor_idx, k):
        # Simplified distractor generation
        return [f"{title_case(term)} is a related concept" for term in pool[:k]]
    
    def shuffle_and_label(arr):
        copy = arr[:]
        random.shuffle(copy)
        return [{'label': chr(65 + i), 'text': text} for i, text in enumerate(copy)]
    
    answer = title_case(target['answer'])
    cand_sentences = [target['sentence']] + nearest_sentences(sentences, target['idx'], 1)
    
    def_text = None
    for s in cand_sentences:
        def_text = definition_pattern(s, target['answer'])
        if def_text:
            break
    
    if not def_text:
        return None
    
    distractors = build_distractors(target['answer'], top_pool, sentences, target['idx'], 3)
    if len(distractors) < 3:
        return None
    
    options_raw = shuffle_and_label([def_text] + distractors)
    correct_index = next(i for i, o in enumerate(options_raw) if o['text'] == def_text)
    
    return {
        'type': 'definition',
        'prompt': f"Which option best defines {answer}?",
        'options': [f"{o['label']}) {o['text']}" for o in options_raw],
        'correct_answer': correct_index
    }


def make_true_about_question(target: Dict, sentences: List[str], top_pool: List[str]) -> Dict:
    """Make a true-about question"""
    import random
    import re
    
    def word_count(s):
        return len((s or '').strip().split())
    
    def title_case(s):
        return re.sub(r'\b([a-z])', lambda m: m.group(1).upper(), s)
    
    def escape_regex(s):
        return re.sub(r'[.*+?^${}()|[\]\\]', r'\\\g<0>', s)
    
    def build_distractors(answer, pool, sentences, anchor_idx, k):
        # Simplified distractor generation
        return [title_case(term) for term in pool[:k]]
    
    def shuffle_and_label(arr):
        copy = arr[:]
        random.shuffle(copy)
        return [{'label': chr(65 + i), 'text': text} for i, text in enumerate(copy)]
    
    answer = title_case(target['answer'])
    s = target['sentence']
    
    if word_count(s) < 10:
        return None
    
    if target['answer'] in s:
        true_stmt = re.sub(r'\b' + escape_regex(target['answer']) + r'\b', answer, s, flags=re.I)
    else:
        true_stmt = s
    
    false_stmts = []
    distractor_terms = build_distractors(target['answer'], top_pool, sentences, target['idx'], 6)
    
    for dt in distractor_terms[:6]:
        wrong = re.sub(r'\b' + escape_regex(answer) + r'\b', dt, true_stmt, flags=re.I)
        if wrong != true_stmt and word_count(wrong) >= 10:
            false_stmts.append(wrong)
        if len(false_stmts) >= 3:
            break
    
    if len(false_stmts) < 3:
        return None
    
    options_raw = shuffle_and_label([true_stmt] + false_stmts)
    correct_index = next(i for i, o in enumerate(options_raw) if o['text'] == true_stmt)
    
    return {
        'type': 'trueabout',
        'prompt': f"Which of the following is true about {answer}?",
        'options': [f"{o['label']}) {o['text']}" for o in options_raw],
        'correct_answer': correct_index
    }


# -------------------- PROPER FLASHCARDS ALGORITHM FROM REPOSITORY --------------------

_FLASHCARD_STOPWORDS = {
    "about", "above", "after", "again", "against", "along", "among", "because",
    "before", "being", "below", "between", "both", "cannot", "could", "does",
    "doing", "during", "each", "every", "first", "found", "from", "further",
    "given", "have", "having", "here", "into", "their", "there", "these",
    "they", "this", "those", "through", "under", "until", "using", "where",
    "which", "while", "with", "within", "without", "would", "other", "should",
    "shall", "might", "must", "such", "than", "that", "then", "also",
    "however", "therefore", "thus", "hence", "example", "examples", "called",
    "known", "used", "different", "various", "important", "when", "what",
    "will", "your", "them", "were", "been", "more", "most", "some", "only",
}


def _make_qa_card(sentence: str, context: str, explain_min: int, explain_max: int,
                  paragraph: Optional[int] = None) -> Dict:
    """Turn a key sentence into a proper question/answer flashcard.

    Prefers a fill-in-the-blank question (key term blanked out) with the
    term as the answer, plus a short explanation built from the surrounding
    context. Falls back to a statement card when no good term exists.
    """
    import re

    explanation = make_short_explanation_from_context(
        context, sentence, explain_min, explain_max
    )

    words = re.findall(r"[A-Za-z][A-Za-z\-']{3,}", sentence)
    candidates = [w for w in words if w.lower() not in _FLASHCARD_STOPWORDS]
    if candidates:
        # Most content-bearing word: longest (proper nouns tend to be longer)
        term = max(candidates, key=len)
        blanked = re.sub(
            r"\b" + re.escape(term) + r"\b", "_____", sentence,
            count=1, flags=re.IGNORECASE,
        )
        if "_____" in blanked:
            return {
                "summary": f"Fill in the blank: {blanked}",
                "explanation": f"Answer: {term}. {explanation}".strip(),
                "paragraph": paragraph,
            }

    # Fallback: statement card (sentence on the front, context on the back)
    return {
        "summary": sentence,
        "explanation": explanation,
        "paragraph": paragraph,
    }


def generate_flashcards_from_text(text: str, cards_per_paragraph: int = 1,
                                  fallback_cards_total: int = 5,
                                  explain_min: int = 25, explain_max: int = 55) -> List[Dict]:
    """Generate proper Q/A flashcards using the repository algorithm.

    1. Scores sentences per paragraph (centroid vectorizer) and keeps the
       most representative, non-redundant, non-banned ones.
    2. Turns each selected sentence into a real question/answer card
       (fill-in-the-blank + short context explanation).
    3. Tops up with globally-selected sentences when paragraph-level
       generation falls short of `fallback_cards_total`.
    """
    cleaned_text = clean_text(text)
    paragraphs = split_into_paragraphs(cleaned_text)
    sentences = split_into_sentences(cleaned_text)
    if not sentences:
        return []

    # Summary sentences are used to build the doc — don't quiz on them.
    # Never ban more than a quarter of the document: on small docs, banning a
    # fixed 12 starved every generator and returned zero flashcards.
    banned_n = min(12, len(sentences) // 4)
    banned = summarize_main_points(sentences, banned_n) if banned_n > 0 else []

    # Hard cap so a huge PDF can never explode the response
    overall_cap = min(60, max(fallback_cards_total, 4))

    cards: List[Dict] = []
    seen_summaries = set()

    # 1) Per-paragraph cards
    for i, paragraph in enumerate(paragraphs, start=1):
        if len(cards) >= overall_cap:
            break
        made = generate_paragraph_flashcards(
            paragraph, banned, max(1, cards_per_paragraph),
            explain_min, explain_max,
        )
        for c in made:
            if len(cards) >= overall_cap:
                break
            if c["summary"] in seen_summaries:
                continue
            seen_summaries.add(c["summary"])
            cards.append(_make_qa_card(
                c["summary"], paragraph, explain_min, explain_max, paragraph=i
            ))

    # 2) Global fallback to top up to `fallback_cards_total`
    if len(cards) < fallback_cards_total:
        global_cards = generate_global_fallback_flashcards(
            cleaned_text, banned,
            fallback_cards_total - len(cards) + 3,
            explain_min, explain_max,
        )
        for c in global_cards:
            if len(cards) >= fallback_cards_total:
                break
            if c["summary"] in seen_summaries:
                continue
            seen_summaries.add(c["summary"])
            cards.append(_make_qa_card(
                c["summary"], cleaned_text, explain_min, explain_max, paragraph=None
            ))

    # 3) Safety net: degenerate inputs (e.g. one sentence repeated) can be
    #    fully banned by the summary filter. Never return zero cards when the
    #    document has usable sentences — retry without the ban list.
    if not cards:
        global_cards = generate_global_fallback_flashcards(
            cleaned_text, [],
            max(3, min(overall_cap, fallback_cards_total)),
            explain_min, explain_max,
        )
        for c in global_cards:
            if len(cards) >= overall_cap:
                break
            if c["summary"] in seen_summaries:
                continue
            seen_summaries.add(c["summary"])
            cards.append(_make_qa_card(
                c["summary"], cleaned_text, explain_min, explain_max, paragraph=None
            ))

    return cards

