import os
import zipfile
import re
from collections import defaultdict
from lxml import etree
from tqdm import tqdm
import nltk
from nltk.stem.snowball import SnowballStemmer

# file management
from google.colab import drive
drive.mount('/content/drive', force_remount=True)

DRIVE_BASE = '/content/drive/My Drive/NLP'
WORKDIR    = '/content/data'
os.makedirs(WORKDIR, exist_ok=True)

cs_folder = os.path.join(WORKDIR, 'documents_cs')
if not os.path.isdir(cs_folder):
    archive = os.path.join(DRIVE_BASE, 'Archive.zip')
    if not os.path.isfile(archive):
        raise FileNotFoundError("Need documents_cs or Archive.zip in Drive")
    with zipfile.ZipFile(archive, 'r') as z:
        print("Unzipping Archive.zip …")
        z.extractall(WORKDIR)


# setup and download for the procedure
!pip install -q lxml tqdm nltk

nltk.download('stopwords', quiet=True)
XML_PARSER = etree.XMLParser(recover=True)
PUNCT_RE = re.compile(r"[^\wáčďéěíňóřšťúůýžäëïöüß]+", re.U)
stemmer_en = SnowballStemmer("english")
CZECH_SUFFIXES = [
    'ováním','ování','ověm','ových','ová','ové',
    'ami','ama','ech','ích','ům','em','ám','ách',
    'ou','y','i','a','e','o','u','í'
]


#  normalization
def normalize(text):
    if text is None:
        return []
    lower = text.lower()
    cleaned = PUNCT_RE.sub(' ', lower)
    tokens = cleaned.split()
    return [t for t in tokens if t]

# stemming
def stem_czech(token):
    for suf in CZECH_SUFFIXES:
        if token.endswith(suf) and len(token) > len(suf) + 2:
            return token[:-len(suf)]
    return token


def normalize_and_stem(text, lang):
    tokens = normalize(text)
    stemmed = []
    for t in tokens:
        if lang == 'en':
            stemmed.append(stemmer_en.stem(t))
        else:
            stemmed.append(stem_czech(t))
    return stemmed


#  Build inverted index
def build_index(collection_path, lang):
    index = defaultdict(set)

    file_names = os.listdir(collection_path)
    for fname in file_names:
        # only real XML documents, skip temp or junk
        if not fname.endswith('.xml') or fname.startswith('~$'):
            continue

        fullpath = os.path.join(collection_path, fname)
        if not os.path.isfile(fullpath):
            continue

        # try to parse; skip if completely invalid
        try:
            tree = etree.parse(fullpath, parser=XML_PARSER)
            root = tree.getroot()
            if root is None:
                continue
        except Exception:
            continue

        # find every <DOC> element
        for doc in root.findall('.//DOC'):
            docno = doc.findtext('DOCNO')
            if docno is None:
                continue

            # prefer TITLE+TEXT fields; fallback to all text
            title = doc.findtext('TITLE') or ''
            body  = doc.findtext('TEXT')  or ''
            if title or body:
                content = title + ' ' + body
            else:
                content = doc.xpath('string()')

            # normalize+stem and add docno to each term's postings
            terms = normalize_and_stem(content, lang)
            unique_terms = set(terms)
            for term in unique_terms:
                index[term].add(docno)

    # convert sets → sorted lists for postings
    final_index = {}
    for term, docs in index.items():
        final_index[term] = sorted(docs)
    return final_index


# Boolean operators on postings lists
def intersect(list1, list2):
    return sorted(set(list1) & set(list2))

def union(list1, list2):
    return sorted(set(list1) | set(list2))

def difference(list1, list2):
    return sorted(set(list1) - set(list2))


# Query parsing & execution
def parse_query(raw_query, lang):
    q = raw_query.strip()
    if ' AND NOT ' in q:
        op = 'AND NOT'
        left, right = q.split(' AND NOT ')
    elif ' AND ' in q:
        op = 'AND'
        left, right = q.split(' AND ')
    elif ' OR ' in q:
        op = 'OR'
        left, right = q.split(' OR ')
    else:
        op = None
        left, right = q, ''

    t1 = normalize_and_stem(left,  lang)
    t2 = normalize_and_stem(right, lang)
    return op, (t1[0] if t1 else ''), (t2[0] if t2 else '')


def run_boolean(query, index, lang):
    op, term1, term2 = parse_query(query, lang)
    postings1 = index.get(term1, [])
    postings2 = index.get(term2, [])

    if op == 'AND':
        return intersect(postings1, postings2)
    elif op == 'OR':
        return union(postings1, postings2)
    elif op == 'AND NOT':
        return difference(postings1, postings2)
    else:
        return postings1


# Load topics & qrels
def load_topics(filepath):
    tree = etree.parse(filepath, parser=XML_PARSER)
    topics = []
    for top in tree.findall('.//top'):
        qid   = top.findtext('num')
        query = top.findtext('query')
        if qid and query:
            topics.append((qid, query))
    return topics

def load_qrels(filepath):
    qrels = defaultdict(dict)
    with open(filepath, encoding='utf-8') as f:
        for line in f:
            qid, _, docno, rel = line.split()
            qrels[qid][docno] = int(rel)
    return qrels


# Run entire pipeline for one language
def run_and_evaluate(lang):
    base = WORKDIR
    doc_dir = os.path.join(base, f"documents_{lang}")
    topics = load_topics(os.path.join(base, f"topics-train_{lang}.xml"))
    qrels = load_qrels (os.path.join(base, f"qrels-train_{lang}.txt"))

    # build inverted index
    index = build_index(doc_dir, lang)

    # some stats for your submission form
    num_terms = len(index)
    total_postings = sum(len(plist) for plist in index.values())
    longest_term = max(index, key=lambda t: len(index[t]))
    max_df = len(index[longest_term])
    avg_df = total_postings / num_terms

    print("Index stats:")
    print("  Unique terms:", num_terms)
    print("  Total postings:", total_postings)
    print("  Top term by DF:", longest_term, "(DF=", max_df, ")")
    print(f"  Average DF: {avg_df:.2f}")

    # write results file
    outname = f"results-{lang}.dat"
    with open(outname, 'w', encoding='utf-8') as fout:
        for qid, qry in topics:
            hits = run_boolean(qry, index, lang)
            for docno in hits:
                fout.write(f"{qid} {docno}\n")
    print("Wrote:", outname)

    # evaluate precision & recall
    print("\nEvaluation:")
    Psum = 0.0
    Rsum = 0.0
    for qid, qry in topics:
        hits = run_boolean(qry, index, lang)
        relmap = qrels.get(qid, {})
        n_rel = sum(relmap.values())
        n_ret = len(hits)
        n_rel_ret = sum(1 for d in hits if relmap.get(d,0) == 1)

        P = (n_rel_ret / n_ret) if n_ret > 0 else 0.0
        R = (n_rel_ret / n_rel) if n_rel > 0 else 0.0
        Psum += P
        Rsum += R

        print(f" {qid}: ret={n_ret}, rel_ret={n_rel_ret}, P={P:.3f}, R={R:.3f}")

    m = len(topics)
    print(f" Average P: {Psum/m:.3f}, Average R: {Rsum/m:.3f}\n")


run_and_evaluate('cs')
run_and_evaluate('en')