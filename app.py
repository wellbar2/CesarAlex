import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib_venn import venn3
import re
import time
import csv
import hashlib
import unicodedata


# =========================
# CONSTANTES (ajustes internos)
# =========================
OPENALEX_DOI_LOOKUP_TIMEOUT = 20          # segundos
OPENALEX_DOI_LOOKUP_RATE_LIMIT = 0.3      # pausa entre requisições por DOI (segundos)
OPENALEX_ORCID_RATE_LIMIT = 1.0           # pausa após consulta OpenAlex por ORCID (segundos)
OPENALEX_PER_PAGE = 200                   # tamanho de página na OpenAlex (máx 200)
OPENALEX_MAX_PAGES = 25                   # limite de páginas para evitar loops (200*25=5000 works)

# Heurística de match (nomeLattes x autores OpenAlex do work)
ENFORCE_OA_TOKENS_SUBSET_OF_LATTES = True
MIN_COMMON_INITIALS_FALLBACK = 2


# =========================
# STREAMLIT CONFIG
# =========================
st.set_page_config(
    page_title="Plataforma CesarAlex",
    page_icon=None,
    layout="centered",
    initial_sidebar_state="auto"
)

st.title("Integração de Currículo Lattes, ORCID e OpenAlex")
st.write(
    """
Este aplicativo recebe arquivos de currículo Lattes (HTML) e, para cada um:
- Identifica se o ORCID está Lattes;
- Consulta a API do ORCID para obter publicações;
- Consulta a API da OpenAlex usando o ORCID para obter publicações (DOIs + ISSN-L);
- Extrai DOIs diretamente do Lattes quando existe link na publicação;
- Consolida as publicações por DOI e permite download em CSV: por pesquisador e geral;
- Apresenta um gráfico presença ORCID/OpenAlex e um diagrama de cobertura multibase.
"""
)


# =========================
# SIDEBAR / OPTIONS
# =========================
st.sidebar.header("Opções")

check_openalex_without_orcid = st.sidebar.checkbox(
    "Verificar OpenAlex mesmo sem ORCID (por DOI)",
    value=False,
    help="Se ativado, pode demorar muito mais porque o sistema consulta a OpenAlex para cada DOI individualmente."
)


# =========================
# HELPERS
# =========================

def normalize_doi(doi: str):
    """Normaliza DOI para comparação: remove prefixos e coloca em minúsculo."""
    if not doi:
        return None
    doi = doi.strip().lower()
    if doi in {"sem doi", "-", "na", "n/a"}:
        return None
    for prefix in (
        "http://dx.doi.org/",
        "https://dx.doi.org/",
        "https://doi.org/",
        "http://doi.org/",
        "doi:",
    ):
        if doi.startswith(prefix):
            doi = doi.replace(prefix, "", 1).strip()
    return doi or None


def extract_orcid_from_html(content: str):
    """Extrai o primeiro link contendo 'orcid.org' do HTML do Lattes."""
    soup = BeautifulSoup(content, "html.parser")
    for link in soup.find_all("a", href=True):
        href = link["href"]
        if "orcid.org" in href:
            return href.strip()
    return None


def extract_nome_lattes_from_html(content: str) -> str:
    """
    Extrai o nome completo do pesquisador a partir do HTML do Lattes:
    <h2 class="nome" tabindex="0">Laísa Soares Cardoso</h2>
    """
    try:
        soup = BeautifulSoup(content, "html.parser")
        h2 = soup.find("h2", class_="nome")
        if h2:
            nome = h2.get_text(" ", strip=True)
            return nome if nome else "Não disponível"
    except Exception:
        pass
    return "Não disponível"


def extract_dois_from_lattes(content: str):
    """
    Extrai DOIs do HTML do Lattes procurando links com classe "icone-producao icone-doi".
    Retorna set de DOIs normalizados.
    """
    soup = BeautifulSoup(content, "html.parser")
    dois = set()
    for link in soup.find_all("a", class_="icone-producao icone-doi"):
        href = link.get("href", "")
        doi_norm = normalize_doi(href)
        if doi_norm:
            dois.add(doi_norm)
    return dois


def extract_articles_info_from_lattes(content: str):
    """
    Extrai informações do item (texto do bloco) e tenta achar um ano (19xx/20xx).
    Retorna dict: {doi: {"Título": ..., "Ano": ...}}
    """
    soup = BeautifulSoup(content, "html.parser")
    articles = {}
    for link in soup.find_all("a", class_="icone-producao icone-doi"):
        href = link.get("href", "")
        doi_norm = normalize_doi(href)
        if not doi_norm:
            continue

        parent = link.find_parent()
        parent_text = parent.get_text(" ", strip=True) if parent else ""
        match = re.search(r"\b(19|20)\d{2}\b", parent_text)
        year = match.group(0) if match else "Não disponível"
        title = parent_text if parent_text else "Não disponível"

        articles[doi_norm] = {"Título": title, "Ano": year}
    return articles


def get_publications_from_orcid(orcid_link: str):
    """Consulta ORCID API e retorna lista de publicações com Título, Ano, DOI(normalizado)."""
    if not orcid_link:
        return []

    orcid_id = (
        orcid_link.replace("https://orcid.org/", "")
        .replace("http://orcid.org/", "")
        .strip()
    )
    if not orcid_id:
        return []

    api_url = f"https://pub.orcid.org/v3.0/{orcid_id}/works"
    headers = {"Accept": "application/json"}

    try:
        resp = requests.get(api_url, headers=headers, timeout=30)
    except Exception:
        st.error(f"Falha ao acessar ORCID API para {orcid_id}.")
        return []

    if resp.status_code != 200:
        st.error(f"Erro ORCID API para {orcid_id}. Código: {resp.status_code}")
        return []

    data = resp.json()
    pubs = []

    for group in data.get("group", []):
        for ws in group.get("work-summary", []):
            title = "Sem título"
            title_data = ws.get("title")
            if isinstance(title_data, dict):
                inner = title_data.get("title")
                if isinstance(inner, dict):
                    title = inner.get("value", "Sem título") or "Sem título"

            year = "Sem data"
            pub_date = ws.get("publication-date")
            if isinstance(pub_date, dict):
                y = pub_date.get("year")
                if isinstance(y, dict):
                    year = y.get("value", "Sem data") or "Sem data"

            doi_norm = None
            external_ids = ws.get("external-ids")
            if isinstance(external_ids, dict):
                for ext in external_ids.get("external-id", []):
                    if isinstance(ext, dict) and ext.get("external-id-type", "").lower() == "doi":
                        doi_norm = normalize_doi(ext.get("external-id-value", ""))
                        break

            pubs.append({"Título": title, "Ano": year, "DOI": doi_norm})

    return pubs


def get_publications_from_openalex_by_orcid(orcid_link: str):
    """
    Consulta OpenAlex por ORCID e retorna lista com Título, Ano, DOI(normalizado), ISSN-L.
    Implementa paginação (cursor) para não truncar em 200 works.
    """
    if not orcid_link:
        return []

    orcid_id = (
        orcid_link.replace("https://orcid.org/", "")
        .replace("http://orcid.org/", "")
        .strip()
    )
    if not orcid_id:
        return []

    pubs = []
    cursor = "*"
    pages = 0

    while pages < OPENALEX_MAX_PAGES:
        url = (
            "https://api.openalex.org/works"
            f"?filter=authorships.author.orcid:{orcid_id}"
            f"&per_page={OPENALEX_PER_PAGE}"
            f"&cursor={cursor}"
        )

        try:
            resp = requests.get(url, timeout=30)
        except Exception:
            st.error(f"Falha ao acessar OpenAlex para ORCID {orcid_id}.")
            break

        if resp.status_code != 200:
            st.error(f"Erro OpenAlex para ORCID {orcid_id}. Código: {resp.status_code}")
            break

        data = resp.json()
        results = data.get("results", [])
        if not results:
            break

        for work in results:
            doi_norm = normalize_doi(work.get("doi")) if work.get("doi") else None
            title = work.get("display_name", "Sem título") or "Sem título"
            year = work.get("publication_year", "Sem data")

            issn_l = "Sem ISSN-L"
            primary_location = work.get("primary_location")
            if isinstance(primary_location, dict):
                source = primary_location.get("source")
                if isinstance(source, dict):
                    issn_l = source.get("issn_l", "Sem ISSN-L") or "Sem ISSN-L"

            pubs.append({"Título": title, "Ano": year, "DOI": doi_norm, "ISSN-L": issn_l})

        meta = data.get("meta", {})
        next_cursor = meta.get("next_cursor")
        if not next_cursor:
            break

        cursor = next_cursor
        pages += 1

    return pubs


# =========================
# Heurística (nomeLattes x autores do work OpenAlex)
# =========================

RE_WORD = re.compile(r"[A-Za-zÀ-ÖØ-öø-ÿ]+", flags=re.UNICODE)
RE_INITIAL = re.compile(r"\b([A-Za-zÀ-ÖØ-öø-ÿ])\.\b|\b([A-Za-zÀ-ÖØ-öø-ÿ])\b", flags=re.UNICODE)

STOP_TOKENS = {
    "de", "da", "do", "das", "dos",
    "del", "la", "las", "los", "el",
    "y", "e",
    "van", "von",
}

def strip_accents(s: str) -> str:
    s = unicodedata.normalize("NFKD", s or "")
    return "".join(ch for ch in s if not unicodedata.combining(ch))

def norm_text(s: str) -> str:
    s = strip_accents(s or "")
    s = s.casefold()

    # trata nomes com hífen (e variações unicode) como separador de tokens
    s = re.sub(r"[-‐-‒–—―]+", " ", s)

    # mantém normalização de espaços
    s = re.sub(r"\s+", " ", s).strip()
    return s


def tokens_nome(nome: str) -> list[str]:
    nome_n = norm_text(nome)
    toks = [t for t in RE_WORD.findall(nome_n)]
    toks = [t for t in toks if len(t) >= 2]
    toks = [t for t in toks if t not in STOP_TOKENS]
    return toks

def initials_from_name(nome: str) -> set[str]:
    nome_n = norm_text(nome)
    toks = tokens_nome(nome_n)

    inits = {t[0] for t in toks if t}
    for m in RE_INITIAL.finditer(nome_n):
        g = m.group(1) or m.group(2)
        if g:
            inits.add(norm_text(g)[0])
    return inits


def overlap_score(nome_lattes: str, nome_oa: str) -> int:
    a = set(tokens_nome(nome_lattes))
    b = set(tokens_nome(nome_oa))
    return len(a.intersection(b))

def fallback_1token_2initials(nome_lattes: str, nome_oa: str) -> tuple[bool, int]:
    lt = set(tokens_nome(nome_lattes))
    ot = set(tokens_nome(nome_oa))
    if not lt or not ot:
        return False, 0

    common_tokens = lt.intersection(ot)
    if len(common_tokens) < 1:
        return False, 0

    li = initials_from_name(nome_lattes)
    oi = initials_from_name(nome_oa)
    if len(li.intersection(oi)) < MIN_COMMON_INITIALS_FALLBACK:
        return False, 0

    return True, len(common_tokens)

def oa_tokens_subset_of_lattes(nome_lattes: str, nome_oa: str) -> bool:
    ls = set(tokens_nome(nome_lattes))
    os = set(tokens_nome(nome_oa))
    if not os:
        return True
    return os.issubset(ls)

def _apply_subset_filter(nome_lattes: str, matches, enabled: bool):
    if not enabled or not matches:
        return matches
    out = []
    for m in matches:
        oa_name = m[1]
        # 1) tokens completos do OA devem estar no Lattes (opcional)
        if not oa_tokens_subset_of_lattes(nome_lattes, oa_name):
            continue
        out.append(m)
    return out


def find_all_matches_nome(nome_lattes: str, autores: list[dict], min_common_tokens: int = 2):
    # Regra principal: tokens>=2
    best_by_id = {}

    for a in autores:
        oa_nome = (a.get("nome") or "").strip()
        oa_id = (a.get("id_autor") or "").strip()
        if not oa_nome or not oa_id:
            continue

        sc = overlap_score(nome_lattes, oa_nome)
        if sc < min_common_tokens:
            continue

        if oa_id not in best_by_id:
            best_by_id[oa_id] = (oa_nome, sc, "tokens>=2")
        else:
            nome_old, sc_old, _ = best_by_id[oa_id]
            if sc > sc_old or (sc == sc_old and len(oa_nome) < len(nome_old)):
                best_by_id[oa_id] = (oa_nome, sc, "tokens>=2")

    matches = [(oid, onome, osc, met) for oid, (onome, osc, met) in best_by_id.items()]
    matches.sort(key=lambda x: (-x[2], len(x[1]), norm_text(x[1]), x[0]))
    matches = _apply_subset_filter(nome_lattes, matches, ENFORCE_OA_TOKENS_SUBSET_OF_LATTES)

    if matches:
        return matches

    # Fallback: 1 token + 2 iniciais
    fb_by_id = {}
    for a in autores:
        oa_nome = (a.get("nome") or "").strip()
        oa_id = (a.get("id_autor") or "").strip()
        if not oa_nome or not oa_id:
            continue

        ok, sc_fb = fallback_1token_2initials(nome_lattes, oa_nome)
        if ok:
            if oa_id not in fb_by_id:
                fb_by_id[oa_id] = (oa_nome, sc_fb, "fallback_1token+2iniciais")
            else:
                nome_old, sc_old, _ = fb_by_id[oa_id]
                if sc_fb > sc_old or (sc_fb == sc_old and len(oa_nome) < len(nome_old)):
                    fb_by_id[oa_id] = (oa_nome, sc_fb, "fallback_1token+2iniciais")

    fb = [(oid, onome, osc, met) for oid, (onome, osc, met) in fb_by_id.items()]
    fb.sort(key=lambda x: (-x[2], len(x[1]), norm_text(x[1]), x[0]))
    fb = _apply_subset_filter(nome_lattes, fb, ENFORCE_OA_TOKENS_SUBSET_OF_LATTES)
    return fb

def join_matches_for_cells(matches):
    """
    Retorna:
      pairs_str: "A...|Nome; A...|Nome"
      ids_str: "A...; A..."
      nomes_str: "Nome; Nome"
      metodos_str: "tokens>=2; fallback..."
      scoremax: int
    """
    if not matches:
        return "-", "-", "-", "-", 0
    pairs = [f"{m[0]}|{m[1]}" for m in matches]
    ids = [m[0] for m in matches]
    nomes = [m[1] for m in matches]
    metodos = [m[3] for m in matches]
    scoremax = max(m[2] for m in matches)
    return "; ".join(pairs), "; ".join(ids), "; ".join(nomes), "; ".join(metodos), scoremax


# =========================
# OpenAlex lookups (com autores)
# =========================

@st.cache_data(show_spinner=False, ttl=60 * 60)
def openalex_author_lookup(author_id: str):
    """
    Busca detalhes do autor na OpenAlex (para obter ORCID quando não vem no authorship).
    Retorna dict: {"found": bool, "orcid": str|None, "display_name": str|None}
    """
    if not author_id:
        return {"found": False, "orcid": None, "display_name": None}

    url = f"https://api.openalex.org/authors/{author_id.strip()}"
    try:
        resp = requests.get(url, timeout=OPENALEX_DOI_LOOKUP_TIMEOUT)
    except Exception:
        return {"found": False, "orcid": None, "display_name": None}

    if resp.status_code != 200:
        return {"found": False, "orcid": None, "display_name": None}

    data = resp.json()
    return {
        "found": True,
        "orcid": data.get("orcid"),
        "display_name": data.get("display_name"),
    }


@st.cache_data(show_spinner=False, ttl=60 * 60)
def openalex_lookup_by_doi(doi_norm: str):
    """
    Consulta OpenAlex por DOI (normalizado). Cacheado para acelerar quando DOI repete.
    Agora também retorna autores para aplicar heurística.
    """
    if not doi_norm:
        return {
            "found": False,
            "title": None,
            "year": None,
            "issn_l": None,
            "authors": [],
        }

    url = f"https://api.openalex.org/works/http://dx.doi.org/{doi_norm}"

    try:
        resp = requests.get(url, timeout=OPENALEX_DOI_LOOKUP_TIMEOUT)
    except Exception:
        return {"found": False, "title": None, "year": None, "issn_l": None, "authors": []}

    if resp.status_code != 200:
        return {"found": False, "title": None, "year": None, "issn_l": None, "authors": []}

    data = resp.json()
    title = data.get("display_name")
    year = data.get("publication_year")

    issn_l = None
    primary_location = data.get("primary_location")
    if isinstance(primary_location, dict):
        source = primary_location.get("source")
        if isinstance(source, dict):
            issn_l = source.get("issn_l")

    # Autores (para heurística)
    authors = []
    authorships = data.get("authorships", [])
    if isinstance(authorships, list):
        for au in authorships:
            if not isinstance(au, dict):
                continue
            a = au.get("author")
            if not isinstance(a, dict):
                continue
            aid = (a.get("id") or "").strip()
            # "id" às vezes vem como URL completa
            if aid.startswith("https://openalex.org/"):
                aid = aid.replace("https://openalex.org/", "").strip()
            nome = (a.get("display_name") or "").strip()
            orcid = (a.get("orcid") or "").strip() if a.get("orcid") else None
            if aid and nome:
                authors.append({"id_autor": aid, "nome": nome, "orcid": orcid})

    return {"found": True, "title": title, "year": year, "issn_l": issn_l, "authors": authors}


def dataframe_to_csv_bytes(df: pd.DataFrame) -> bytes:
    """
    Converte DataFrame para CSV com:
    - separador ;
    - aspas duplas em TODOS os campos
    - encoding UTF-8
    """
    return df.to_csv(
        index=False,
        sep=";",
        quoting=csv.QUOTE_ALL,
        encoding="utf-8"
    ).encode("utf-8")


def files_fingerprint(uploaded_files) -> str:
    """
    Gera uma assinatura estável dos arquivos enviados.
    Usa nome + tamanho + hash do conteúdo.
    """
    h = hashlib.sha256()
    for f in uploaded_files:
        data = f.getvalue()  # bytes
        h.update(f.name.encode("utf-8"))
        h.update(str(len(data)).encode("utf-8"))
        h.update(hashlib.sha256(data).digest())
    return h.hexdigest()


def init_state():
    st.session_state.setdefault("fingerprint", None)
    st.session_state.setdefault("processed", False)
    st.session_state.setdefault("results", None)


def make_arrow_friendly(df: pd.DataFrame) -> pd.DataFrame:
    """
    Streamlit usa Arrow internamente para exibir dataframes.
    Se uma coluna mistura tipos (ex: int e str), pode gerar warning/erro.
    Aqui padronizamos as colunas problemáticas para string somente para EXIBIÇÃO.
    """
    if df is None or df.empty:
        return df

    df2 = df.copy()

    # colunas que costumam misturar tipos
    for col in ["Ano", "Presente no Lattes", "Presente no ORCID", "Presente na OpenAlex", "MatchScoreMax"]:
        if col in df2.columns:
            df2[col] = df2[col].astype(str)

    # colunas textuais (garante não virar NaN/float)
    for col in [
        "Arquivo", "ORCID", "DOI", "Título", "ISSN-L", "Motivo", "Exemplos_DOI_OpenAlex",
        "nomeLattes", "MatchPairs_OpenAlex", "IdsOpenAlex_Provaveis", "NomesOpenAlex_Provaveis",
        "ORCIDs_Provaveis", "MetodoMatch"
    ]:
        if col in df2.columns:
            df2[col] = df2[col].fillna("").astype(str)

    return df2


# ===== VENN (como contarOverlapPorPeriodo) - por observação =====

def venn_category_for_row(presente_lattes, presente_orcid, presente_openalex):
    l = bool(presente_lattes)
    o = bool(presente_orcid)
    a = bool(presente_openalex)

    if l and o and a:
        return "Lattes_ORCID_OpenAlex"
    if l and o and not a:
        return "Lattes_ORCID_Apenas"
    if l and not o and a:
        return "Lattes_OpenAlex_Apenas"
    if not l and o and a:
        return "ORCID_OpenAlex_Apenas"
    if l and not o and not a:
        return "Lattes_Apenas"
    if not l and o and not a:
        return "ORCID_Apenas"
    if not l and not o and a:
        return "OpenAlex_Apenas"
    return None


def compute_venn_counts_from_observations(all_report_rows):
    counts = {
        "Lattes_Apenas": 0,
        "ORCID_Apenas": 0,
        "OpenAlex_Apenas": 0,
        "Lattes_ORCID_Apenas": 0,
        "Lattes_OpenAlex_Apenas": 0,
        "ORCID_OpenAlex_Apenas": 0,
        "Lattes_ORCID_OpenAlex": 0,
    }
    for r in all_report_rows:
        cat = venn_category_for_row(
            r.get("Presente no Lattes", False),
            r.get("Presente no ORCID", False),
            r.get("Presente na OpenAlex", False),
        )
        if cat:
            counts[cat] += 1
    total = sum(counts.values())
    return counts, total


def create_venn_like_contarOverlap(counts, total, title):
    subsets = (
        counts["Lattes_Apenas"],            # 100
        counts["ORCID_Apenas"],             # 010
        counts["Lattes_ORCID_Apenas"],      # 110
        counts["OpenAlex_Apenas"],          # 001
        counts["Lattes_OpenAlex_Apenas"],   # 101
        counts["ORCID_OpenAlex_Apenas"],    # 011
        counts["Lattes_ORCID_OpenAlex"],    # 111
    )

    cor_lattes = "#009B3A"
    cor_orcid = "#6A8EAE"
    cor_openalex = "#D47A70"
    alpha = 0.6

    region_ids = ["100", "010", "110", "001", "101", "011", "111"]

    def _region_colors_from_dummy():
        tmp = plt.figure(figsize=(1, 1))
        vtmp = venn3(
            subsets=(1, 1, 1, 1, 1, 1, 1),
            set_labels=("Lattes", "ORCID", "OpenAlex"),
            alpha=alpha,
            set_colors=(cor_lattes, cor_orcid, cor_openalex),
        )
        colors = {}
        for rid in region_ids:
            p = vtmp.get_patch_by_id(rid)
            if p is not None:
                colors[rid] = p.get_facecolor()
        plt.close(tmp)
        return colors

    region_colors = _region_colors_from_dummy()

    fig = plt.figure(figsize=(8, 8))
    v = venn3(
        subsets=subsets,
        set_labels=("Lattes", "ORCID", "OpenAlex"),
        alpha=alpha,
        set_colors=(cor_lattes, cor_orcid, cor_openalex),
    )

    region_counts = dict(zip(region_ids, subsets))

    for rid in region_ids:
        lbl = v.get_label_by_id(rid)
        if lbl:
            n = region_counts[rid]
            pct = (n / total) * 100 if total else 0.0
            lbl.set_text(f"{n}\n({pct:.2f}%)")
            lbl.set_fontsize(14)

    for lab in v.set_labels:
        if lab:
            lab.set_fontweight("bold")
            lab.set_fontsize(16)

    legend_items = [
        ("100", "Somente Lattes"),
        ("010", "Somente ORCID"),
        ("001", "Somente OpenAlex"),
        ("110", "Lattes ∩ ORCID"),
        ("101", "Lattes ∩ OpenAlex"),
        ("011", "ORCID ∩ OpenAlex"),
        ("111", "Lattes ∩ ORCID ∩ OpenAlex"),
    ]

    handles = []
    for rid, label in legend_items:
        fc = region_colors.get(rid, (0.8, 0.8, 0.8, alpha))
        handles.append(mpatches.Patch(facecolor=fc, edgecolor="black", label=label))

    plt.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.08),
        ncol=2,
        frameon=True,
        title="Regiões do diagrama"
    )

    plt.title(title, fontsize=18)
    plt.tight_layout()
    return fig


# =========================
# MAIN
# =========================
uploaded_files = st.file_uploader(
    "Selecione os arquivos de currículo Lattes (HTML)",
    accept_multiple_files=True
)

init_state()

current_fp = files_fingerprint(uploaded_files) if uploaded_files else None

# Se os arquivos mudaram, invalida o processamento anterior
if current_fp and st.session_state["fingerprint"] != current_fp:
    st.session_state["fingerprint"] = current_fp
    st.session_state["processed"] = False
    st.session_state["results"] = None

process_clicked = st.button(
    "Processar currículos",
    type="primary",
    disabled=(not uploaded_files)
)

if uploaded_files and not st.session_state["processed"]:
    st.info("Envie os currículos e clique em **Processar currículos** para iniciar o processamento.")


# =========================
# PROCESSAMENTO (roda 1x)
# =========================
if uploaded_files and process_clicked and not st.session_state["processed"]:
    # Contadores para Sankey
    files_with_orcid = 0
    files_without_orcid = 0
    files_with_openalex_count = 0
    files_without_openalex_count = 0

    # Para avisar pesquisadores: "tem OpenAlex mas não tem ORCID no Lattes"
    no_orcid_but_openalex_files = []
    no_orcid_but_openalex_rows = []

    all_report = []
    per_file_df = {}

    file_progress = st.progress(0)
    status = st.empty()
    total_files = len(uploaded_files)

    for file_idx, uploaded_file in enumerate(uploaded_files, start=1):
        status.write(f"Processando {file_idx}/{total_files}: **{uploaded_file.name}**")
        st.markdown(f"---\n### Processando: **{uploaded_file.name}**")

        try:
            content = uploaded_file.getvalue().decode("utf-8")
        except Exception as e:
            st.error(f"Erro ao ler {uploaded_file.name}: {e}")
            file_progress.progress(int((file_idx / total_files) * 100))
            continue

        # Nome confiável do Lattes (para heurística)
        nomeLattes = extract_nome_lattes_from_html(content)

        # Extração Lattes
        lattes_dois = extract_dois_from_lattes(content)
        lattes_articles = extract_articles_info_from_lattes(content)

        # ORCID
        orcid_link = extract_orcid_from_html(content)

        # =========================
        # CASO: SEM ORCID NO HTML
        # =========================
        if not orcid_link:
            st.warning("ORCID não encontrado no HTML do Lattes.")
            files_without_orcid += 1

            consolidated_rows = []
            found_in_openalex = []

            doi_progress = None
            doi_status = None

            dois_list = sorted(lattes_dois)
            n_dois = len(dois_list)

            if check_openalex_without_orcid and n_dois > 0:
                doi_progress = st.progress(0)
                doi_status = st.empty()

            for i, doi in enumerate(dois_list, start=1):
                info = lattes_articles.get(doi, {"Título": "Não disponível", "Ano": "Não disponível"})

                presente_openalex = False
                issn_l = "Não disponível"
                title_oa = None
                year_oa = None

                # Campos novos (heurística)
                match_pairs = "-"
                ids_prob = "-"
                nomes_prob = "-"
                orcids_prob = "-"
                metodo_match = "-"
                score_max = 0

                if check_openalex_without_orcid:
                    lookup = openalex_lookup_by_doi(doi)
                    presente_openalex = bool(lookup.get("found", False))

                    if presente_openalex:
                        found_in_openalex.append(doi)
                        issn_l = lookup.get("issn_l") or "Sem ISSN-L"
                        title_oa = lookup.get("title")
                        year_oa = lookup.get("year")

                        # ---- heurística para identificar o autor dono do currículo ----
                        autores_work = lookup.get("authors", []) or []
                        matches = find_all_matches_nome(nomeLattes, autores_work, min_common_tokens=2)

                        match_pairs, ids_prob, nomes_prob, metodo_match, score_max = join_matches_for_cells(matches)

                        # ORCID(s) provável(is) para os ids encontrados
                        orcid_list = []
                        if matches:
                            for (aid, anome, _, _) in matches:
                                # 1) tenta ORCID já vindo no authorship
                                orcid_direct = None
                                for aw in autores_work:
                                    if (aw.get("id_autor") or "").strip() == aid:
                                        orcid_direct = aw.get("orcid")
                                        break
                                if orcid_direct:
                                    orcid_list.append(orcid_direct)
                                else:
                                    # 2) fallback: consulta /authors/A...
                                    au = openalex_author_lookup(aid)
                                    orcid_list.append(au.get("orcid") or "-")
                                time.sleep(OPENALEX_DOI_LOOKUP_RATE_LIMIT)
                        orcids_prob = "; ".join(orcid_list) if orcid_list else "-"

                    if doi_progress and doi_status:
                        doi_status.write(f"Consultando OpenAlex por DOI: {i}/{n_dois}")
                        doi_progress.progress(int((i / n_dois) * 100))

                    time.sleep(OPENALEX_DOI_LOOKUP_RATE_LIMIT)

                # Prioridade Título/Ano (sem ORCID): Lattes > OpenAlex(lookup)
                title_final = info.get("Título", "Não disponível")
                year_final = info.get("Ano", "Não disponível")
                if (title_final in ["Não disponível", "", None]) and title_oa:
                    title_final = title_oa
                if (year_final in ["Não disponível", "", None]) and year_oa:
                    year_final = year_oa

                consolidated_rows.append({
                    "Arquivo": uploaded_file.name,
                    "nomeLattes": nomeLattes,
                    "ORCID": "Não disponível",
                    "DOI": doi,
                    "Título": title_final,
                    "Ano": year_final,
                    "Presente no Lattes": True,
                    "Presente no ORCID": False,
                    "Presente na OpenAlex": presente_openalex,
                    "ISSN-L": issn_l,

                    # Novos campos (heurística)
                    "MatchPairs_OpenAlex": match_pairs,
                    "IdsOpenAlex_Provaveis": ids_prob,
                    "NomesOpenAlex_Provaveis": nomes_prob,
                    "ORCIDs_Provaveis": orcids_prob,
                    "MetodoMatch": metodo_match,
                    "MatchScoreMax": score_max,
                })

            if doi_progress:
                doi_progress.empty()
            if doi_status:
                doi_status.empty()

            df_file = pd.DataFrame(consolidated_rows)
            per_file_df[uploaded_file.name] = df_file
            all_report.extend(consolidated_rows)

            if check_openalex_without_orcid and len(found_in_openalex) > 0:
                sample = found_in_openalex[:10]
                st.info(
                    f"**Sem ORCID no Lattes, mas com presença na OpenAlex via DOI**: "
                    f"{len(found_in_openalex)} DOI(s) encontrado(s)."
                )
                no_orcid_but_openalex_files.append({
                    "Arquivo": uploaded_file.name,
                    "nomeLattes": nomeLattes,
                    "Qtd_DOIs_Lattes": len(lattes_dois),
                    "Qtd_DOIs_encontrados_OpenAlex": len(found_in_openalex),
                    "Exemplos_DOI_OpenAlex": "; ".join(sample)
                })

                # Linhas detalhadas (agora enriquecidas com heurística)
                for doi in found_in_openalex:
                    # tenta recuperar a linha já construída (para reaproveitar campos de heurística)
                    linha = next((r for r in consolidated_rows if r.get("DOI") == doi), None)
                    no_orcid_but_openalex_rows.append({
                        "Arquivo": uploaded_file.name,
                        "nomeLattes": nomeLattes,
                        "DOI": doi,
                        "Motivo": "OpenAlex encontrado por DOI, mas ORCID ausente no HTML do Lattes",

                        "MatchPairs_OpenAlex": (linha.get("MatchPairs_OpenAlex") if linha else "-"),
                        "IdsOpenAlex_Provaveis": (linha.get("IdsOpenAlex_Provaveis") if linha else "-"),
                        "NomesOpenAlex_Provaveis": (linha.get("NomesOpenAlex_Provaveis") if linha else "-"),
                        "ORCIDs_Provaveis": (linha.get("ORCIDs_Provaveis") if linha else "-"),
                        "MetodoMatch": (linha.get("MetodoMatch") if linha else "-"),
                        "MatchScoreMax": (linha.get("MatchScoreMax") if linha else 0),
                    })

            file_progress.progress(int((file_idx / total_files) * 100))
            continue

        # =========================
        # CASO: COM ORCID
        # =========================
        st.success(f"ORCID encontrado: {orcid_link}")
        files_with_orcid += 1

        # ORCID API
        orcid_pubs = get_publications_from_orcid(orcid_link)
        orcid_dict = {p["DOI"]: p for p in orcid_pubs if p.get("DOI")}

        # OpenAlex API por ORCID (paginado)
        openalex_pubs = get_publications_from_openalex_by_orcid(orcid_link)
        time.sleep(OPENALEX_ORCID_RATE_LIMIT)
        openalex_dict = {p["DOI"]: p for p in openalex_pubs if p.get("DOI")}

        if openalex_pubs:
            files_with_openalex_count += 1
        else:
            files_without_openalex_count += 1

        # União de DOIs
        union_dois = set(lattes_dois) | set(orcid_dict.keys()) | set(openalex_dict.keys())
        union_dois_list = sorted(union_dois)
        n_union = len(union_dois_list)

        # ✅ Barra de progresso para consulta por DOI (somente se checkbox estiver ativo)
        doi_progress = None
        doi_status = None
        if check_openalex_without_orcid and n_union > 0:
            doi_progress = st.progress(0)
            doi_status = st.empty()

        consolidated_rows = []
        for i, doi in enumerate(union_dois_list, start=1):
            row = {"Arquivo": uploaded_file.name, "nomeLattes": nomeLattes, "ORCID": orcid_link, "DOI": doi}

            # prioridade Título/Ano: ORCID > Lattes > OpenAlex
            if doi in orcid_dict:
                row["Título"] = orcid_dict[doi].get("Título", "Sem título")
                row["Ano"] = orcid_dict[doi].get("Ano", "Sem data")
                row["Presente no ORCID"] = True
            elif doi in lattes_articles:
                row["Título"] = lattes_articles[doi].get("Título", "Não disponível")
                row["Ano"] = lattes_articles[doi].get("Ano", "Não disponível")
                row["Presente no ORCID"] = False
            elif doi in openalex_dict:
                row["Título"] = openalex_dict[doi].get("Título", "Sem título")
                row["Ano"] = openalex_dict[doi].get("Ano", "Sem data")
                row["Presente no ORCID"] = False
            else:
                row["Título"] = "Não disponível"
                row["Ano"] = "Não disponível"
                row["Presente no ORCID"] = False

            row["Presente no Lattes"] = (doi in lattes_dois)
            row["Presente na OpenAlex"] = (doi in openalex_dict)
            row["ISSN-L"] = openalex_dict[doi].get("ISSN-L", "Sem ISSN-L") if doi in openalex_dict else "Não disponível"

            # Defaults da heurística (homogêneo)
            row["MatchPairs_OpenAlex"] = "-"
            row["IdsOpenAlex_Provaveis"] = "-"
            row["NomesOpenAlex_Provaveis"] = "-"
            row["ORCIDs_Provaveis"] = "-"
            row["MetodoMatch"] = "-"
            row["MatchScoreMax"] = 0

            # ✅ Heurística (também para quem TEM ORCID), só se checkbox estiver ativo
            if check_openalex_without_orcid and doi:
                lookup = openalex_lookup_by_doi(doi)
                if lookup.get("found", False):
                    autores_work = lookup.get("authors", []) or []
                    matches = find_all_matches_nome(nomeLattes, autores_work, min_common_tokens=2)

                    match_pairs, ids_prob, nomes_prob, metodo_match, score_max = join_matches_for_cells(matches)

                    # ORCID(s) provável(is) para os ids encontrados
                    orcid_list = []
                    if matches:
                        for (aid, _, _, _) in matches:
                            # 1) ORCID direto (authorships)
                            orcid_direct = None
                            for aw in autores_work:
                                if (aw.get("id_autor") or "").strip() == aid:
                                    orcid_direct = aw.get("orcid")
                                    break

                            if orcid_direct:
                                orcid_list.append(orcid_direct)
                            else:
                                # 2) fallback: /authors/A...
                                au = openalex_author_lookup(aid)
                                orcid_list.append(au.get("orcid") or "-")

                            time.sleep(OPENALEX_DOI_LOOKUP_RATE_LIMIT)

                    row["MatchPairs_OpenAlex"] = match_pairs
                    row["IdsOpenAlex_Provaveis"] = ids_prob
                    row["NomesOpenAlex_Provaveis"] = nomes_prob
                    row["ORCIDs_Provaveis"] = "; ".join(orcid_list) if orcid_list else "-"
                    row["MetodoMatch"] = metodo_match
                    row["MatchScoreMax"] = score_max

                # mantém rate limit também entre DOIs
                time.sleep(OPENALEX_DOI_LOOKUP_RATE_LIMIT)

                # Atualiza barra
                if doi_progress and doi_status:
                    doi_status.write(f"Consultando OpenAlex por DOI: {i}/{n_union}")
                    doi_progress.progress(int((i / n_union) * 100))

            consolidated_rows.append(row)

        # Fecha barra
        if doi_progress:
            doi_progress.empty()
        if doi_status:
            doi_status.empty()

        df_file = pd.DataFrame(consolidated_rows)
        per_file_df[uploaded_file.name] = df_file
        all_report.extend(consolidated_rows)

        file_progress.progress(int((file_idx / total_files) * 100))

    status.write("Processamento concluído.")
    file_progress.empty()
    status.empty()

    df_all = pd.DataFrame(all_report)
    df_warn_files = pd.DataFrame(no_orcid_but_openalex_files) if no_orcid_but_openalex_files else pd.DataFrame()
    df_warn_rows = pd.DataFrame(no_orcid_but_openalex_rows) if no_orcid_but_openalex_rows else pd.DataFrame()

    st.session_state["results"] = {
        "per_file_df": per_file_df,
        "all_report_df": df_all,
        "warn_files_df": df_warn_files,
        "warn_rows_df": df_warn_rows,
        "sankey": {
            "files_with_orcid": files_with_orcid,
            "files_without_orcid": files_without_orcid,
            "files_with_openalex_count": files_with_openalex_count,
            "files_without_openalex_count": files_without_openalex_count,
        },
        "options": {
            "check_openalex_without_orcid": check_openalex_without_orcid
        }
    }
    st.session_state["processed"] = True
    st.success("Processamento concluído. Agora você pode baixar os arquivos sem reprocessar.")


# =========================
# EXIBIÇÃO (reusa resultados)
# =========================
if st.session_state["processed"] and st.session_state["results"]:
    results = st.session_state["results"]

    # Downloads por arquivo
    st.markdown("## Arquivos por pesquisador")
    for fname, df_file in results["per_file_df"].items():
        st.download_button(
            label=f"Baixar CSV de: {fname}",
            data=dataframe_to_csv_bytes(df_file),
            file_name=f"{fname.rsplit('.', 1)[0]}_individual.csv",
            mime="text/csv",
            key=f"dl_file_{fname}"
        )

    # Sankey
    st.markdown("## Identificação nas bases")

    s = results["sankey"]
    sankey_labels = ["Arquivos Enviados", "Tem ORCID", "Sem ORCID", "Tem OpenAlex", "Sem OpenAlex"]
    sankey_source = [0, 0, 1, 1]
    sankey_target = [1, 2, 3, 4]
    sankey_values = [
        s["files_with_orcid"],
        s["files_without_orcid"],
        s["files_with_openalex_count"],
        s["files_without_openalex_count"],
    ]

    sankey_fig = go.Figure(data=[go.Sankey(
        arrangement="snap",
        textfont=dict(
            family="Arial, sans-serif",
            size=14,
            color="black"
        ),
        node=dict(
            pad=18,
            thickness=22,
            line=dict(color="black", width=0.8),
            label=sankey_labels,
            color=["lightblue", "lightgreen", "salmon", "gold", "orange"]
        ),
        link=dict(
            source=sankey_source,
            target=sankey_target,
            value=sankey_values,
            color="rgba(160,160,160,0.55)"
        )
    )])

    sankey_fig.update_layout(
        title=dict(text="Currículos com: ORCID e OpenAlex", x=0.0),
        margin=dict(l=10, r=10, t=50, b=10),
        height=420,
        font=dict(
            family="Arial, sans-serif",
            size=14,
            color="black"
        ),
    )

    # ✅ Streamlit: use_container_width -> width="stretch" (evita warning / compat futuro)
    st.plotly_chart(sankey_fig, width="stretch")

    # Venn
    st.markdown("## Cobertura entre as bases")
    all_report_rows = results["all_report_df"].to_dict(orient="records")
    counts, total = compute_venn_counts_from_observations(all_report_rows)
    venn_fig = create_venn_like_contarOverlap(
        counts=counts,
        total=total,
        title="Cobertura: Lattes × ORCID × OpenAlex"
    )
    st.pyplot(venn_fig)
    plt.close(venn_fig)

    # Alerta: OpenAlex sem ORCID no Lattes
    st.markdown("## Com OpenAlex, mas sem ORCID no Lattes (identificado por DOI)")
    if results["options"].get("check_openalex_without_orcid", True):
        if not results["warn_files_df"].empty:
            st.dataframe(make_arrow_friendly(results["warn_files_df"]), width="stretch")
            st.download_button(
                label="Baixar CSV (arquivos sem ORCID, mas com OpenAlex via DOI)",
                data=dataframe_to_csv_bytes(results["warn_files_df"]),
                file_name="sem_orcid_mas_com_openalex_por_doi.csv",
                mime="text/csv",
                key="dl_warn_files"
            )

            if not results["warn_rows_df"].empty:
                st.dataframe(make_arrow_friendly(results["warn_rows_df"]), width="stretch")
                st.download_button(
                    label="Baixar CSV (DOIs encontrados na OpenAlex sem ORCID no Lattes)",
                    data=dataframe_to_csv_bytes(results["warn_rows_df"]),
                    file_name="dois_openalex_sem_orcid_no_lattes.csv",
                    mime="text/csv",
                    key="dl_warn_rows"
                )
        else:
            st.info("Nenhum caso encontrado (ou a opção de consulta por DOI está desativada).")
    else:
        st.info("Ative a opção 'Verificar OpenAlex mesmo sem ORCID (por DOI)' para detectar estes casos.")

    # Relatório consolidado
    st.markdown("## Relatório Consolidado")
    st.dataframe(make_arrow_friendly(results["all_report_df"]), width="stretch")
    st.download_button(
        label="Baixar relatório completo em CSV",
        data=dataframe_to_csv_bytes(results["all_report_df"]),
        file_name="relatorio_consolidado.csv",
        mime="text/csv",
        key="dl_all_report"
    )

st.markdown(
    "<div style='text-align: center;'><p>Plataforma desenvolvida por Wellbar - 2026</p></div>",
    unsafe_allow_html=True
)

