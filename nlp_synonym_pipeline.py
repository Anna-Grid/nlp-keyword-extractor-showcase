"""
Portfolio Showcase: NLP-based Keyword Synonym Generator
Context: Extracting domain-specific keywords and synonyms for a B2B E-commerce search engine.
Technologies: Python, Pandas, SpaCy (NLP), KeyBERT (Embeddings), Fuzzy Matching.
Note: This is a sanitized implementation illustrating the ETL/Data Cleaning logic.
"""

import pandas as pd
import re
import spacy
from keybert import KeyBERT
from collections import OrderedDict
import difflib

# 1. INITIALIZATION: Load NLP Models
print("🧠 Initializing NLP Models (SpaCy & KeyBERT)...")
try:
    nlp = spacy.load("de_core_news_sm")
except OSError:
    print("Downloading SpaCy model 'de_core_news_sm'...")
    spacy.cli.download("de_core_news_sm")
    nlp = spacy.load("de_core_news_sm")

kw_model = KeyBERT(model='paraphrase-multilingual-mpnet-base-v2')

# ==========================================
# 2. DOMAIN KNOWLEDGE (Ontology & Blacklists)
# ==========================================
BLACKLIST = [
    "gummifotze", "erektion", "kondom", "bitch", "babe", "baby", "brüste", "pimmel", # NSFW filter
    "dusche", "bescheid", "abschlag", "pferdefuß", "nachteil", "hindernis", # False metaphors
    "spd", "cdu", "fdp", "grünen", "partei", "bundestag", # Politics
    "charakter", "gemüt", "format", "eigenart", "physis", "persönlichkeit" # Abstract terms
]

STOP_WORDS = [
    "weiß", "schwarz", "rot", "blau", "grün", "grau", "bunt", "farbig", "beige",
    "groß", "klein", "mittel", "profi", "premium", "deluxe", "eco", "line",
    "set", "stk", "stück", "beutel", "pack", "karton", "ve", "vpe", "serie",
    "maße", "maß", "gr.", "größe", "inhalt", "art", "nr", "modell", "typ"
]

MATERIAL_WORDS = [
    "edelstahl", "stahl", "aluminium", "alu", "messing", "kupfer", "chrom",
    "kunststoff", "plastik", "silikon", "pvc", "pet", "polypropylen", "pp",
    "glas", "holz", "buchenholz", "eichenholz", "porzellan", "keramik",
    "papier", "pappe", "karton", "hartpapier", "kraftpapier", "bagasse"
]

# German compound word roots for precise tokenization
_GERMAN_ROOTS_RAW = {
    "handschuhe": "Handschuhe, Arbeitshandschuhe",
    "schuhe": "Arbeitsschuhe, Sicherheitsschuhe",
    "jacke": "Kochjacke, Arbeitskleidung",
    "spülmaschine": "Geschirrspülmaschine, Gläserspülmaschine",
    "kocher": "Nudelkocher, Eierkocher, Reiskocher",
    "maschine": "Kaffeemaschine, Spülmaschine, Knetmaschine",
    "wagen": "Servierwagen, Transportwagen, Regalwagen",
    "spender": "Seifenspender, Desinfektion"
}

# Sort roots by length (longest first) to prevent partial matching (e.g., 'schuhe' before 'handschuhe')
GERMAN_ROOTS = OrderedDict(
    sorted(_GERMAN_ROOTS_RAW.items(), key=lambda item: len(item[0]), reverse=True)
)

# ==========================================
# 3. CORE NLP FUNCTIONS
# ==========================================
def extract_core_keyword_final(text):
    """
    Extracts the main object/noun from a messy product description.
    Includes protection against word truncation and handles German Umlauts.
    """
    original_text = str(text)
    clean_text = original_text

    # 1. Slice everything after technical markers
    cut_markers = [" - ", " mit ", " ohne ", " inkl", " für ", " Maße", " GN "]
    for marker in cut_markers:
        pos = clean_text.lower().find(marker.lower())
        if pos != -1: clean_text = clean_text[:pos]

    # 2. Sanitize characters (Allowing German ÄÖÜß)
    clean_text = re.sub(r'[^A-Za-z\säöüßÄÖÜ-]', ' ', clean_text)

    # 3. SpaCy Tokenization: Extracting Nouns
    doc = nlp(clean_text)
    tokens = [t for t in doc if t.pos_ in ['NOUN', 'PROPN'] and len(t.text) > 2]

    if not tokens:
        fallback = [t.strip() for t in clean_text.split() if len(t) > 2]
        return fallback[-1].capitalize() if fallback else "Produkt"

    # 4. Keyword Selection Logic
    material_fallback = None
    for token in reversed(tokens):
        lemma = token.lemma_.capitalize()
        original_token = token.text.capitalize()
        
        # Heuristic rules for German grammar edge cases
        if len(token.text) > 7 and token.text.endswith('e') and not token.lemma_.endswith('e'):
             final_word = original_token
        elif len(token.text) > 5 and len(token.lemma_) < (len(token.text) - 1):
             final_word = original_token
        else:
             final_word = lemma

        t_low = final_word.lower()

        # Exclude trash keywords
        if t_low in STOP_WORDS or t_low in MATERIAL_WORDS or t_low in BLACKLIST or t_low.isdigit():
            if not material_fallback and t_low not in STOP_WORDS:
                material_fallback = final_word
            continue

        # Match against domain-specific roots
        for root in GERMAN_ROOTS.keys():
            if t_low.endswith(root): return final_word

        return final_word

    return material_fallback if material_fallback else tokens[-1].text.capitalize()

def get_synonyms(word):
    """
    Finds synonyms utilizing domain mapping and fuzzy string matching.
    Includes strict length limits and garbage protection.
    """
    if not word or len(word) < 3 or str(word).lower() in STOP_WORDS:
        return ""

    word_lower = str(word).lower()
    if word_lower in BLACKLIST:
        return ""

    found_synonyms = set()

    # Domain Knowledge matching
    for root, syns in GERMAN_ROOTS.items():
        if word_lower.endswith(root):
            found_synonyms.update([s.strip() for s in syns.split(",")])
            if word_lower != root:
                found_synonyms.add(root.capitalize())
            break

    # Strict Filtering & Sanitization
    final_list = []
    for s in found_synonyms:
        s_clean = s.replace("...", "").replace("..", "").strip()
        s_low = s_clean.lower()
        
        # Prevent near-duplicates (e.g., Messer vs Messers)
        similarity = difflib.SequenceMatcher(None, word_lower, s_low).ratio()

        if (s_low == word_lower or s_low in BLACKLIST or s_low in MATERIAL_WORDS or 
            len(s_low) < 3 or similarity > 0.85 or len(s_low.split()) > 2 or 
            "(" in s_low or ")" in s_low or any(bad in s_low for bad in BLACKLIST)):
            continue

        final_list.append(s_clean.capitalize())

    return ", ".join(sorted(list(set(final_list)))[:6])

def enrich_tags(row):
    """
    Enriches tags with specific item attributes while protecting against NaNs.
    """
    tags = []
    if pd.notna(row.get('Synonyms')) and str(row['Synonyms']).strip():
        tags.extend([s.strip() for s in str(row['Synonyms']).split(",") if s.strip()])

    orig = str(row.get('Original_Name', '')).lower()
    if not orig or orig == 'nan':
        return ""

    attributes = {
        "edelstahl": "Edelstahl, rostfrei",
        "profi": "Gastro, Professional",
        "rollen": "Fahrbar, Rollen, mobil",
        "beheizt": "Beheizt, Warmhaltung"
    }

    for key, val in attributes.items():
        if key in orig:
            tags.extend([v.strip() for v in val.split(",")])

    clean_tags = []
    for t in tags:
        t_low = t.lower().strip()
        if (t_low and t_low not in BLACKLIST and len(t_low) > 2 and len(t_low.split()) <= 2 and "(" not in t_low):
            clean_tags.append(t.capitalize())

    return ", ".join(sorted(list(set(clean_tags))))

# ==========================================
# 4. DATA PIPELINE EXECUTION (MOCK)
# ==========================================
if __name__ == "__main__":
    print("\n🚀 Starting Mock Data Engineering Pipeline...\n")
    
    # 1. Mock Data Ingestion (Simulating messy CSV/BigQuery data)
    mock_data = [
        "Profi Edelstahl-Arbeitstisch mit Aufkantung 200x60 cm",
        "Ecoline Spülmaschine mit integriertem Dosierspender",
        "Premium Rindfleisch-Kühlschrank auf Rollen (GN 2/1)",
        "Gummifotze Scherzartikel (Should be blocked)" # Triggering NSFW filter
    ]
    
    df = pd.DataFrame({'Original_Name': mock_data})
    
    # 2. NLP Transformation: Keyword Extraction
    print("⚙️ Extracting core nouns via SpaCy...")
    df['Extracted_Keyword'] = df['Original_Name'].apply(extract_core_keyword_final)
    
    # 3. Synonym Generation & Semantic Enrichment
    print("⚙️ Mapping domain synonyms...")
    df['Synonyms'] = df['Extracted_Keyword'].apply(get_synonyms)
    df['Final_Tags'] = df.apply(enrich_tags, axis=1)
    
    # 4. Result Output (Ready for DWH Ingestion)
    print("\n✅ Pipeline complete! Transformed Data:")
    print("-" * 60)
    for idx, row in df.iterrows():
        print(f"Original : {row['Original_Name']}")
        print(f"Keyword  : {row['Extracted_Keyword']}")
        print(f"SEO Tags : {row['Final_Tags']}")
        print("-" * 60)
