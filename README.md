"""
Portfolio Showcase: NLP-based Keyword Synonym Generator
Context: Extracting domain-specific keywords for an internal e-commerce search engine.
Library used: KeyBERT (Minimalist NLP technique leveraging BERT embeddings).
Note: This is a mock implementation stripped of proprietary company data and DB connections.
"""

from keybert import KeyBERT
import pandas as pd

# 1. Initialize the NLP Model (Using a lightweight multilingual model for fast execution)
kw_model = KeyBERT(model='paraphrase-multilingual-MiniLM-L12-v2')

# 2. Mock Product Data (Simulating a database fetch from BigQuery/PostgreSQL)
mock_product_catalog = [
    {"product_id": 101, "description": "Professionelle Edelstahl-Espressomaschine mit Dual-Boiler-System für die Gastronomie."},
    {"product_id": 102, "description": "Industrieller Heißluftofen mit 10 Einschüben, programmierbarer Steuerung und Dampffunktion."},
    {"product_id": 103, "description": "Ergonomischer Gastronorm-Kühlschrank (GN 2/1) aus rostfreiem Stahl, Energieklasse A."}
]

def extract_seo_keywords(text, top_n=5):
    """
    Extracts the most relevant keywords/keyphrases using BERT embeddings.
    """
    # Extracting unigrams and bigrams
    keywords = kw_model.extract_keywords(
        text, 
        keyphrase_ngram_range=(1, 2), 
        stop_words='german', 
        top_n=top_n
    )
    return [kw[0] for kw in keywords]

def process_catalog(catalog):
    """
    Simulates the ETL transformation phase before loading data back into the search engine DB.
    """
    enriched_data = []
    print("🚀 Starting NLP Keyword Extraction Pipeline...\n")
    
    for item in catalog:
        product_id = item["product_id"]
        description = item["description"]
        
        # Core logic: generate synonyms/tags
        generated_tags = extract_seo_keywords(description)
        
        enriched_data.append({
            "product_id": product_id,
            "original_desc": description,
            "nlp_search_tags": generated_tags
        })
        
        print(f"Product ID: {product_id}")
        print(f"Tags generated: {generated_tags}\n")
        
    return pd.DataFrame(enriched_data)

if __name__ == "__main__":
    # Execute the mock pipeline
    df_results = process_catalog(mock_product_catalog)
    
    # In a real scenario, this dataframe is loaded into BigQuery via pandas_gbq
    print("✅ Pipeline complete. Ready for BigQuery ingestion.")
    # print(df_results.head())
