# 🔍 NLP-Based SEO & Keyword Synonym Pipeline
**Status:** Portfolio Showcase (Mock Data) | **Domain:** B2B E-Commerce & Data Engineering

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![NLP](https://img.shields.io/badge/NLP-SpaCy_%7C_KeyBERT-FFD43B?logo=python&logoColor=black)
![Data Engineering](https://img.shields.io/badge/Data_Engineering-Pandas_%7C_ETL-150458?logo=pandas&logoColor=white)
![BigQuery](https://img.shields.io/badge/GCP-BigQuery-4285F4?logo=googlecloud&logoColor=white)

## 📌 Project Overview
This repository contains a sanitized, conceptual demonstration of an NLP (Natural Language Processing) and Data Cleaning pipeline built for a B2B E-commerce platform.

The core business challenge was to improve the internal search engine's coverage and relevance. By leveraging **SpaCy** (for precise German noun tokenization/lemmatization) and custom **fuzzy matching algorithms** (`difflib`), this pipeline automatically cleans messy product descriptions, extracts domain-specific root words, and generates highly relevant keyword synonyms.

> ⚠️ **Note:** Due to NDA and proprietary constraints, this repository contains only mock data and core logical extraction flows. No actual company database connections or production credentials are included.

## ⚙️ Architecture & Data Flow

1. **Data Ingestion (Mocked):** Raw, unstructured product descriptions are fetched (simulating an extraction from Google BigQuery or PostgreSQL).
2. **Text Sanitization & Filtering:** The text is rigorously cleaned from stop-words, irrelevant material descriptions, and NSFW terms using predefined ontologies (`BLACKLIST`, `STOP_WORDS`).
3. **NLP Processing (SpaCy):** The text is passed through the `de_core_news_sm` model to extract and lemmatize core nouns, handling complex German compound words and Umlauts.
4. **Fuzzy Synonym Mapping:** Extracted keywords are matched against a domain-specific dictionary (`GERMAN_ROOTS`, `GASTRO_MAPPING`) using `SequenceMatcher` to prevent near-duplicates (e.g., *Messer* vs *Messers*).
5. **Data Enrichment:** The extracted, sanitized keywords are appended to the product metadata as SEO search tags (`Final_Tags`).

## 🚀 Business Impact (Production Results)
In the actual production environment, this automated semantic core generation tool achieved:
* **~70% improvement** in search coverage and relevance for the internal search engine.
* **~50% reduction** in manual error-analysis time for 404-pages across an e-commerce platform with ~15,000 products.

## 💻 Code Structure
* `nlp_synonym_pipeline.py`: Contains the core NLP extraction logic, ontology mapping, sanitization functions, and the mocked ETL pipeline execution flow.

## 🛠️ Technologies Demonstrated
* **Python** (Pandas for data manipulation, Regex)
* **NLP** (SpaCy lemmatization, KeyBERT embeddings concept)
* **Data Cleaning** (Fuzzy string matching, custom stop-words handling for DACH market)
* **ETL Automation** (Data extraction, transformation, and load preparation)

---
*Developed by [Anna Gridasova](https://github.com/Anna-Grid)*
