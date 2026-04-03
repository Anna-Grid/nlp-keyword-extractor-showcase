# 🔍 NLP-Based SEO & Keyword Synonym Extractor
**Status:** Portfolio Showcase (Mock Data) | **Domain:** B2B E-Commerce & Data Engineering

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python&logoColor=white)
![NLP](https://img.shields.io/badge/NLP-KeyBERT-FFD43B?logo=python&logoColor=black)
![Data Engineering](https://img.shields.io/badge/Data_Engineering-Pandas%20%7C%20ETL-150458?logo=pandas&logoColor=white)
![BigQuery](https://img.shields.io/badge/GCP-BigQuery-4285F4?logo=googlecloud&logoColor=white)

## 📌 Project Overview
This repository contains a sanitized, conceptual demonstration of an NLP (Natural Language Processing) pipeline built for a B2B E-commerce platform.

The core business challenge was to improve the internal search engine's coverage and relevance. By leveraging **KeyBERT** (a minimal and easy-to-use keyword extraction technique based on BERT embeddings), this pipeline automatically generates highly relevant keyword synonyms and domain-specific root words from unstructured product descriptions.

> ⚠️ **Note:** Due to NDA and proprietary constraints, this repository contains only mock data and core logical extraction flows. No actual company database connections or production credentials are included.

## ⚙️ Architecture & Data Flow

1. **Data Ingestion (Mocked):** Raw product descriptions are fetched (simulating an extraction from Google BigQuery or PostgreSQL).
2. **NLP Processing (KeyBERT):** The text is passed through a pre-trained multilingual transformer model (`paraphrase-multilingual-MiniLM-L12-v2`).
3. **N-gram Extraction:** The model extracts unigrams and bigrams (1-2 word combinations) that are semantically closest to the overall document embedding.
4. **Data Enrichment:** The extracted keywords are appended to the product metadata as search tags.
5. **Data Loading:** The enriched DataFrame is prepared for ingestion back into the centralized Data Warehouse (e.g., BigQuery) to train the internal search engine.

## 🚀 Business Impact (Production Results)
In the actual production environment, this automated semantic core generation tool achieved:
* **~70% improvement** in search coverage and relevance for the internal search engine.
* **~50% reduction** in manual error-analysis time for 404-pages across an e-commerce platform with ~15,000 products.
* Significant decrease in manual SEO reporting efforts by integrating GA4 and API data directly into BigQuery.

## 💻 Code Structure
* `main.py`: Contains the core NLP extraction logic and the mocked ETL pipeline flow.

## 🛠️ Technologies Demonstrated
* **Python** (Pandas for data manipulation)
* **NLP** (KeyBERT, HuggingFace Transformers)
* **ETL Automation** (Data extraction, transformation, and load preparation)
* **German Language Processing** (Custom stop-words handling for DACH market)

---
*Developed by [Anna Gridasova](https://github.com/Anna-Grid)*
