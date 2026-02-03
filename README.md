
**A private, hallucination-resistant Hybrid RAG system running 100% locally.**
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Stack](https://img.shields.io/badge/Stack-NetworkX%20%7C%20ChromaDB%20%7C%20LM%20Studio-green)
## 🧐 What is this?
This is a **Local Hybrid RAG** implementation designed for complex reasoning tasks where standard vector search fails. Unlike "naive" RAG that just retrieves similar text, this system builds a **Strict Knowledge Graph** to understand relationships between entities (People, Organizations, Artifacts).
It is explicitly engineered to **eliminate hallucinations** and perform **deep multi-hop reasoning** (e.g., tracing a 6-step money laundering scheme) without sending any data to the cloud.
## 🚀 Key Features
### 1. 🛡️ Hallucination "Firewall"
- **Strict Ontology:** Enforces specific relationship types (`OWNS`, `FINANCES`, `EMPLOYED_BY`) during ingestion. Vague "edge soup" connections are rejected.
- **Auto-Refusal:** If the system cannot find a valid graph path or semantic evidence, it explicitly replies: *"I cannot determine this from the available data"* instead of making up facts.
### 2. 🧠 Advanced Reasoning
- **Deep Traversal:** Capable of **6-hop lookahead**, finding connections between entities that are textually distant (e.g., *Subject A -> Company B -> Shell Corp C -> Bank D -> Subject E*).
- **Concept Extraction:** Extracts not just named entities but also **Artifacts** (e.g., "physical keys", "server logs") and **Roles** (e.g., "Double Agent"), allowing for granular forensic queries.
### 3. 🔒 100% Privacy
- **No API Keys Required:** Designed to run with **LM Studio** (or Ollama) hosting local models like Llama 3 or Mistral.
- **Local Persistence:** Uses **ChromaDB** (Vector Store) and **NetworkX** (Graph Store) on your local disk.
## 🛠️ Tech Stack
- **Graph Engine:** `NetworkX` (Lightweight, pure Python graph operations)
- **Vector Database:** `ChromaDB` (Local semantic storage)
- **LLM Interface:** OpenAI-compatible API (targeting local `LM Studio` instance)
- **Model Recommendation:** Mistral Nemo 12B or Llama 3 8B (for speed), or a 13B+ model (for complex reasoning).
## ⚡ Quick Start
### Prerequisites
1.  Install **LM Studio** and load a model (e.g., `Mythomax-13B` or `Llama-3-8B-Instruct`).
2.  Start the Local Server on port `1234`.
### Installation
```bash
git clone [https://github.com/yourusername/local-graphrag-robust.git](https://github.com/yourusername/local-graphrag-robust.git)
cd local-graphrag-robust
pip install -r requirements.txt
