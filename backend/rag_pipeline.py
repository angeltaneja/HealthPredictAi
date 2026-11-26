import os
import pickle
import requests
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from dotenv import load_dotenv

load_dotenv()

DB_FILE = "data/vector_store.pkl"

class Document:
    def __init__(self, page_content, metadata):
        self.page_content = page_content
        self.metadata = metadata

def get_gemini_embedding(text, api_key):
    """Fetch 768-dimensional embedding from Gemini API."""
    url = f"https://generativelanguage.googleapis.com/v1beta/models/text-embedding-004:embedContent?key={api_key}"
    headers = {"Content-Type": "application/json"}
    payload = {
        "model": "models/text-embedding-004",
        "content": {
            "parts": [{"text": text}]
        }
    }
    response = requests.post(url, headers=headers, json=payload, timeout=10)
    if response.status_code == 200:
        return response.json()["embedding"]["values"]
    else:
        raise Exception(f"Gemini Embedding error: {response.status_code} - {response.text}")

def initialize_vector_db():
    print("Initializing Custom Vector Database...")
    summaries_dir = "data/discharge_summaries"
    if not os.path.exists(summaries_dir):
        raise FileNotFoundError(f"Summaries directory {summaries_dir} not found. Run generate_data.py first.")
        
    documents = []
    for filename in os.listdir(summaries_dir):
        if filename.endswith(".txt"):
            pid = "_".join(filename.split("_")[:2])  # PAT_0001
            file_path = os.path.join(summaries_dir, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                content = f.read()
            documents.append({"page_content": content, "metadata": {"patient_id": pid}})
            
    print(f"Loaded {len(documents)} patient records. Generating representation...")
    
    gemini_key = os.getenv("GEMINI_API_KEY")
    
    if gemini_key:
        print("GEMINI_API_KEY found. Generating semantic embeddings via Gemini API...")
        embeddings = []
        for i, doc in enumerate(documents, 1):
            if i % 50 == 0:
                print(f"  Processed {i}/{len(documents)}...")
            try:
                emb = get_gemini_embedding(doc["page_content"], gemini_key)
                embeddings.append(emb)
            except Exception as e:
                print(f"Error getting embedding for {doc['metadata']['patient_id']}: {e}")
                # Fall back to zero vector in case of rate limits
                embeddings.append([0.0] * 768)
        
        db_data = {
            "type": "gemini",
            "documents": documents,
            "embeddings": np.array(embeddings)
        }
    else:
        print("GEMINI_API_KEY not found. Fitting Scikit-Learn TF-IDF vectorizer...")
        vectorizer = TfidfVectorizer(stop_words="english")
        texts = [doc["page_content"] for doc in documents]
        tfidf_matrix = vectorizer.fit_transform(texts)
        
        db_data = {
            "type": "tfidf",
            "documents": documents,
            "vectorizer": vectorizer,
            "embeddings": tfidf_matrix
        }
        
    os.makedirs("data", exist_ok=True)
    with open(DB_FILE, "wb") as f:
        pickle.dump(db_data, f)
    print(f"Successfully saved Custom Vector Store to {DB_FILE}")

def query_patient_assistant(patient_id, query_text):
    print(f"Querying Custom RAG Assistant for patient {patient_id}...")
    if not os.path.exists(DB_FILE):
        initialize_vector_db()
        
    with open(DB_FILE, "rb") as f:
        db = pickle.load(f)
        
    documents = db["documents"]
    embeddings = db["embeddings"]
    db_type = db["type"]
    
    # Filter documents by patient_id
    filtered_indices = [i for i, doc in enumerate(documents) if doc["metadata"]["patient_id"] == patient_id]
    
    if not filtered_indices:
        return f"Error: No clinical records found for Patient ID: {patient_id}."
        
    # Standard patient record Q&A only works on the filtered patient
    filtered_docs = [documents[i] for i in filtered_indices]
    
    # We retrieve the single most relevant page content for the patient (usually one complete summary)
    context = filtered_docs[0]["page_content"]
    
    # Perform a similarity/relevance verification if multiple documents exist (not typical in this simulation)
    if len(filtered_docs) > 1:
        if db_type == "gemini":
            gemini_key = os.getenv("GEMINI_API_KEY")
            if gemini_key:
                try:
                    q_emb = get_gemini_embedding(query_text, gemini_key)
                    filtered_embs = embeddings[filtered_indices]
                    similarities = cosine_similarity([q_emb], filtered_embs)[0]
                    best_idx = np.argmax(similarities)
                    context = filtered_docs[best_idx]["page_content"]
                except:
                    pass
        else:
            vectorizer = db["vectorizer"]
            q_vec = vectorizer.transform([query_text])
            filtered_embs = embeddings[filtered_indices]
            similarities = cosine_similarity(q_vec, filtered_embs)[0]
            best_idx = np.argmax(similarities)
            context = filtered_docs[best_idx]["page_content"]

    # Answer query
    gemini_key = os.getenv("GEMINI_API_KEY")
    if not gemini_key:
        print("GEMINI_API_KEY not found. Using local rule-based fallback QA...")
        return offline_fallback_qa(context, query_text)
        
    print("GEMINI_API_KEY found. Generating answer via Gemini 1.5 Flash...")
    prompt = f"""
You are an expert clinical virtual assistant helping a doctor review patient charts.
Answer the clinician's query accurately using ONLY the provided clinical discharge summary.
If the answer is not in the text, state that the information is not documented in the current record.

CLINICAL DISCHARGE SUMMARY:
{context}

CLINICIAN'S QUERY:
{query_text}

Provide a concise, professional clinical answer. Format the response cleanly using markdown (bullet points, bold text).
"""
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={gemini_key}"
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{
                "parts": [{"text": prompt}]
            }]
        }
        response = requests.post(url, headers=headers, json=payload, timeout=10)
        if response.status_code == 200:
            return response.json()["candidates"][0]["content"]["parts"][0]["text"]
        else:
            return f"Error from Gemini API (HTTP {response.status_code}). Fallback response:\n\n" + offline_fallback_qa(context, query_text)
    except Exception as e:
        return f"Error: LLM service unavailable ({e}). Fallback response:\n\n" + offline_fallback_qa(context, query_text)

def offline_fallback_qa(context, query):
    """Local keyword-based extractor to answer queries when offline."""
    query_lower = query.lower()
    
    sections = {}
    current_section = "General"
    sections[current_section] = []
    
    for line in context.split("\n"):
        line_strip = line.strip()
        if not line_strip:
            continue
        if line_strip.isupper() and (len(line_strip) > 3 or ":" in line_strip):
            current_section = line_strip.replace(":", "")
            sections[current_section] = []
        else:
            sections[current_section].append(line_strip)
            
    for sect in list(sections.keys()):
        sections[sect] = "\n".join(sections[sect])
        
    def find_relevant_section(keywords):
        for sect_name, content in sections.items():
            if any(k in sect_name.lower() for k in keywords):
                return f"### {sect_name}\n{content}"
        return None

    if any(k in query_lower for k in ["medication", "drug", "prescription", "take", "pill"]):
        sec = find_relevant_section(["instruction", "plan", "status", "course"])
        if sec:
            return f"**[Local Fallback Assistant]** Here are the medication and discharge instructions:\n\n{sec}"
            
    if any(k in query_lower for k in ["lab", "result", "hba1c", "creatinine", "sodium", "blood"]):
        lab_lines = []
        for line in context.split("\n"):
            if any(l in line.lower() for l in ["hba1c", "creatinine", "sodium", "lab"]):
                lab_lines.append(f"- {line.strip()}")
        if lab_lines:
            return f"**[Local Fallback Assistant]** Laboratory values recorded:\n\n" + "\n".join(lab_lines)
            
    if any(k in query_lower for k in ["comorbidity", "history", "illness", "chronic", "condition"]):
        sec = find_relevant_section(["history", "comorbidities"])
        if sec:
            return f"**[Local Fallback Assistant]** Medical history and comorbidities:\n\n{sec}"
            
    if any(k in query_lower for k in ["discharge", "where", "go", "disposition"]):
        for line in context.split("\n"):
            if "discharge disposition" in line.lower() or "discharged to" in line.lower():
                return f"**[Local Fallback Assistant]** Discharge disposition: {line.strip()}"

    summary_lines = []
    lines = context.split("\n")
    for line in lines[:10]:
        if line.strip() and "=" not in line:
            summary_lines.append(line.strip())
            
    return (
        f"**[Local Fallback Assistant]** Summary of clinical records for this query:\n\n"
        + "\n".join(summary_lines[:8])
        + "\n\n*(Note: Set `GEMINI_API_KEY` in `backend/.env` to enable full clinician conversation)*"
    )

if __name__ == "__main__":
    # Test compilation
    initialize_vector_db()
    
    # Test local query
    ans = query_patient_assistant("PAT_0001", "What are the patient's discharge instructions and medications?")
    print("\nTest Assistant Response:")
    print(ans)
