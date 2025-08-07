import os
import json
import pickle, torch, faiss
import numpy as np
from tqdm import tqdm
class PhenotypeMatcher:
    def __init__(self, phenotype_list, model_name, cache_dir="embedding_cache"):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model_name = model_name.replace("/", "_")  # safe filename
        self.phenotype_list = phenotype_list
        self.cache_dir = cache_dir

        os.makedirs(cache_dir, exist_ok=True)
        self.emb_path = os.path.join(cache_dir, f"{self.model_name}_embeddings.npy")
        self.text_path = os.path.join(cache_dir, f"{self.model_name}_phenotypes.pkl")

        # Load model
        from transformers import AutoTokenizer, AutoModel
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(self.device)
        self.model.eval()

        # Load or compute embeddings
        self.phenotype_embeddings = self._load_or_compute_embeddings()
        self.index = self._build_faiss_index(self.phenotype_embeddings)

    def _mean_pooling(self, model_output, attention_mask):
        token_embeddings = model_output.last_hidden_state
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / input_mask_expanded.sum(1)

    def _embed_texts(self, texts):
        embeddings = []
        with torch.no_grad():
            for text in tqdm(texts, desc = "Embedding HPO database:"):
                inputs = self.tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(self.device)
                outputs = self.model(**inputs)

                if "MiniLM" in self.model_name or "ELECTRA" in self.model_name:
                    embedding = self._mean_pooling(outputs, inputs['attention_mask'])
                else:
                    embedding = outputs.last_hidden_state[:, 0, :]  # CLS token

                normed = torch.nn.functional.normalize(embedding, dim=1)
                embeddings.append(normed.cpu().numpy())
        return np.vstack(embeddings)

    def _load_or_compute_embeddings(self):
        if os.path.exists(self.emb_path) and os.path.exists(self.text_path):
            print("Detected existing HPO Database Embeddings")
            with open(self.text_path, "rb") as f:
                cached_texts = pickle.load(f)
            if cached_texts == self.phenotype_list:
                return np.load(self.emb_path)
        print("No existing HPO Database Embeddings are stored - Running embedding now")
        # Recompute
        embeddings = self._embed_texts(self.phenotype_list)
        np.save(self.emb_path, embeddings)
        with open(self.text_path, "wb") as f:
            pickle.dump(self.phenotype_list, f)
        return embeddings

    def _build_faiss_index(self, embeddings):
        dim = embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(embeddings)
        return index

    def match(self, query, top_k=1):
        query_embedding = self._embed_texts([query])
        scores, indices = self.index.search(query_embedding, top_k)
        results = [(self.phenotype_list[i], float(scores[0][j])) for j, i in enumerate(indices[0])]
        return results if top_k > 1 else results[0]
