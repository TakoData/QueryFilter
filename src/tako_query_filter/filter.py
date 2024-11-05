import json
import logging
from pathlib import Path
import re
import joblib
import numpy as np
from typing import Iterable, List, Optional, Set
from sklearn.linear_model import LogisticRegressionCV
from huggingface_hub import hf_hub_download, snapshot_download
import spacy
from spacy.language import Language


class TakoQueryFilter:
    def __init__(
        self,
        topic_model: LogisticRegressionCV,
        spacy_model: Language,
        keywords: Set[str],
    ):
        self.topic_model = topic_model
        self.spacy_model = spacy_model
        self.keywords = keywords
        self.keyword_match_score = 0.9
        self.embeddings_model = None

    @classmethod
    def load_from_hf(
        cls,
        scikit_path: str = "TakoData/ScikitModels",
        topic_revision: Optional[str] = "a8a257f706ec28a63eeb40b088b8e05b30670971",
        spacy_revision: Optional[str] = "156303cfba1f9ac5ef7cfd35fe5dc8c9238a459d",
        force_download: bool = False,
    ):
        topic_model = joblib.load(
            hf_hub_download(
                repo_id=scikit_path,
                filename="models/topic_model.pkl",
                revision=topic_revision,
                force_download=force_download,
            )
        )
        spacy_model_dir = snapshot_download(
            repo_id="TakoData/ner-model-best",
            revision=spacy_revision,
            force_download=force_download,
        )
        spacy_model = spacy.load(spacy_model_dir)
        keywords_file = hf_hub_download(
            repo_id=scikit_path,
            filename="models/keywords.json",
            revision=topic_revision,
            force_download=force_download,
        )
        with open(keywords_file, "r") as f:
            keywords = set(json.load(f))

        return cls(topic_model, spacy_model, keywords)

    def create_embeddings(self, queries: Iterable[str]) -> np.ndarray:
        if not self.embeddings_model:
            from sentence_transformers import SentenceTransformer

            self.embeddings_model = SentenceTransformer(
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            )

        embeddings = self.embeddings_model.encode(
            list(queries), normalize_embeddings=True
        )
        return embeddings

    def extract_spacy_features(self, query: str) -> np.ndarray:
        vector = np.zeros((256,))

        doc = self.spacy_model(query)
        spans = doc.spans["sc"]
        scores = doc.spans["sc"].attrs["scores"]
        for span, score in zip(spans, scores):
            if score:
                vector += np.array(span.vector) * score

        if len(spans) > 0:
            # Normalize vector
            norm = np.linalg.norm(vector)
            if norm > 0:
                vector = (vector - np.mean(vector)) / norm

        return vector

    def predict(
        self,
        queries: List[str],
        embeddings: np.ndarray = np.array([]),
    ):
        # Use predict_proba to get class predictions
        probs = self.predict_proba(queries, embeddings)
        # Convert probabilities to binary predictions
        predictions = (probs > 0.5).astype(int)
        return predictions

    def predict_proba(
        self,
        queries: List[str],
        embeddings: np.ndarray = np.array([]),
    ) -> np.ndarray:
        if len(embeddings) != len(queries):
            if len(embeddings) > 0:
                logging.warning(
                    f"Provided embeddings of len {len(embeddings)} are not the same length as queries {len(queries)}, generating embeddings"
                )
            embeddings = self.create_embeddings(queries)

        spacy_vectors = [self.extract_spacy_features(query) for query in queries]
        # Combine embeddings with spacy vectors
        X = np.hstack([embeddings, spacy_vectors])

        # Get probabilities from both models
        probs = self.topic_model.predict_proba(X)
        positive_probs = probs[:, 1]

        for i, query in enumerate(queries):
            split_query = self._split_query(query)
            if any(split for split in split_query if split in self.keywords):
                positive_probs[i] = self.keyword_match_score

        return positive_probs

    def _split_query(self, query: str) -> List[str]:
        split_keywords = ["vs", "vs.", "versus", "or", "and"]
        split_keywords = r"\s+(vs\.?|versus|or|and)\s+"
        subqueries = re.split(split_keywords, query.lower(), re.IGNORECASE)

        return [
            sq.strip()
            for sq in subqueries
            if sq.strip() and sq.strip() not in ["vs", "vs.", "versus", "or", "and"]
        ]
