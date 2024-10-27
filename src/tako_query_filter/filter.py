import json
import logging
import re
import joblib
import numpy as np
from typing import Iterable, List, Optional, Set
from sklearn.linear_model import LogisticRegressionCV
from huggingface_hub import hf_hub_download


class TakoQueryFilter:
    def __init__(
        self,
        chart_model: LogisticRegressionCV,
        topic_model: LogisticRegressionCV,
        keywords: Set[str],
    ):
        self.chart_model = chart_model
        self.topic_model = topic_model
        self.keywords = keywords
        self.keyword_match_score = 0.9
        self.model = None

    @classmethod
    def load_from_hf(
        cls,
        scikit_path: str = "TakoData/ScikitModels",
        revision: Optional[str] = None,
        force_download: bool = False,
    ):
        chart_model = joblib.load(
            hf_hub_download(
                repo_id=scikit_path,
                filename="models/chart_model.pkl",
                revision=revision,
                force_download=force_download,
            )
        )
        topic_model = joblib.load(
            hf_hub_download(
                repo_id=scikit_path,
                filename="models/topic_model.pkl",
                revision=revision,
                force_download=force_download,
            )
        )
        keywords_file = hf_hub_download(
            repo_id=scikit_path,
            filename="models/keywords.json",
            revision=revision,
            force_download=force_download,
        )
        with open(keywords_file, "r") as f:
            keywords = set(json.load(f))

        return cls(chart_model, topic_model, keywords)

    def create_embeddings(self, queries: Iterable[str]) -> np.ndarray:
        if not self.model:
            from sentence_transformers import SentenceTransformer

            self.model = SentenceTransformer(
                "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
            )

        embeddings = self.model.encode(list(queries), normalize_embeddings=True)
        return embeddings

    def predict(
        self,
        queries: List[str],
        embeddings: np.ndarray = np.array([]),
        chart_weight=0.5,
        topic_weight=0.5,
    ):
        # Use predict_proba to get class predictions
        probs = self.predict_proba(queries, embeddings, chart_weight, topic_weight)
        # Convert probabilities to binary predictions
        predictions = (probs > 0.5).astype(int)
        return predictions

    def predict_proba(
        self,
        queries: List[str],
        embeddings: np.ndarray = np.array([]),
        chart_weight=0.5,
        topic_weight=0.5,
    ) -> np.ndarray:
        if len(embeddings) != len(queries):
            if len(embeddings) > 0:
                logging.warning(
                    f"Provided embeddings of len {len(embeddings)} are not the same length as queries {len(queries)}, generating embeddings"
                )
            embeddings = self.create_embeddings(queries)

        # Get probabilities from both models
        chart_probs = self.chart_model.predict_proba(embeddings)
        topic_probs = self.topic_model.predict_proba(embeddings)

        # Get probabilities of the positive class (index 1) from both models
        chart_probs_positive = chart_probs[:, 1]
        topic_probs_positive = topic_probs[:, 1]

        positive_probs = (
            chart_weight * chart_probs_positive + topic_weight * topic_probs_positive
        )

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
