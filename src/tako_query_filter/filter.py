import json
import logging
import re
import joblib
import spacy
import numpy as np
from typing import Iterable, List, Optional, Set, Union
from functools import cached_property
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegressionCV
from sklearn.svm import SVC
from spacy.language import Language
from torch import Tensor
from huggingface_hub import snapshot_download, hf_hub_download


class TakoQueryFilter:
    def __init__(
        self,
        chart_model: SVC,
        topic_model: LogisticRegressionCV,
        spacy_model: Language,
        keywords: Set[str],
    ):
        self.chart_model = chart_model
        self.topic_model = topic_model
        self.spacy_model = spacy_model
        self.keywords = keywords
        self.keyword_match_score = 0.9
        self.spacy_dims = spacy_model.vocab.vectors_length

    @classmethod
    def load_from_hf(
        cls,
        scikit_path: str = "TakoData/QueryClassifier",
        spacy_path: str = "TakoData/ner-model-best",
        revision: Optional[str] = "test-models",  # TODO: Make this None
    ):
        spacy_model = spacy.load(snapshot_download(spacy_path))

        chart_model = joblib.load(
            hf_hub_download(
                repo_id=scikit_path,
                filename="tako-classifier/chart_model.pkl",
                revision=revision,
            )
        )
        topic_model = joblib.load(
            hf_hub_download(
                repo_id=scikit_path,
                filename="tako-classifier/topic_model.pkl",
                revision=revision,
            )
        )
        keywords_file = hf_hub_download(
            repo_id=scikit_path,
            filename="tako-classifier/keywords.json",
            revision=revision,
        )
        with open(keywords_file, "r") as f:
            keywords = set(json.load(f))

        return cls(chart_model, topic_model, spacy_model, keywords)

    @cached_property
    def model(self) -> SentenceTransformer:
        return SentenceTransformer(
            "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
        )

    def create_embeddings(
        self, queries: Iterable[str]
    ) -> Union[List[Tensor], np.ndarray, Tensor]:
        return self.model.encode(list(queries))

    def _extract_spacy_features(self, queries: Iterable[str]) -> List[np.ndarray]:
        vectors = []
        for query in queries:
            span_count = 0
            vector = np.zeros((self.spacy_dims,))
            doc = self.spacy_model(query)
            for span_group in doc.spans.values():
                for span in span_group:
                    vector += span.vector
                    span_count += 1
            vectors.append(vector / span_count)

        return vectors

    def predict(
        self,
        queries: List[str],
        embeddings: Union[List[Tensor], np.ndarray, Tensor] = np.array([]),
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
        embeddings: Union[List[Tensor], np.ndarray, Tensor] = np.array([]),
        chart_weight=0.5,
        topic_weight=0.5,
    ) -> np.ndarray:
        if len(embeddings) != len(queries):
            if len(embeddings) > 0:
                logging.warning(
                    f"Provided embeddings of len {len(embeddings)} are not the same length as queries {len(queries)}, generating embeddings"
                )
            embeddings = self.create_embeddings(queries)

        # spacy_features = self._extract_spacy_features(queries)
        # topic_features = np.hstack((embeddings, spacy_features))

        # Get probabilities from both models
        chart_probs = self.chart_model.predict_proba(embeddings)
        # topic_probs = self.topic_model.predict_proba(topic_features)
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
        query = query.lower()
        for keyword in split_keywords:
            # Create a regex pattern to match the keyword as a whole word
            pattern = r"\b{}\b".format(re.escape(keyword))
            if re.search(pattern, query):
                parts = [part.strip() for part in re.split(pattern, query)]
                if len(parts) > 1:
                    return parts

        return [query]  # Return the original query if no split occurs
