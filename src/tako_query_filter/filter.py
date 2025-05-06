import hashlib
import json
import re
from typing import Iterable, List

import spacy

from tako_query_filter.keywords import keywords
from tako_query_filter.whitelist import whitelist


class TakoQueryFilter:
    def __init__(
        self,
        keyword_hashes: Iterable[str] = keywords,
        whitelist_hashes: Iterable[str] = whitelist,
    ):
        self.nlp = spacy.load("en_tako_query_analyzer")
        self.keywords_hashes = set(keyword_hashes)
        self.whitelist_hashes = set(whitelist_hashes)
        self.keyword_match_score = 0.9
        self.whitelist_match_score = 0.8

    @classmethod
    def load_with_keywords(
        cls,
        keywords_path: str,
    ):
        """Load TakoQueryFilter with a set of whitelist keywords.

        Args:
            keywords_path: Path to the md5 hashed whitelist keywords JSON file

        Returns:
            TakoQueryFilter: Initialized filter with models loaded from local paths
        """
        with open(keywords_path, "r") as f:
            keyword_hashes = json.load(f)

        return cls(keyword_hashes)

    def predict(
        self,
        queries: List[str],
    ) -> List[int]:
        probs = self.predict_proba(queries)
        predictions = [1 if p > 0.3 else 0 for p in probs]
        return predictions

    def predict_proba(
        self,
        queries: List[str],
    ) -> List[float]:
        with self.nlp.select_pipes(enable=["tok2vec", "ner", "textcat_classify"]):
            preds = self.nlp.pipe(queries)

            probs = []
            for pred in preds:
                accept = pred.cats["ACCEPT"]
                reject = pred.cats["REJECT"]
                # Just to be safe, normalize the probabilities
                probs.append(accept / (accept + reject))

            # Check whitelist
            for i, query in enumerate(queries):
                split_query = query.lower().split()
                if any(
                    self._hash_string(split) in self.whitelist_hashes
                    for split in split_query
                ):
                    probs[i] = self.whitelist_match_score

            # Check keywords
            for i, query in enumerate(queries):
                split_query = self._split_query(query)
                split_hashes = {self._hash_string(split) for split in split_query}
                if any(
                    split_hash in self.keywords_hashes for split_hash in split_hashes
                ):
                    probs[i] = self.keyword_match_score

            return probs

    def _split_query(self, query: str) -> List[str]:
        split_keywords = ["vs", "vs.", "versus", "or", "and"]
        split_keywords = r"\s+(vs\.?|versus|or|and)\s+"
        subqueries = re.split(split_keywords, query.lower(), re.IGNORECASE)

        return [
            sq.strip()
            for sq in subqueries
            if sq.strip() and sq.strip() not in ["vs", "vs.", "versus", "or", "and"]
        ]

    @staticmethod
    def _hash_string(s: str) -> str:
        return hashlib.md5(s.lower().encode("utf-8")).hexdigest()
