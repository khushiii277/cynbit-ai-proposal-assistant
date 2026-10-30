"""Retrieve relevant reference projects using embeddings + FAISS, with keyword fallback."""
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import numpy as np

BASE_DIR = Path(__file__).resolve().parents[1]
DEFAULT_PROJECTS_PATH = BASE_DIR / "data" / "past_projects.json"


def load_projects(path: str | Path = DEFAULT_PROJECTS_PATH) -> list[dict[str, Any]]:
    """Load the project knowledge base."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("past_projects.json must contain a JSON list.")
    return data


def _project_text(project: dict[str, Any]) -> str:
    fields = [
        project.get("title", ""),
        project.get("domain", ""),
        project.get("summary", ""),
        project.get("problem", ""),
        project.get("solution", ""),
        " ".join(project.get("technologies", [])),
        " ".join(project.get("deliverables", [])),
        " ".join(project.get("risks", [])),
    ]
    return "\n".join(str(x) for x in fields if x)


def _tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-zA-Z][a-zA-Z0-9+#.-]{1,}", text.lower()))


def _keyword_search(query: str, projects: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
    q_tokens = _tokens(query)
    scored = []
    for project in projects:
        p_tokens = _tokens(_project_text(project))
        overlap = len(q_tokens & p_tokens)
        # Normalize by query length to make scores easier to interpret.
        score = overlap / max(len(q_tokens), 1)
        if score > 0:
            scored.append((score, project))
    scored.sort(key=lambda item: item[0], reverse=True)
    return [
        {**project, "retrieval_score": round(float(score), 4), "retrieval_method": "keyword"}
        for score, project in scored[:top_k]
    ]


class ProjectRetriever:
    """Lazy-load Sentence Transformers and FAISS so app startup can work offline."""

    def __init__(self, projects_path: str | Path = DEFAULT_PROJECTS_PATH):
        self.projects_path = Path(projects_path)
        self.projects = load_projects(self.projects_path)
        self._model = None
        self._index = None
        self._matrix = None
        self._texts = [_project_text(p) for p in self.projects]
        self.embedding_error: str | None = None

    def _build_index(self) -> bool:
        if self._index is not None:
            return True
        try:
            import faiss
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
            vectors = self._model.encode(
                self._texts, convert_to_numpy=True, normalize_embeddings=True
            ).astype("float32")
            index = faiss.IndexFlatIP(vectors.shape[1])
            index.add(vectors)
            self._matrix = vectors
            self._index = index
            return True
        except Exception as exc:  # Offline environments may not have model weights.
            self.embedding_error = str(exc)
            return False

    def search(self, query: str, top_k: int = 3) -> list[dict[str, Any]]:
        """Return top matching projects, preferring semantic retrieval."""
        query = (query or "").strip()
        if not query:
            return []
        top_k = max(1, min(int(top_k), len(self.projects)))
        if self._build_index():
            vector = self._model.encode(
                [query], convert_to_numpy=True, normalize_embeddings=True
            ).astype("float32")
            scores, indices = self._index.search(vector, top_k)
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0:
                    continue
                results.append({
                    **self.projects[int(idx)],
                    "retrieval_score": round(float(score), 4),
                    "retrieval_method": "sentence-transformers + FAISS",
                })
            return results
        return _keyword_search(query, self.projects, top_k)


def format_references(projects: list[dict[str, Any]]) -> str:
    """Format retrieved records as context for the LLM prompt."""
    blocks = []
    for p in projects:
        blocks.append(
            f"""REFERENCE ID: {p.get('project_id')}
TITLE: {p.get('title')}
DOMAIN: {p.get('domain')}
SUMMARY: {p.get('summary')}
PROBLEM: {p.get('problem')}
SOLUTION: {p.get('solution')}
TECHNOLOGIES: {', '.join(p.get('technologies', []))}
DELIVERABLES: {', '.join(p.get('deliverables', []))}
RISKS: {', '.join(p.get('risks', []))}
EFFORT (synthetic reference only): {json.dumps(p.get('effort_person_days', {}))}
DATA NOTE: {p.get('note', 'Reference data may be incomplete.')}"""
        )
    return "\n\n---\n\n".join(blocks) or "No reference projects were retrieved."
