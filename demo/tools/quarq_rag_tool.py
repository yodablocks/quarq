"""
quarq RAG Tool for Open WebUI

Queries the quarq RAG corpus (ECB FSR, BdF reports, AMF SFDR, CAC 40 factsheet)
and returns grounded answers with source citations.

quarq must be running: quarq serve (http://127.0.0.1:8000)
"""

import httpx


class Tools:
    def __init__(self):
        self.base_url = "http://host.docker.internal:8000"

    def query_financial_documents(self, question: str) -> str:
        """Query the quarq RAG corpus for answers grounded in institutional documents.

        Use this tool when the user asks about:
        - ECB financial stability policy
        - Banque de France macro outlook
        - CAC 40 index composition or performance
        - AMF SFDR regulatory requirements
        - French institutional finance topics

        Args:
            question: The user's question in French or English.

        Returns:
            Answer with source citations from indexed institutional documents.
        """
        try:
            response = httpx.post(
                f"{self.base_url}/rag/query",
                json={"question": question, "n_results": 5},
                timeout=30.0,
            )
            response.raise_for_status()
            data = response.json()
            answer = data.get("answer", "No answer returned.")
            sources = data.get("sources", [])
            if sources:
                source_lines = "\n".join(
                    f"- {s.get('source', 'unknown')} (page {s.get('page', '?')})"
                    for s in sources
                )
                return f"{answer}\n\nSources:\n{source_lines}"
            return answer
        except httpx.ConnectError:
            return (
                "quarq server is not running. "
                "Start it with: quarq serve"
            )
        except Exception as exc:
            return f"RAG query failed: {exc}"
