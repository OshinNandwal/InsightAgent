import os

import faiss
from sentence_transformers import SentenceTransformer


class BusinessKnowledgeRetriever:
    """Retrieve relevant business knowledge using RAG."""

    def __init__(self, document_path):

        self.document_path = document_path

        # Lightweight embedding model
        self.model = SentenceTransformer(
            "all-MiniLM-L6-v2"
        )

        self.chunks = []

        self.index = None

        self._load_document()

        self._build_index()


    # =========================================================
    # LOAD KNOWLEDGE DOCUMENT
    # =========================================================

    def _load_document(self):
        """Read and intelligently split the knowledge document."""

        if not os.path.exists(
            self.document_path
        ):

            raise FileNotFoundError(
                f"Knowledge document not found: "
                f"{self.document_path}"
            )


        with open(
            self.document_path,
            "r",
            encoding="utf-8"
        ) as file:

            text = file.read()


        # -----------------------------------------------------
        # Split the document into meaningful lines/sections
        # -----------------------------------------------------

        lines = [
            line.strip()
            for line in text.splitlines()
            if line.strip()
        ]


        current_chunk = []


        for line in lines:

            # Section headings become separate chunks
            if (
                line in [
                    "INSIGHTAGENT BUSINESS KNOWLEDGE",
                    "Business Metrics",
                    "Business Analysis Guidelines",
                    "Example Business Questions",
                ]
            ):

                if current_chunk:

                    self.chunks.append(
                        " ".join(current_chunk)
                    )

                    current_chunk = []

                continue


            # Numbered guidelines become individual chunks
            if line[0].isdigit() and "." in line:

                if current_chunk:

                    self.chunks.append(
                        " ".join(current_chunk)
                    )

                    current_chunk = []


                current_chunk.append(line)

                continue


            # Metric definitions
            if line.endswith(":"):

                if current_chunk:

                    self.chunks.append(
                        " ".join(current_chunk)
                    )

                    current_chunk = []


                current_chunk.append(line)

                continue


            # Continue the current chunk
            current_chunk.append(line)


        # Add remaining content
        if current_chunk:

            self.chunks.append(
                " ".join(current_chunk)
            )


        # Remove duplicates and empty chunks
        self.chunks = list(
            dict.fromkeys(
                chunk.strip()
                for chunk in self.chunks
                if chunk.strip()
            )
        )


    # =========================================================
    # BUILD FAISS INDEX
    # =========================================================

    def _build_index(self):
        """Create embeddings and build FAISS index."""

        if not self.chunks:

            raise ValueError(
                "No knowledge chunks were found."
            )


        embeddings = self.model.encode(
            self.chunks,
            convert_to_numpy=True
        ).astype("float32")


        dimension = embeddings.shape[1]


        self.index = faiss.IndexFlatL2(
            dimension
        )


        self.index.add(
            embeddings
        )


    # =========================================================
    # SEARCH KNOWLEDGE
    # =========================================================

    def search(
        self,
        query,
        top_k=3
    ):
        """Retrieve the most relevant knowledge chunks."""

        query_embedding = self.model.encode(
            [query],
            convert_to_numpy=True
        ).astype("float32")


        distances, indices = self.index.search(
            query_embedding,
            min(
                top_k,
                len(self.chunks)
            )
        )


        results = []


        for distance, index in zip(
            distances[0],
            indices[0]
        ):

            results.append(
                {
                    "text": self.chunks[index],
                    "distance": float(distance),
                }
            )


        return results