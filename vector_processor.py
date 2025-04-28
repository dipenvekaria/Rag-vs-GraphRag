import os
import uuid
from datetime import datetime
from dotenv import load_dotenv
import re
import time
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http import models
from sklearn.metrics.pairwise import cosine_similarity
from utils import extract_pdf_text

load_dotenv()

class VectorDBProcessor:
    def __init__(self):
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.qdrant_client = QdrantClient(
            url=os.getenv("QDRANT_URL"),
            api_key=os.getenv("QDRANT_API_KEY"),
            timeout=60
        )
        self.collection_name = "pdf_text_vectors"
        self._ensure_collection_exists()

    def _ensure_collection_exists(self):
        if not self.qdrant_client.collection_exists(collection_name=self.collection_name):
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(size=1536, distance=models.Distance.COSINE)
            )
            print(f"Created collection: {self.collection_name}")

    def _chunk_text(self, text, max_chunk_size=500, min_chunk_size=100, similarity_threshold=0.5):
        # Split text into sentences and filter out empty strings
        sentences = [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]
        if not sentences:
            return []  # Return empty list if no valid sentences

        # Log the sentences for debugging
        print(f"Sentences to embed: {sentences}")

        # Embed sentences
        sentence_embeddings = self._embed_documents(sentences)
        if len(sentences) != len(sentence_embeddings):
            raise ValueError("Mismatch between number of sentences and embeddings")

        chunks, chunk, chunk_embedding = [], [], []
        chunk_token_length = 0

        for i, (sentence, embedding) in enumerate(zip(sentences, sentence_embeddings)):
            if not chunk:
                chunk.append(sentence)
                chunk_embedding.append(embedding)
                chunk_token_length = len(sentence)
                continue

            similarity = cosine_similarity([embedding], [chunk_embedding[-1]])[0][0]
            if chunk_token_length + len(sentence) > max_chunk_size:
                if chunk_token_length >= min_chunk_size:
                    chunks.append(" ".join(chunk))
                chunk = [sentence]
                chunk_embedding = [embedding]
                chunk_token_length = len(sentence)
            elif similarity > similarity_threshold:
                chunk.append(sentence)
                chunk_embedding.append(embedding)
                chunk_token_length += len(sentence)
            else:
                if chunk_token_length >= min_chunk_size:
                    chunks.append(" ".join(chunk))
                chunk = [sentence]
                chunk_embedding = [embedding]
                chunk_token_length = len(sentence)

        if chunk and (chunk_token_length >= min_chunk_size or len(chunks) == 0):
            chunks.append(" ".join(chunk))
        return list(dict.fromkeys(chunks))

    def _embed_documents(self, texts):
        response = self.openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=texts
        )
        return [item.embedding for item in response.data]

    def _embed_query(self, text):
        response = self.openai_client.embeddings.create(
            model="text-embedding-3-small",
            input=[text]
        )
        return response.data[0].embedding

    def process_and_store_pdf(self, file_obj):
        self._ensure_collection_exists()
        text = extract_pdf_text(file_obj.name)
        filename = os.path.basename(file_obj.name)
        chunks = self._chunk_text(text)
        points = []
        batch_size = 100
        seen_chunks = set()

        for i in range(0, len(chunks), batch_size):
            batch_chunks = chunks[i:i + batch_size]
            embeddings = self._embed_documents(batch_chunks)

            for j, (chunk, embedding) in enumerate(zip(batch_chunks, embeddings)):
                if chunk in seen_chunks:
                    continue
                seen_chunks.add(chunk)
                point_id = str(uuid.uuid4())
                points.append(
                    models.PointStruct(
                        id=point_id,
                        vector=embedding,
                        payload={
                            "text": chunk,
                            "source_file": filename,
                            "chunk_id": i + j,
                            "original_file_name": filename,
                            "conversion_date": datetime.now().isoformat()
                        }
                    )
                )

            if points:
                self.qdrant_client.upsert(collection_name=self.collection_name, points=points)
                points = []

        return filename

    def query(self, question):
        query_embedding = self._embed_query(question)
        query_response = self.qdrant_client.query_points(
            collection_name=self.collection_name,
            query=query_embedding,
            limit=5,
            with_payload=True
        )
        search_results = query_response.points
        if not search_results:
            return {"answer": "No relevant information found.", "chunks": []}

        context = "\n\n".join([result.payload.get("text", "No text available") for result in search_results])
        chunks = [result.payload.get("text", "No text available") for result in search_results]
        source = search_results[0].payload.get("original_file_name", "Unknown")

        prompt = f"""Based solely on the provided context, provide a concise, accurate answer to the question. Do not use external knowledge.

Question: {question}
Context: {context}

Answer: """
        response = self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on provided context."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=100
        )
        answer = response.choices[0].message.content.strip()
        if "context does not" in answer:
            final_answer = answer
        else:
            final_answer = f"{answer} [Source: {source}]"

        return {"answer": final_answer, "chunks": chunks}

    def get_processed_files(self):
        processed_files = set()
        scroll_response = self.qdrant_client.scroll(
            collection_name=self.collection_name,
            limit=100,
            with_payload=True
        )
        points, next_offset = scroll_response
        while points:
            for point in points:
                original_file_name = point.payload.get("original_file_name")
                if original_file_name:
                    processed_files.add(original_file_name)
            if next_offset is None:
                break
            scroll_response = self.qdrant_client.scroll(
                collection_name=self.collection_name,
                limit=100,
                offset=next_offset,
                with_payload=True
            )
            points, next_offset = scroll_response
        return processed_files

    def delete_file_points(self, filename):
        filter_condition = models.Filter(
            must=[
                models.FieldCondition(
                    key="original_file_name",
                    match=models.MatchValue(value=filename)
                )
            ]
        )
        self.qdrant_client.delete(
            collection_name=self.collection_name,
            points_selector=models.FilterSelector(filter=filter_condition)
        )
        time.sleep(0.5)
        return f"Deleted points for file: {filename}"