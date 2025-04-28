import os
from dotenv import load_dotenv
from vector_processor import VectorDBProcessor
from graph_processor import GraphDBProcessor
from openai import OpenAI

# Load environment variables from .env file
load_dotenv()

class HybridProcessor:
    def __init__(self):
        self.vector_processor = VectorDBProcessor()
        self.graph_processor = GraphDBProcessor()
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    def process_and_store_pdf(self, file_obj):
        vector_filename = self.vector_processor.process_and_store_pdf(file_obj)
        graph_filename = self.graph_processor.process_and_store_pdf(file_obj)
        return vector_filename

    def query(self, question):
        vector_result = self.vector_processor.query(question)
        graph_result = self.graph_processor.query(question)

        combined_context = f"Vector DB Context:\n{vector_result['answer']}\n\nGraph DB Context:\n{graph_result['answer']}"
        prompt = f"""Based on the combined context from vector and graph databases, provide a concise, accurate answer to the question. Prioritize information that appears in both sources for reliability.

Question: {question}
Combined Context: {combined_context}

Answer: """
        response = self.openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that answers questions based on combined vector and graph data."},
                {"role": "user", "content": prompt}
            ],
            temperature=0,
            max_tokens=100
        )
        answer = response.choices[0].message.content.strip()

        return {
            "answer": answer,
            "chunks": {
                "vector": vector_result["chunks"],
                "graph": graph_result["chunks"]
            }
        }

    def get_processed_files(self):
        vector_files = self.vector_processor.get_processed_files()
        graph_files = self.graph_processor.get_processed_files()
        return vector_files.intersection(graph_files)

    def delete_file_points(self, filename):
        vector_status = self.vector_processor.delete_file_points(filename)
        graph_status = self.graph_processor.delete_file_points(filename)
        return f"{vector_status}\n{graph_status}"