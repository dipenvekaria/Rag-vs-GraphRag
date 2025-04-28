import os
import uuid
from datetime import datetime
from dotenv import load_dotenv
from neo4j import GraphDatabase
from openai import OpenAI  # Use OpenAI API directly
from utils import extract_pdf_text
import logging
import re
import json
load_dotenv()

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GraphDBProcessor:
    def __init__(self):
        self.openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))  # Initialize OpenAI client
        try:
            self.driver = GraphDatabase.driver(
                os.getenv("NEO4J_URI"),
                auth=(os.getenv("NEO4J_USERNAME"), os.getenv("NEO4J_PASSWORD"))
            )
            self._ensure_graph_schema()
        except Exception as e:
            logger.error(f"Failed to initialize Neo4j driver: {str(e)}")
            raise

    def _ensure_graph_schema(self):
        try:
            with self.driver.session() as session:
                session.run("""
                    CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE
                """)
                session.run("""
                    CREATE CONSTRAINT IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE
                """)
        except Exception as e:
            logger.error(f"Error creating schema constraints: {str(e)}")
            raise

    def _extract_entities_and_relationships(self, text):
        try:
            prompt = f"""Extract key entities (e.g., people, organizations, locations) and their relationships from the following text. Return a JSON list of entities and relationships in this format:

{{
  "entities": [{{"id": "unique_id", "name": "entity_name", "type": "entity_type"}}],
  "relationships": [{{"source": "source_id", "target": "target_id", "type": "relationship_type"}}]
}}

Entities should include people (type: "Person"), organizations (type: "Organization"), and locations (type: "Location"). Relationships should describe connections like WORKS_FOR (e.g., a person works for an organization), HEADQUARTERED_IN (e.g., an organization is located in a city), FOUNDED (e.g., a person founded an organization), or COLLABORATED_WITH (e.g., two people collaborated). Ensure relationship types are in uppercase (e.g., WORKS_FOR, FOUNDED). Ensure IDs are unique, lowercase, and use underscores instead of spaces (e.g., "alice_johnson"). If no entities or relationships are found, return empty lists. Pay special attention to identifying founding relationships (e.g., "Founded by Alice Johnson" should result in a FOUNDED relationship).

Examples:
Text: "Alice Johnson is the CEO of TechCorp, which is headquartered in San Francisco. TechCorp was founded by Alice Johnson."
Output:
{{
  "entities": [
    {{"id": "alice_johnson", "name": "Alice Johnson", "type": "Person"}},
    {{"id": "techcorp", "name": "TechCorp", "type": "Organization"}},
    {{"id": "san_francisco", "name": "San Francisco", "type": "Location"}}
  ],
  "relationships": [
    {{"source": "alice_johnson", "target": "techcorp", "type": "WORKS_FOR"}},
    {{"source": "alice_johnson", "target": "techcorp", "type": "FOUNDED"}},
    {{"source": "techcorp", "target": "san_francisco", "type": "HEADQUARTERED_IN"}}
  ]
}}

Text: "TechCorp was founded in 2015."
Output:
{{
  "entities": [
    {{"id": "techcorp", "name": "TechCorp", "type": "Organization"}}
  ],
  "relationships": []
}}

Text: {text}

Output: """
            logger.info(f"Input text to LLM (first 5000 chars): {text[:5000]}")

            # Call OpenAI API
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that extracts entities and relationships from text in JSON format."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=1000
            )
            result = response.choices[0].message.content.strip()
            logger.info(f"Raw LLM output for entity extraction: {result}")

            # Clean up Markdown code block markers
            cleaned_result = re.sub(r'```json\s*|\s*```', '', result).strip()
            logger.info(f"Cleaned LLM output for JSON parsing: {cleaned_result}")

            # Validate JSON
            parsed_result = json.loads(cleaned_result)
            if not isinstance(parsed_result, dict):
                logger.error("LLM output is not a dictionary")
                return {"entities": [], "relationships": []}
            if "entities" not in parsed_result or "relationships" not in parsed_result:
                logger.error("LLM output missing 'entities' or 'relationships' keys")
                return {"entities": [], "relationships": []}

            # Validate entities
            entities = parsed_result["entities"]
            if not isinstance(entities, list):
                logger.error(f"Entities is not a list: {entities}")
                return {"entities": [], "relationships": []}
            for entity in entities:
                if not isinstance(entity, dict):
                    logger.error(f"Entity is not a dict: {entity}")
                    return {"entities": [], "relationships": []}
                if not all(key in entity for key in ["id", "name", "type"]):
                    logger.error(f"Entity missing required keys: {entity}")
                    return {"entities": [], "relationships": []}
                if entity["type"] not in ["Person", "Organization", "Location"]:
                    logger.warning(f"Unexpected entity type: {entity['type']}")

            # Validate relationships
            relationships = parsed_result["relationships"]
            if not isinstance(relationships, list):
                logger.error(f"Relationships is not a list: {relationships}")
                return {"entities": [], "relationships": []}
            for rel in relationships:
                if not isinstance(rel, dict):
                    logger.error(f"Relationship is not a dict: {rel}")
                    return {"entities": [], "relationships": []}
                if not all(key in rel for key in ["source", "target", "type"]):
                    logger.error(f"Relationship missing required keys: {rel}")
                    return {"entities": [], "relationships": []}
                entity_ids = {e["id"] for e in entities}
                if rel["source"] not in entity_ids or rel["target"] not in entity_ids:
                    logger.error(f"Relationship references unknown entity IDs: {rel}")
                    return {"entities": [], "relationships": []}
                rel["type"] = rel["type"].upper()

            logger.info(f"Extracted {len(entities)} entities and {len(relationships)} relationships")
            return parsed_result
        except json.JSONDecodeError as e:
            logger.error(f"JSON parsing error: {str(e)}")
            return {"entities": [], "relationships": []}
        except Exception as e:
            logger.error(f"Error extracting entities: {str(e)}")
            return {"entities": [], "relationships": []}

    def process_and_store_pdf(self, file_obj):
        try:
            text = extract_pdf_text(file_obj.name)
            filename = os.path.basename(file_obj.name)
            doc_id = str(uuid.uuid4())

            graph_data = self._extract_entities_and_relationships(text)
            entities = graph_data["entities"]
            relationships = graph_data["relationships"]

            if not entities:
                logger.warning(f"No entities extracted for {filename}")

            with self.driver.session() as session:
                logger.info(f"Creating Document node with id: {doc_id}")
                session.run("""
                    MERGE (d:Document {id: $doc_id})
                    SET d.filename = $filename, d.conversion_date = $date
                """, doc_id=doc_id, filename=filename, date=datetime.now().isoformat())

                for entity in entities:
                    logger.info(f"Creating Entity node: {entity}")
                    session.run("""
                        MERGE (e:Entity {id: $id})
                        SET e.name = $name, e.type = $type
                        MERGE (d:Document {id: $doc_id})
                        MERGE (e)-[:MENTIONED_IN]->(d)
                    """, id=entity["id"], name=entity["name"], type=entity["type"], doc_id=doc_id)

                for rel in relationships:
                    logger.info(f"Creating RELATED relationship: {rel}")
                    session.run("""
                        MATCH (source:Entity {id: $source})
                        MATCH (target:Entity {id: $target})
                        MERGE (source)-[r:RELATED {type: $type}]->(target)
                    """, source=rel["source"], target=rel["target"], type=rel["type"])

                result = session.run("""
                    MATCH (e:Entity)-[r:RELATED]->(t:Entity)
                    WHERE e.id IN $entity_ids
                    RETURN e, r, t
                """, entity_ids=[e["id"] for e in entities])
                created_relationships = [record for record in result]
                logger.info(f"Verified {len(created_relationships)} RELATED relationships for {filename}")

                result = session.run("""
                    MATCH (e:Entity)-[r:MENTIONED_IN]->(d:Document)
                    WHERE d.id = $doc_id
                    RETURN e, r, d
                """, doc_id=doc_id)
                mentioned_relationships = [record for record in result]
                logger.info(f"Verified {len(mentioned_relationships)} MENTIONED_IN relationships for {filename}")

            logger.info(f"Processed PDF {filename} in Neo4j with {len(entities)} entities and {len(relationships)} relationships")
            return filename
        except Exception as e:
            logger.error(f"Error processing PDF {file_obj.name}: {str(e)}")
            raise

    def query(self, question):
        try:
            # Define the prompt for Cypher query generation
            prompt = f"""Convert the following question into a valid Cypher query to retrieve relevant entities and relationships from a Neo4j database. The database has Entity nodes (with id, name, type) and Document nodes (with id, filename), connected by MENTIONED_IN relationships, and Entity nodes connected by RELATED relationships with a 'type' property (e.g., WORKS_FOR, HEADQUARTERED_IN). Return a single Cypher query without using UNION. If a relationship is queried, assign it to a variable (e.g., [r:RELATED]) and use it in the RETURN clause (e.g., type(r) AS relationshipType). Ensure all variables in the RETURN clause are defined in the MATCH clause. For entity names in the question, use a CONTAINS condition to match partial names (e.g., for 'Alice', use name CONTAINS 'Alice' to match 'Alice Johnson'). For questions asking about the relationship between two entities, focus on the direct relationship between them and return only the relationship type. Return only the Cypher query text, without any Markdown, code blocks, or prefixes like '```cypher'.

    Examples:
    Question: "What is the relationship between Alice and TechCorp?"
    Cypher Query:
    MATCH (e:Entity)-[r:RELATED]->(t:Entity {{name: 'TechCorp'}})
    WHERE e.name CONTAINS 'Alice'
    RETURN type(r) AS relationshipType

    Question: "Who works for TechCorp?"
    Cypher Query:
    MATCH (e:Entity)-[r:RELATED {{type: 'WORKS_FOR'}}]->(t:Entity {{name: 'TechCorp'}})
    RETURN e.name AS entityName

    Question: {question}

    Cypher Query: """

            # Call OpenAI API to generate the Cypher query
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system",
                     "content": "You are a helpful assistant that generates Cypher queries for a Neo4j database."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0,
                max_tokens=150
            )
            cypher_query = response.choices[0].message.content.strip()

            # Clean up any Markdown or code block markers
            cypher_query = re.sub(r'```cypher\s*|\s*```', '', cypher_query).strip()
            logger.info(f"Generated Cypher query: {cypher_query}")

            # Basic query validation
            if not cypher_query.startswith(('MATCH', 'WITH', 'MERGE', 'CREATE', 'RETURN')):
                raise ValueError(f"Invalid Cypher query: {cypher_query}")

            # Validate variables in RETURN clause
            return_clause = cypher_query[cypher_query.upper().find("RETURN"):].lower()
            return_vars = set()
            # Improved parsing of RETURN clause
            return_parts = return_clause.split(',')
            for part in return_parts:
                part = part.strip()
                if not part:
                    continue
                # Remove 'return' keyword if present
                part = part.replace('return', '').strip()
                if ' as ' in part:
                    var = part.split(' as ')[0].strip()
                else:
                    var = part.strip()
                # Handle expressions like type(r)
                if var.startswith('type('):
                    var = var[5:var.find(')')]
                # Handle dot notation like e.name
                elif '.' in var:
                    var = var.split('.')[0]
                if var and var not in {'type', 'as'}:
                    return_vars.add(var)

            match_clause = cypher_query[:cypher_query.upper().find("RETURN")].lower()
            defined_vars = set()
            # Extract variables from MATCH and WHERE clauses
            for match in re.finditer(r'\b([a-zA-Z_]\w*)\b(?=:|\s*{|\s*-|\s*\))', match_clause):
                defined_vars.add(match.group(1))
            # Extract additional variables from WHERE clause or other patterns
            for match in re.finditer(r'\b([a-zA-Z_]\w*)\b', match_clause):
                var = match.group(1)
                if var not in {'match', 'optional', 'related', 'mentioned_in', 'entity', 'document', 'name', 'where',
                               'contains'}:
                    defined_vars.add(var)

            undefined_vars = return_vars - defined_vars
            if undefined_vars:
                raise ValueError(f"Undefined variables in RETURN clause: {undefined_vars}")

            with self.driver.session() as session:
                result = session.run(cypher_query)
                records = [record.data() for record in result]

            context = "\n".join([str(record) for record in records])
            if not context:
                return {"answer": "No relevant information found in graph.", "chunks": []}

            # Generate a human-readable answer using OpenAI API
            answer_prompt = f"""Based solely on the provided graph data, provide a concise, accurate answer to the question. Do not use external knowledge. If the graph data contains multiple relationships, list them all in a human-readable format (e.g., 'Alice Johnson founded TechCorp (FOUNDED) and works for TechCorp (WORKS_FOR).').

    Question: {question}
    Graph Data: {context}

    Answer: """
            answer_response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system",
                     "content": "You are a helpful assistant that answers questions based on graph data."},
                    {"role": "user", "content": answer_prompt}
                ],
                temperature=0,
                max_tokens=100
            )
            answer = answer_response.choices[0].message.content.strip()
            return {"answer": answer, "chunks": records}
        except Exception as e:
            logger.error(f"Error querying graph: {str(e)}")
            return {"answer": f"Graph query failed: {str(e)}", "chunks": []}

    def get_processed_files(self):
        try:
            with self.driver.session() as session:
                total_nodes = session.run("MATCH (d:Document) RETURN count(d) AS count").single()["count"]
                result = session.run("""
                    MATCH (d:Document)
                    WHERE d.filename IS NOT NULL
                    RETURN d.filename AS filename
                """)
                files = {record["filename"] for record in result}
                logger.info(f"Found {len(files)} Document nodes with filename property out of {total_nodes} total Document nodes.")
                if not files:
                    logger.warning("No Document nodes with filename property found in Neo4j. Check database for inconsistent nodes.")
                return files
        except Exception as e:
            logger.error(f"Error fetching processed files from Neo4j: {str(e)}")
            return set()

    def delete_file_points(self, filename):
        try:
            with self.driver.session() as session:
                session.run("""
                    MATCH (d:Document {filename: $filename})
                    OPTIONAL MATCH (d)<-[:MENTIONED_IN]-(e:Entity)
                    OPTIONAL MATCH (e)-[r:RELATED]-()
                    DELETE d, r
                    WITH e
                    WHERE NOT (e)-[:MENTIONED_IN]->()
                    DELETE e
                """, filename=filename)
            logger.info(f"Deleted graph data for file: {filename}")
            return f"Deleted graph data for file: {filename}"
        except Exception as e:
            logger.error(f"Error deleting file {filename}: {str(e)}")
            return f"Error deleting file: {str(e)}"