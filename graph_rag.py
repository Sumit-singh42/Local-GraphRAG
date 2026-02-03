import networkx as nx
import chromadb
from chromadb.config import Settings
import uuid
import json
import re
from config import VECTOR_DB_CTX, GRAPH_STORAGE_PATH
from utils import query_llm, get_embedding

class GraphRAG:
    ALLOWED_RELATIONS = [
        "OWNS", "FINANCES", "ADVISES", "CONTROLS", "OPERATES", 
        "EMPLOYED_BY", "PARTNER_WITH", "LOCATED_IN", "SUBSIDIARY_OF",
        "KNOWS", "RELATED_TO" # Generic fallback
    ]

    def __init__(self):
        # Initialize Vector Store (ChromaDB)
        self.chroma_client = chromadb.PersistentClient(path=VECTOR_DB_CTX)
        self.collection = self.chroma_client.get_or_create_collection(name="rag_collection")
        
        # Initialize Graph Store (NetworkX)
        self.graph = nx.Graph()
        
    def ingest_text(self, text, source="unknown"):
        """
        Ingests text: Chunks it, embeds it (Vector), and extracts entities (Graph).
        """
        chunks = self._chunk_text(text)
        print(f"Processing {len(chunks)} chunks...")
        
        for i, chunk in enumerate(chunks):
            chunk_id = str(uuid.uuid4())
            
            # 1. Vector Store
            embedding = get_embedding(chunk)
            self.collection.add(
                documents=[chunk],
                embeddings=[embedding],
                metadatas=[{"source": source, "chunk_index": i}],
                ids=[chunk_id]
            )
            
            # 2. Graph Extraction
            triplets = self._extract_relations(chunk)
            self._update_graph(triplets, chunk_id)
            
        print("Ingestion complete.")
        self.save_graph()

    def _chunk_text(self, text, chunk_size=500):
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start = end - 50  # Overlap
        return chunks

    def _extract_relations(self, text):
        """
        Uses LLM to extract (Entity, Relation, Entity) triplets with strict types.
        """
        relations_list = ", ".join(self.ALLOWED_RELATIONS)
        
        prompt = f"""
        Task: Extract meaningful relationships from the text.
        Target Entities: People, Organizations, Locations, **Artifacts** (e.g., keys, devices), and **Roles** (e.g., double agent).
        
        Constraint: Use ONLY these relationship types: [{relations_list}].
        - Use "OWNS" for possession of artifacts.
        - Use "IS_A" or "RELATED_TO" for Roles if applicable (or node attributes).
        If a relationship is vague, use "RELATED_TO".
        
        Output Format: One triplet per line.
        Format: HEAD | RELATION | TAIL
        
        Example Input: "John Doe manages Acme Corp."
        Example Output:
        John Doe | EMPLOYED_BY | Acme Corp
        John Doe | CONTROLS | Acme Corp

        Text:
        "{text}"

        Output:
        """
        
        response = query_llm(prompt, temperature=0.0)
        triplets = []
        
        try:
            lines = response.strip().split('\n')
            for line in lines:
                parts = line.split('|')
                if len(parts) == 3:
                     h = parts[0].strip()
                     r = parts[1].strip().upper() # Normalize relation
                     t = parts[2].strip()
                     
                     # Validation: Enforce ontology (soft check)
                     if r not in self.ALLOWED_RELATIONS:
                         r = "RELATED_TO"
                         
                     triplets.append([h, r, t])
            return triplets
        except Exception as e:
            print(f"Error parsing triplets: {e}")
            return []

    def _update_graph(self, triplets, chunk_id):
        for item in triplets:
            if len(item) == 3:
                head, relation, tail = item
                self.graph.add_node(head)
                self.graph.add_node(tail)
                self.graph.add_edge(head, tail, relation=relation, chunk_id=chunk_id)

    def _extract_query_entities(self, question):
        """
        Identifies the key entities in the question to guide pathfinding.
        """
        prompt = f"""
        Task: Extract named entities (People, Companies, Locations) AND key concepts (Artifacts, Roles, specific items) from the question.
        Output: A comma-separated list of names/concepts. NO labels, NO prefixes.
        
        Example Input: "Who holds the physical keys?"
        Example Output: physical keys, key holder
        
        Example Input: "Connection between Steve Jobs and Pixar?"
        Example Output: Steve Jobs, Pixar

        Question: "{question}"
        Output:
        """
        response = query_llm(prompt, temperature=0.0)
        if not response: return []
        
        # Clean up response (remove potential labels if LLM disobeys)
        response = re.sub(r'(People|Company|Location|Concept):', '', response, flags=re.IGNORECASE)
        
        entities = [e.strip() for e in response.split(',')]
        # Clean entities
        return [e for e in entities if len(e) > 0]

    def _find_paths(self, entities):
        """
        Finds paths between extracted entities in the graph.
        """
        paths_context = []
        found_nodes = [node for node in entities if node in self.graph]
        
        if len(found_nodes) < 1:
            return "No graph nodes found matching query entities."

        # Case 1: Multiple specific entities found -> Find connections (Paths)
        if len(found_nodes) >= 2:
            path_found_flag = False
            # Check paths between pairs
            for i in range(len(found_nodes)):
                for j in range(i + 1, len(found_nodes)):
                    source, target = found_nodes[i], found_nodes[j]
                    try:
                        # Find shortest simple paths (Limit depth INCREASED to 6 hops)
                        all_paths = list(nx.all_simple_paths(self.graph, source, target, cutoff=6))
                        
                        if all_paths:
                            path_found_flag = True
                            # Limit to top 3 paths
                            for path in all_paths[:3]:
                                path_str = ""
                                for k in range(len(path) - 1):
                                    u, v = path[k], path[k+1]
                                    rel = self.graph[u][v].get('relation', 'RELATED')
                                    path_str += f"({u}) --[{rel}]--> "
                                path_str += f"({path[-1]})"
                                paths_context.append(path_str)
                            
                    except nx.NetworkXNoPath:
                         continue
            
            # FALLBACK: If multiple nodes found but NO PATHS connecting them,
            # dump the neighborhood of each node so LLM can try to bridge the gap.
            if not path_found_flag:
                paths_context.append("No direct path found. Providing node neighborhoods:")
                for node in found_nodes:
                    neighbors = self.graph[node]
                    for neighbor, attrs in list(neighbors.items())[:5]:
                        rel = attrs.get('relation', 'RELATED')
                        paths_context.append(f"({node}) --[{rel}]--> ({neighbor})")

        # Case 2: Only one entity found -> Egocentric Context (Neighbors)
        elif len(found_nodes) == 1:
            root = found_nodes[0]
            neighbors = self.graph[root]
            # Limit neighbors to avoid context explosion
            for neighbor, attrs in list(neighbors.items())[:15]: 
                rel = attrs.get('relation', 'RELATED')
                paths_context.append(f"({root}) --[{rel}]--> ({neighbor})")

        return "\n".join(paths_context)

    def reset_collection(self):
        """
        Clears the vector collection and graph for a fresh start.
        """
        try:
             self.chroma_client.delete_collection(name="rag_collection")
             self.collection = self.chroma_client.get_or_create_collection(name="rag_collection")
             self.graph = nx.Graph()
             print("Knowledge Base (Graph + Vectors) successfully reset.")
        except Exception as e:
             print(f"Error resetting collection: {e}")

    def query(self, question):
        """
        Robust Query: Validates paths before answering.
        """
        print(f"Querying: {question}")
        
        # 1. Vector Search
        q_embedding = get_embedding(question)
        vector_results = self.collection.query(
            query_embeddings=[q_embedding],
            n_results=5 # Increased context window
        )
        vector_text = []
        if vector_results['documents']:
             vector_text = vector_results['documents'][0]
        
        # 2. Graph Search (Path Finding)
        q_entities = self._extract_query_entities(question)
        print(f"Identified Entities: {q_entities}")
        
        graph_context = self._find_paths(q_entities)
        
        # 3. Fuse & Refuse
        full_context = f"""
        === UNSTRUCTURED TEXT EVIDENCE ===
        {chr(10).join(vector_text)}
        
        === STRUCTURED GRAPH PATHS ===
        {graph_context}
        """
        
        final_prompt = f"""
        You are an intelligence analyst. Answer the user's question based on the evidence provided.
        
        Rules:
        1. Use both the Unstructured Text and the Structured Graph Paths as sources of truth.
        2. If the graph connects entities, use that connection in your answer.
        3. Determine the answer by synthesizing the graph and text context.
        4. If the provided evidence is completely unrelated to the question, state: "I cannot determine this from the available data."
        
        Context:
        {full_context}
        
        Question: {question}
        Answer:
        """
        
        return query_llm(final_prompt)

    def save_graph(self):
        nx.write_gml(self.graph, GRAPH_STORAGE_PATH)

    def load_graph(self):
        if os.path.exists(GRAPH_STORAGE_PATH):
            self.graph = nx.read_gml(GRAPH_STORAGE_PATH)
