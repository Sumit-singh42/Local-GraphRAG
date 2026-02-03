from graph_rag import GraphRAG
import time

def test_rag():
    print("Initializing GraphRAG...")
    rag = GraphRAG()
    # rag.reset_collection() # Uncomment to wipe database on start
    
    # 1. Test Ingestion
    print("\n--- Testing Ingestion ---")
    start_time = time.time()
    
    # 1. Ingest Data
    print("\n--- Ingesting Data ---")
    try:
        with open("data.txt", "r", encoding="utf-8") as f:
            text_data = f.read()
        print(f"Loaded {len(text_data)} characters from data.txt")
        rag.ingest_text(text_data, source="data.txt")
    except FileNotFoundError:
        print("data.txt not found! Using default text.")
        text_data = """
        Sarah Connor works at Cyberdyne Systems. Cyberdyne Systems is a tech company located in California. 
        The T-800 is a robot sent from the future to protect Sarah. 
        Skynet is an artificial intelligence that views humanity as a threat.
        """
        rag.ingest_text(text_data, source="terminator_lore.txt")
    print(f"Ingestion took {time.time() - start_time:.2f} seconds.")
    
    # 2. Inspect Graph
    print("\n--- Inspecting Graph ---")
    print(f"Number of nodes: {rag.graph.number_of_nodes()}")
    print(f"Number of edges: {rag.graph.number_of_edges()}")
    print("Nodes:", rag.graph.nodes())
    print("Edges:", rag.graph.edges(data=True))
    
    # 3. Interactive Query Loop
    print("\n--- Ready for Questions! (Type 'exit' to quit) ---")
    while True:
        query = input("\nEnter your question: ")
        if query.lower() in ['exit', 'quit']:
            break
        
        answer = rag.query(query)
        print("\n>> ANSWER:")
        print(answer)
        print("-" * 30)

if __name__ == "__main__":
    test_rag()
