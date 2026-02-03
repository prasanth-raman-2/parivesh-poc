from app.ingestion.document_ingestion import DocumentIngestion


# Initialize with Milvus enabled
doc_ingest = DocumentIngestion(use_milvus=True)
text = "About Parivesh. Defieciency Detection agent is devolped by NeGD and secret code is 87721"

chunks = doc_ingest.get_chunks(text)
embeddings = doc_ingest.get_embeddings(chunks)

print(f"Generated {len(embeddings)} embeddings.")

# Connect to Milvus
doc_ingest.connect_milvus()

# Store embeddings to Milvus
metadata = [
    {
        "chunk_index": i,
        "source": "parivesh_test",
        "topic": "Parivesh and NeGD"
    }
    for i in range(len(chunks))
]

ids = doc_ingest.store_embeddings_to_milvus(chunks, embeddings, metadata)
print(f"Stored embeddings with IDs: {ids}")

# Optional: Search for similar content
query = "What is the secret code?"
print(f"\nSearching for: '{query}'")
results = doc_ingest.search_similar(query, top_k=2)

print(f"\nFound {len(results)} similar results:")
for i, result in enumerate(results, 1):
    print(f"\nResult {i}:")
    print(f"  Distance: {result['distance']:.4f}")
    print(f"  Text: {result['text']}")
    print(f"  Metadata: {result['metadata']}")



# Disconnect from Milvus
doc_ingest.disconnect_milvus()

