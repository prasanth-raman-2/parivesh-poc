# How to View Milvus Data

There are several ways to view and explore data stored in Milvus:

## 1. Using the View Data Script (Recommended for Quick Access)

We've created a utility script to easily view your Milvus data:

### View All Data
```bash
python -m app.milvus.view_data
# or
python -m app.milvus.view_data view
```

This will display:
- Collection statistics (name, total entities)
- All stored entities with their ID, text, and metadata

### Search for Similar Text
```bash
python -m app.milvus.view_data search "What is the secret code?"
```

This performs semantic search and shows the top 5 most similar results.

### Get Specific Entity by ID
```bash
python -m app.milvus.view_data get 12345
```

Retrieves and displays a specific entity by its ID.

### Delete All Data
```bash
python -m app.milvus.view_data delete
```

Deletes the entire collection (prompts for confirmation).

---

## 2. Using Python Code Directly

You can also view data programmatically:

```python
from pymilvus import Collection, connections
from app.core.settings import settings

# Connect to Milvus
connections.connect("default", host=settings.MILVUS_HOST, port=settings.MILVUS_PORT)

# Get collection
collection = Collection(settings.MILVUS_COLLECTION_NAME)

# Load collection into memory
collection.load()

# Query all data
results = collection.query(
    expr="id >= 0",
    output_fields=["id", "text", "metadata"],
    limit=100
)

# Print results
for entity in results:
    print(f"ID: {entity['id']}")
    print(f"Text: {entity['text']}")
    print(f"Metadata: {entity['metadata']}")
    print("-" * 80)

# Release collection
collection.release()

# Disconnect
connections.disconnect("default")
```

---

## 3. Using Attu (Milvus Web UI) - Best for Visual Exploration

Attu is the official GUI tool for Milvus:

### Installation via Docker:
```bash
docker run -p 8000:3000 -e MILVUS_URL=localhost:19530 zilliz/attu:latest
```

Then open your browser to: **http://localhost:8000**

### Features:
- Visual collection browser
- Search interface
- Schema viewer
- Performance metrics
- Query builder

---

## 4. Using Milvus CLI

Install Milvus CLI:
```bash
pip install milvus-cli
```

Start CLI:
```bash
milvus_cli
```

Connect and view data:
```bash
connect -h localhost -p 19530
show collections
describe collection -c document_embeddings
query -c document_embeddings -o id,text,metadata
```

---

## 5. Custom Query Examples

### Count Total Entities
```python
from pymilvus import Collection, connections

connections.connect("default", host="localhost", port="19530")
collection = Collection("document_embeddings")
print(f"Total entities: {collection.num_entities}")
connections.disconnect("default")
```

### Query with Filter
```python
from pymilvus import Collection, connections

connections.connect("default", host="localhost", port="19530")
collection = Collection("document_embeddings")
collection.load()

# Query entities where metadata contains specific source
results = collection.query(
    expr='metadata["source"] == "parivesh_test"',
    output_fields=["id", "text", "metadata"],
    limit=10
)

for r in results:
    print(r)

collection.release()
connections.disconnect("default")
```

### Get Embedding Vectors
```python
from pymilvus import Collection, connections

connections.connect("default", host="localhost", port="19530")
collection = Collection("document_embeddings")
collection.load()

# Query with embedding vector
results = collection.query(
    expr="id == 447738943630381359",
    output_fields=["id", "text", "embedding"]
)

if results:
    embedding_vector = results[0]["embedding"]
    print(f"Embedding dimension: {len(embedding_vector)}")
    print(f"First 5 values: {embedding_vector[:5]}")

collection.release()
connections.disconnect("default")
```

---

## Quick Commands Summary

| Task | Command |
|------|---------|
| View all data | `python -m app.milvus.view_data` |
| Search similar | `python -m app.milvus.view_data search "query text"` |
| Get by ID | `python -m app.milvus.view_data get 123456` |
| Delete all | `python -m app.milvus.view_data delete` |
| Web UI | `docker run -p 8000:3000 -e MILVUS_URL=localhost:19530 zilliz/attu:latest` |

---

## Troubleshooting

### Collection Not Found
If you get "collection not found" error:
```bash
python -c "from pymilvus import utility, connections; connections.connect('default', host='localhost', port='19530'); print(utility.list_collections())"
```

### Connection Refused
Make sure Milvus is running:
```bash
docker ps | grep milvus
```

If not running, start it:
```bash
docker start milvus-standalone
```
