# Milvus Integration for Document Embeddings

This project now includes Milvus integration for storing and searching document embeddings.

## Prerequisites

1. **Install pymilvus**:
   ```bash
   pip install pymilvus
   ```

2. **Run Milvus Server**:
   
   Using Docker (recommended):
   ```bash
   # Standalone mode
   docker run -d --name milvus-standalone -p 19530:19530 -p 9091:9091 milvusdb/milvus:latest
   ```
   
   Or using Docker Compose:
   ```bash
   wget https://github.com/milvus-io/milvus/releases/download/v2.3.0/milvus-standalone-docker-compose.yml -O docker-compose.yml
   docker-compose up -d
   ```

## Environment Configuration

Add the following to your `.env` file:

```env
# Milvus Configuration
MILVUS_HOST=localhost
MILVUS_PORT=19530
MILVUS_COLLECTION_NAME=document_embeddings
EMBEDDING_DIMENSION=3072
```

## Usage

### Basic Usage

```python
from app.ingestion.document_ingestion import DocumentIngestion

# Initialize with Milvus enabled
doc_ingestion = DocumentIngestion(use_milvus=True)

# Connect to Milvus
doc_ingestion.connect_milvus()

# Ingest and store document
result = doc_ingestion.ingest_and_store(
    text="Your document text here...",
    metadata={"source": "example.pdf", "author": "John Doe"}
)

# Search for similar content
results = doc_ingestion.search_similar(
    query_text="What is machine learning?",
    top_k=5
)

# Disconnect when done
doc_ingestion.disconnect_milvus()
```

### Running the Example

```bash
python -m app.ingestion.example_milvus_usage
```

## Key Features

### MilvusClient

The `MilvusClient` class provides:
- **Connection management**: Connect/disconnect from Milvus server
- **Collection management**: Create collections with proper schema
- **Index creation**: Automatic IVF_FLAT index for efficient search
- **Insertion**: Store embeddings with text and metadata
- **Search**: Semantic search using L2 distance
- **Statistics**: Get collection information and stats

### DocumentIngestion Updates

Enhanced with:
- **Milvus integration**: Automatically store embeddings to Milvus
- **Complete pipeline**: `ingest_and_store()` handles chunking, embedding, and storage
- **Semantic search**: `search_similar()` finds relevant documents
- **Flexible metadata**: Attach custom metadata to chunks
- **Toggle support**: Can disable Milvus with `use_milvus=False`

## Schema

The Milvus collection uses the following schema:

| Field     | Type          | Description                    |
|-----------|---------------|--------------------------------|
| id        | INT64         | Auto-generated primary key     |
| text      | VARCHAR       | Original text chunk (max 65535)|
| embedding | FLOAT_VECTOR  | Embedding vector (dim 3072)    |
| metadata  | JSON          | Custom metadata                |

## Methods

### MilvusClient Methods

- `connect()`: Connect to Milvus server
- `disconnect()`: Disconnect from server
- `create_collection(drop_existing=False)`: Create/get collection
- `insert_embeddings(texts, embeddings, metadata)`: Insert data
- `search(query_embeddings, top_k, output_fields)`: Search similar vectors
- `load_collection()`: Load collection into memory
- `get_collection_stats()`: Get collection statistics

### DocumentIngestion Methods

- `connect_milvus()`: Initialize Milvus connection
- `disconnect_milvus()`: Close Milvus connection
- `get_chunks(text)`: Split text into chunks
- `get_embeddings(chunks)`: Generate embeddings
- `store_embeddings_to_milvus(chunks, embeddings, metadata)`: Store to Milvus
- `ingest_and_store(text, metadata)`: Complete ingestion pipeline
- `search_similar(query_text, top_k)`: Semantic search

## Troubleshooting

### Connection Issues

If you can't connect to Milvus:
```bash
# Check if Milvus is running
docker ps | grep milvus

# Check Milvus logs
docker logs milvus-standalone
```

### Collection Already Exists

To recreate a collection:
```python
doc_ingestion.milvus_client.create_collection(drop_existing=True)
```

### Memory Issues

If Milvus runs out of memory, adjust Docker resources or use a smaller embedding dimension.

## Performance Tips

1. **Batch insertion**: Insert multiple documents at once
2. **Index tuning**: Adjust `nlist` parameter based on dataset size
3. **Search optimization**: Use appropriate `nprobe` value for speed/accuracy tradeoff
4. **Collection loading**: Keep frequently-used collections loaded in memory
