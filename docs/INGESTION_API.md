# Document Ingestion API

API endpoints for ingesting, managing, and viewing document chunks with metadata.

## Base URL

```
http://localhost:8000/ingestion
```

## Endpoints

### 1. Ingest Document

**POST** `/ingestion/ingest`

Ingest a document into the vector database with AI-powered metadata extraction.

**Request Body:**
```json
{
  "file_path": "/path/to/document.md",
  "clear_existing": false,
  "chunk_size": 4000,
  "overlap": 200
}
```

**Parameters:**
- `file_path` (string, required): Absolute path to the document file
- `clear_existing` (boolean, optional): Clear existing data before ingestion (default: false)
- `chunk_size` (integer, optional): Size of text chunks in characters (1000-10000, default: 4000)
- `overlap` (integer, optional): Overlap between chunks in characters (0-1000, default: 200)

**Response:**
```json
{
  "success": true,
  "message": "Successfully ingested 150 chunks from /path/to/document.md",
  "stats": {
    "total_chunks": 150,
    "total_embeddings": 150,
    "milvus_ids_sample": [450001234567890, 450001234567891, ...],
    "chunk_types": {
      "section_content": 100,
      "table": 20,
      "baseline_data": 15,
      "general": 15
    },
    "unique_sections": 45,
    "page_range": "1 - 250",
    "top_keywords": [
      {"keyword": "environmental", "count": 120},
      {"keyword": "PCBL", "count": 95}
    ],
    "source_document": "EIA_Report_PCBL_TN_Limited.md"
  }
}
```

**Example:**
```bash
curl -X POST "http://localhost:8000/ingestion/ingest" \
  -H "Content-Type: application/json" \
  -d '{
    "file_path": "/home/prasa/projects/negd/parivesh-poc/app/data/document.md",
    "clear_existing": true,
    "chunk_size": 4000,
    "overlap": 200
  }'
```

---

### 2. List Documents

**GET** `/ingestion/documents`

Get a list of all ingested documents with summary statistics.

**Response:**
```json
{
  "total_documents": 2,
  "documents": [
    {
      "source_document": "EIA_Report_PCBL_TN_Limited.md",
      "total_chunks": 150,
      "chunk_types": {
        "section_content": 100,
        "table": 20,
        "baseline_data": 15,
        "general": 15
      },
      "page_range": "1 - 250",
      "unique_sections": 45
    }
  ]
}
```

**Example:**
```bash
curl "http://localhost:8000/ingestion/documents"
```

---

### 3. Get Chunks (Paginated)

**GET** `/ingestion/chunks`

Get paginated list of chunks with full metadata for verification.

**Query Parameters:**
- `page` (integer, optional): Page number starting from 1 (default: 1)
- `page_size` (integer, optional): Number of items per page, 1-100 (default: 10)
- `source_document` (string, optional): Filter by source document name
- `chunk_type` (string, optional): Filter by chunk type (table, section_content, etc.)
- `section_number` (string, optional): Filter by section number (e.g., "2.3")

**Response:**
```json
{
  "total": 150,
  "page": 1,
  "page_size": 10,
  "total_pages": 15,
  "chunks": [
    {
      "id": 450001234567890,
      "text": "2.0 PROJECT DESCRIPTION\n\n2.1 Introduction\nPCBL (TN) Limited...",
      "metadata": {
        "chunk_index": 0,
        "page_number": 15,
        "section_number": "2.1",
        "section_title": "Introduction",
        "chunk_type": "section_content",
        "keywords": [
          "PCBL",
          "carbon black",
          "manufacturing",
          "Tamil Nadu",
          "environmental clearance"
        ],
        "entities": {
          "locations": ["Tamil Nadu", "Tiruvallur", "Gummidipoondi"],
          "organizations": ["PCBL (TN) Limited", "SIPCOT"],
          "numbers": ["240 TPD", "150 MW"],
          "dates": ["2024"]
        },
        "char_count": 3850,
        "word_count": 620,
        "has_tables": false,
        "has_numbers": true,
        "source_document": "EIA_Report_PCBL_TN_Limited.md"
      }
    }
  ]
}
```

**Examples:**

Get first page with 20 items:
```bash
curl "http://localhost:8000/ingestion/chunks?page=1&page_size=20"
```

Filter by chunk type:
```bash
curl "http://localhost:8000/ingestion/chunks?chunk_type=table&page_size=5"
```

Filter by section:
```bash
curl "http://localhost:8000/ingestion/chunks?section_number=2.3&page_size=10"
```

Filter by source document:
```bash
curl "http://localhost:8000/ingestion/chunks?source_document=EIA_Report_PCBL_TN_Limited.md"
```

---

### 4. Get Chunk by ID

**GET** `/ingestion/chunks/{chunk_id}`

Get a specific chunk by its Milvus ID.

**Path Parameters:**
- `chunk_id` (integer, required): The Milvus ID of the chunk

**Response:**
```json
{
  "id": 450001234567890,
  "text": "Full text content of the chunk...",
  "metadata": {
    "chunk_index": 5,
    "page_number": 23,
    "section_number": "3.2.1",
    "section_title": "Air Quality Baseline Data",
    "chunk_type": "baseline_data",
    "keywords": ["air quality", "PM2.5", "monitoring", "baseline"],
    "entities": {
      "locations": ["Monitoring Station A", "Monitoring Station B"],
      "organizations": [],
      "numbers": ["45 μg/m³", "38 μg/m³"],
      "dates": ["January 2024", "February 2024"]
    },
    "char_count": 4200,
    "word_count": 680,
    "has_tables": true,
    "has_numbers": true,
    "source_document": "EIA_Report_PCBL_TN_Limited.md"
  }
}
```

**Example:**
```bash
curl "http://localhost:8000/ingestion/chunks/450001234567890"
```

---

## Metadata Fields

Each chunk includes the following metadata fields extracted by AI:

| Field | Type | Description |
|-------|------|-------------|
| `chunk_index` | integer | Sequential index of the chunk |
| `page_number` | integer | Page number where chunk appears (0 if unknown) |
| `section_number` | string | Section number (e.g., "2.3.1") |
| `section_title` | string | Title of the section |
| `chunk_type` | string | Content type: table, section_content, executive_summary, toc, baseline_data, impact_assessment, compliance, general |
| `keywords` | array[string] | Top extracted keywords |
| `entities.locations` | array[string] | Location names |
| `entities.organizations` | array[string] | Organization names |
| `entities.numbers` | array[string] | Numerical values with units |
| `entities.dates` | array[string] | Temporal references |
| `char_count` | integer | Character count |
| `word_count` | integer | Word count |
| `has_tables` | boolean | Contains table data |
| `has_numbers` | boolean | Contains numerical data |
| `source_document` | string | Source document filename |

---

## Error Responses

All endpoints may return the following error responses:

**404 Not Found**
```json
{
  "detail": "File not found: /path/to/document.md"
}
```

**500 Internal Server Error**
```json
{
  "detail": "Ingestion failed: Connection to Milvus failed"
}
```

---

## Usage Workflow

1. **Ingest a document:**
   ```bash
   POST /ingestion/ingest
   ```

2. **List all ingested documents:**
   ```bash
   GET /ingestion/documents
   ```

3. **Browse chunks with pagination:**
   ```bash
   GET /ingestion/chunks?page=1&page_size=20
   ```

4. **Filter chunks by type:**
   ```bash
   GET /ingestion/chunks?chunk_type=baseline_data
   ```

5. **Inspect specific chunk:**
   ```bash
   GET /ingestion/chunks/{chunk_id}
   ```
