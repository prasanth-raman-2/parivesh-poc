# RAG Chat Agent - CLI Q&A System

A command-line interface chat agent that answers questions based on knowledge stored in Milvus using Retrieval-Augmented Generation (RAG).

## Features

- 🔍 **Semantic Search**: Retrieves relevant context from Milvus knowledge base
- 🤖 **LLM-Powered Answers**: Generates accurate answers using GPT models
- 💬 **Interactive Chat**: Maintains conversation history for context-aware responses
- 📊 **Multiple Modes**: Single-query or interactive chat sessions
- 🎛️ **Customizable**: Adjust model, retrieval count, and temperature

## Quick Start

### Interactive Chat Mode

```bash
python -m app.chat.rag_agent
```

This launches an interactive session where you can ask multiple questions.

### Single Question Mode

```bash
python -m app.chat.rag_agent --query "What is the secret code?"
```

Get a quick answer to a single question.

### Show Retrieved Context

```bash
python -m app.chat.rag_agent --query "Tell me about Parivesh" --show-context
```

View the context chunks retrieved from Milvus before the answer.

## Usage Examples

### Example 1: Basic Interactive Chat

```bash
$ python -m app.chat.rag_agent

================================================================================
RAG CHAT AGENT - Knowledge Base Q&A
================================================================================

Commands:
  /help     - Show this help message
  /context  - Toggle showing retrieved context
  /clear    - Clear chat history
  /stats    - Show knowledge base statistics
  /quit     - Exit the chat

Ask any question based on the knowledge stored in Milvus!
================================================================================

🧑 You: What is the secret code?

🔍 Searching knowledge base...
💭 Generating answer...

🤖 Assistant: Based on the knowledge base, the secret code is 87721.

🧑 You: What is Parivesh?

🔍 Searching knowledge base...
💭 Generating answer...

🤖 Assistant: Parivesh is mentioned in the context. The Deficiency Detection 
agent is developed by NeGD for Parivesh.

🧑 You: /quit
👋 Goodbye!
```

### Example 2: Using Different Models

```bash
# Use GPT-4
python -m app.chat.rag_agent --model gpt-4

# Use GPT-3.5 Turbo
python -m app.chat.rag_agent --model gpt-3.5-turbo

# Use Claude
python -m app.chat.rag_agent --model claude-3-sonnet-20240229
```

### Example 3: Adjust Retrieval Settings

```bash
# Retrieve more context chunks
python -m app.chat.rag_agent --top-k 5

# Use lower temperature for more focused answers
python -m app.chat.rag_agent --temperature 0.3
```

## Interactive Commands

While in interactive mode, use these commands:

| Command | Description |
|---------|-------------|
| `/help` | Show available commands |
| `/context` | Toggle display of retrieved context chunks |
| `/clear` | Clear chat history (start fresh conversation) |
| `/stats` | Show knowledge base statistics |
| `/quit` | Exit the chat agent |

## Command-Line Arguments

```bash
python -m app.chat.rag_agent [OPTIONS]

Options:
  --model MODEL           LLM model to use (default: gpt-4o-mini)
  --top-k K              Number of chunks to retrieve (default: 3)
  --temperature TEMP     LLM temperature 0-1 (default: 0.7)
  --query "QUESTION"     Single question mode
  --show-context         Display retrieved context chunks
  -h, --help            Show help message
```

## How It Works

1. **Question Input**: User asks a question
2. **Embedding**: Question is converted to an embedding vector
3. **Retrieval**: Searches Milvus for top-k similar chunks
4. **Context Building**: Retrieved chunks are formatted as context
5. **LLM Generation**: LLM generates answer based on context
6. **Response**: Answer is displayed to user

## Architecture

```
┌─────────────┐
│ User Query  │
└──────┬──────┘
       │
       ▼
┌─────────────────┐
│ Generate        │
│ Embedding       │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│ Search Milvus   │◄─── Knowledge Base
│ (Semantic       │     (Embeddings)
│  Search)        │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│ Retrieve Top-K  │
│ Chunks          │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│ Build Context   │
│ + Chat History  │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│ LLM Generation  │
│ (GPT-4, etc.)   │
└──────┬──────────┘
       │
       ▼
┌─────────────────┐
│ Return Answer   │
└─────────────────┘
```

## Programmatic Usage

You can also use the RAG agent in your Python code:

```python
from app.chat import RAGChatAgent

# Initialize agent
agent = RAGChatAgent(
    model="gpt-4o-mini",
    top_k=3,
    temperature=0.7
)

# Connect to Milvus
agent.connect()

# Ask a question
answer = agent.answer_question(
    "What is the secret code?",
    show_context=True
)

print(answer)

# Clear history
agent.clear_history()

# Run interactive session
agent.run_interactive()

# Disconnect
agent.disconnect()
```

## Tips for Best Results

1. **Populate Knowledge Base First**: Make sure you've ingested documents into Milvus before querying
2. **Ask Specific Questions**: More specific questions yield better answers
3. **Use Context Display**: Enable `/context` to see what information is being retrieved
4. **Adjust Top-K**: Increase `--top-k` for broader context, decrease for focused answers
5. **Clear History**: Use `/clear` if switching to unrelated topics
6. **Right Temperature**: Use lower (0.3) for factual Q&A, higher (0.7-0.9) for creative answers

## Troubleshooting

### "No relevant information found"

This means no chunks in Milvus match your query. Solutions:
- Check if data is ingested: `python -m app.milvus.view_data`
- Try rephrasing your question
- Ingest more relevant documents

### Connection Errors

Ensure Milvus is running:
```bash
docker ps | grep milvus
```

### Empty Answers

If answers are vague:
- Increase `--top-k` to retrieve more context
- Lower `--temperature` for more factual responses
- Check if your knowledge base contains relevant information

## Advanced Configuration

### Custom System Prompt

To customize the agent's behavior, modify the `system_prompt` in [app/chat/rag_agent.py](app/chat/rag_agent.py):

```python
system_prompt = """Your custom instructions here..."""
```

### Change Embedding Model

Modify `self.embedding_model` in the `RAGChatAgent` class:

```python
self.embedding_model = "text-embedding-3-small"  # Faster, lower quality
```

### Adjust Search Parameters

Modify the search in `retrieve_context()` method to use different metrics or parameters.

## Examples with Real Data

After ingesting the Parivesh document from `test.py`:

```bash
$ python -m app.chat.rag_agent

🧑 You: Who developed the Deficiency Detection agent?
🤖 Assistant: The Deficiency Detection agent is developed by NeGD.

🧑 You: What is the secret code mentioned?
🤖 Assistant: The secret code is 87721.

🧑 You: Tell me about Parivesh
🤖 Assistant: Based on the available information, Parivesh is associated 
with a Deficiency Detection agent developed by NeGD with the secret code 87721.
```

---

**Happy Chatting! 🚀**
