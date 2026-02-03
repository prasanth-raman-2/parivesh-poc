"""
CLI-based RAG (Retrieval-Augmented Generation) Chat Agent.
Answers questions based on knowledge stored in Milvus.
"""

from app.milvus import MilvusClient
from app.core.settings import settings
from litellm import completion, embedding
from typing import List, Dict, Any
import sys


class RAGChatAgent:
    def __init__(
        self,
        model: str = "gpt-5.2",
        top_k: int = 3,
        temperature: float = 0.7
    ):
        """
        Initialize RAG Chat Agent.
        
        Args:
            model: LLM model to use for generating answers
            top_k: Number of relevant chunks to retrieve from Milvus
            temperature: LLM temperature for response generation
        """
        self.model = model
        self.top_k = top_k
        self.temperature = temperature
        self.embedding_model = "text-embedding-3-large"
        
        # Initialize Milvus client
        self.milvus_client = MilvusClient(
            host=settings.MILVUS_HOST,
            port=settings.MILVUS_PORT,
            collection_name=settings.MILVUS_COLLECTION_NAME,
            dim=settings.EMBEDDING_DIMENSION
        )
        
        # Chat history
        self.chat_history = []
    
    def connect(self):
        """Connect to Milvus."""
        try:
            self.milvus_client.connect()
            self.milvus_client.load_collection()
            print(f"✓ Connected to Milvus knowledge base")
            
            # Get stats
            stats = self.milvus_client.get_collection_stats()
            print(f"✓ Knowledge base contains {stats['num_entities']} chunks")
        except Exception as e:
            print(f"✗ Failed to connect to Milvus: {e}")
            raise
    
    def disconnect(self):
        """Disconnect from Milvus."""
        self.milvus_client.disconnect()
    
    def retrieve_context(self, query: str) -> List[Dict[str, Any]]:
        """
        Retrieve relevant context from Milvus.
        
        Args:
            query: User question
            
        Returns:
            List of relevant chunks with metadata
        """
        try:
            # Generate query embedding
            response = embedding(
                model=self.embedding_model,
                input=query
            )
            query_embedding = response['data'][0]['embedding']
            
            # Search Milvus
            results = self.milvus_client.search(
                query_embeddings=[query_embedding],
                top_k=self.top_k,
                output_fields=["text", "metadata"]
            )
            
            return results[0] if results else []
        
        except Exception as e:
            print(f"Error retrieving context: {e}")
            return []
    
    def generate_answer(
        self,
        query: str,
        context_chunks: List[Dict[str, Any]]
    ) -> str:
        """
        Generate answer using LLM with retrieved context.
        
        Args:
            query: User question
            context_chunks: Retrieved context from Milvus
            
        Returns:
            Generated answer
        """
        # Build context string
        if not context_chunks:
            context_text = "No relevant information found in the knowledge base."
        else:
            context_parts = []
            for i, chunk in enumerate(context_chunks, 1):
                context_parts.append(f"[Context {i}]:\n{chunk['text']}\n")
            context_text = "\n".join(context_parts)
        
        # Build system prompt
        system_prompt = """You are a helpful AI assistant that answers questions based on the provided context from a knowledge base.

Instructions:
- Answer questions based ONLY on the provided context
- If the context doesn't contain relevant information, say so clearly
- Be concise and accurate
- If you're not sure, say "I don't have enough information to answer that"
- Cite which context section you used if relevant"""
        
        # Build user message
        user_message = f"""Context from knowledge base:
{context_text}

Question: {query}

Please provide a helpful answer based on the context above."""
        
        # Build messages including chat history
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add recent chat history (last 3 exchanges)
        for msg in self.chat_history[-6:]:
            messages.append(msg)
        
        # Add current query
        messages.append({"role": "user", "content": user_message})
        
        try:
            # Generate response
            response = completion(
                model=self.model,
                messages=messages,
                temperature=self.temperature
            )
            
            answer = response.choices[0].message.content
            
            # Update chat history
            self.chat_history.append({"role": "user", "content": query})
            self.chat_history.append({"role": "assistant", "content": answer})
            
            return answer
        
        except Exception as e:
            return f"Error generating answer: {e}"
    
    def answer_question(self, query: str, show_context: bool = False) -> str:
        """
        Complete RAG pipeline: retrieve and generate.
        
        Args:
            query: User question
            show_context: Whether to show retrieved context
            
        Returns:
            Generated answer
        """
        # Retrieve context
        print("\n🔍 Searching knowledge base...")
        context_chunks = self.retrieve_context(query)
        
        if show_context and context_chunks:
            print(f"\n📚 Retrieved {len(context_chunks)} relevant chunks:")
            for i, chunk in enumerate(context_chunks, 1):
                print(f"\n  [{i}] Distance: {chunk['distance']:.4f}")
                print(f"      Text: {chunk['text'][:150]}...")
                if chunk.get('metadata'):
                    print(f"      Metadata: {chunk['metadata']}")
        
        # Generate answer
        print("\n💭 Generating answer...")
        answer = self.generate_answer(query, context_chunks)
        
        return answer
    
    def clear_history(self):
        """Clear chat history."""
        self.chat_history = []
        print("✓ Chat history cleared")
    
    def run_interactive(self):
        """Run interactive chat session."""
        print("\n" + "="*80)
        print("RAG CHAT AGENT - Knowledge Base Q&A")
        print("="*80)
        print("\nCommands:")
        print("  /help     - Show this help message")
        print("  /context  - Toggle showing retrieved context")
        print("  /clear    - Clear chat history")
        print("  /stats    - Show knowledge base statistics")
        print("  /quit     - Exit the chat")
        print("\nAsk any question based on the knowledge stored in Milvus!")
        print("="*80)
        
        show_context = False
        
        while True:
            try:
                # Get user input
                query = input("\n🧑 You: ").strip()
                
                if not query:
                    continue
                
                # Handle commands
                if query.lower() == '/quit':
                    print("\n👋 Goodbye!")
                    break
                
                elif query.lower() == '/help':
                    print("\nCommands:")
                    print("  /help     - Show this help message")
                    print("  /context  - Toggle showing retrieved context")
                    print("  /clear    - Clear chat history")
                    print("  /stats    - Show knowledge base statistics")
                    print("  /quit     - Exit the chat")
                    continue
                
                elif query.lower() == '/context':
                    show_context = not show_context
                    status = "enabled" if show_context else "disabled"
                    print(f"\n✓ Context display {status}")
                    continue
                
                elif query.lower() == '/clear':
                    self.clear_history()
                    continue
                
                elif query.lower() == '/stats':
                    stats = self.milvus_client.get_collection_stats()
                    print(f"\n📊 Knowledge Base Statistics:")
                    print(f"   Collection: {stats['name']}")
                    print(f"   Total chunks: {stats['num_entities']}")
                    print(f"   Description: {stats['description']}")
                    continue
                
                # Process question
                answer = self.answer_question(query, show_context=show_context)
                
                # Display answer
                print(f"\n🤖 Assistant: {answer}")
            
            except KeyboardInterrupt:
                print("\n\n👋 Goodbye!")
                break
            
            except Exception as e:
                print(f"\n❌ Error: {e}")
                import traceback
                traceback.print_exc()


def main():
    """Main entry point for CLI chat agent."""
    import argparse
    
    parser = argparse.ArgumentParser(description="RAG Chat Agent - Q&A based on Milvus knowledge base")
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-5.2",
        help="LLM model to use (default: gpt-4o-mini)"
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=3,
        help="Number of relevant chunks to retrieve (default: 3)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="LLM temperature (default: 0.7)"
    )
    parser.add_argument(
        "--query",
        type=str,
        help="Single question to ask (non-interactive mode)"
    )
    parser.add_argument(
        "--show-context",
        action="store_true",
        help="Show retrieved context chunks"
    )
    
    args = parser.parse_args()
    
    # Initialize agent
    agent = RAGChatAgent(
        model=args.model,
        top_k=args.top_k,
        temperature=args.temperature
    )
    
    try:
        # Connect to Milvus
        agent.connect()
        
        # Single query mode
        if args.query:
            answer = agent.answer_question(args.query, show_context=args.show_context)
            print(f"\n🤖 Answer: {answer}\n")
        
        # Interactive mode
        else:
            agent.run_interactive()
    
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    finally:
        agent.disconnect()


if __name__ == "__main__":
    main()
