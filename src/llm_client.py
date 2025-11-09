import requests
import json
from typing import List, Dict, Optional, Generator

class OllamaLLM:
    """
    Client for Ollama API to use Llama 3.1 8B locally.
    """
    
    def __init__(
        self,
        model_name: str = "llama3.1:8b",
        base_url: str = "http://localhost:11434",
        temperature: float = 0.7,
        top_p: float = 0.9,
        max_tokens: int = 2048
    ):
        """
        Initialize Ollama client.
        
        Args:
            model_name: Ollama model name (e.g., 'llama3.1:8b')
            base_url: Ollama API endpoint
            temperature: Sampling temperature (0-1)
            top_p: Nucleus sampling parameter
            max_tokens: Maximum tokens to generate
        """
        self.model_name = model_name
        self.base_url = base_url
        self.temperature = temperature
        self.top_p = top_p
        self.max_tokens = max_tokens
        
        # Verify connection
        self._verify_connection()
    
    def _verify_connection(self):
        """Check if Ollama is running and model is available."""
        try:
            response = requests.get(f"{self.base_url}/api/tags")
            if response.status_code == 200:
                models = [m["name"] for m in response.json().get("models", [])]
                if self.model_name not in models:
                    print(f"⚠️  Model {self.model_name} not found.")
                    print(f"Available models: {models}")
                    print(f"Pull it with: ollama pull {self.model_name}")
                else:
                    print(f"✓ Connected to Ollama. Model {self.model_name} ready.")
            else:
                print("⚠️  Could not connect to Ollama API")
        except requests.exceptions.ConnectionError:
            print("⚠️  Ollama not running. Start it with: ollama serve")
    
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        stream: bool = False,
        **kwargs
    ) -> str:
        """
        Generate completion from prompt.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            stream: Whether to stream response
            **kwargs: Additional generation parameters
            
        Returns:
            Generated text
        """
        url = f"{self.base_url}/api/generate"
        
        payload = {
            "model": self.model_name,
            "prompt": prompt,
            "stream": stream,
            "options": {
                "temperature": kwargs.get("temperature", self.temperature),
                "top_p": kwargs.get("top_p", self.top_p),
                "num_predict": kwargs.get("max_tokens", self.max_tokens),
            }
        }
        
        if system_prompt:
            payload["system"] = system_prompt
        
        response = requests.post(url, json=payload)
        
        if stream:
            return self._handle_stream(response)
        else:
            return response.json()["response"]
    
    def chat(
        self,
        messages: List[Dict[str, str]],
        stream: bool = False,
        **kwargs
    ) -> str:
        """
        Chat completion with conversation history.
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            stream: Whether to stream response
            **kwargs: Additional generation parameters
            
        Returns:
            Assistant's response
        """
        url = f"{self.base_url}/api/chat"
        
        payload = {
            "model": self.model_name,
            "messages": messages,
            "stream": stream,
            "options": {
                "temperature": kwargs.get("temperature", self.temperature),
                "top_p": kwargs.get("top_p", self.top_p),
                "num_predict": kwargs.get("max_tokens", self.max_tokens),
            }
        }
        
        response = requests.post(url, json=payload)
        
        if stream:
            return self._handle_stream(response)
        else:
            return response.json()["message"]["content"]
    
    def _handle_stream(self, response) -> Generator[str, None, None]:
        """Handle streaming response."""
        for line in response.iter_lines():
            if line:
                chunk = json.loads(line)
                if "response" in chunk:
                    yield chunk["response"]
                elif "message" in chunk:
                    yield chunk["message"]["content"]


class RAGLLMClient:
    """
    Specialized LLM client for RAG with legal documents.
    """
    
    def __init__(self, llm: OllamaLLM):
        self.llm = llm
    
    def generate_answer(
        self,
        query: str,
        contexts: List[str],
        metadata: Optional[List[Dict]] = None,
        max_contexts: int = 5
    ) -> str:
        """
        Generate answer based on retrieved contexts.
        
        Args:
            query: User query
            contexts: Retrieved document chunks
            metadata: Optional metadata for citations
            max_contexts: Maximum contexts to include
            
        Returns:
            Generated answer
        """
        # Limit contexts
        contexts = contexts[:max_contexts]
        
        # Build context string with citations
        context_str = ""
        for i, ctx in enumerate(contexts, 1):
            source_info = ""
            if metadata and i-1 < len(metadata):
                meta = metadata[i-1]
                source_info = f" [Source: {meta.get('source', 'Unknown')}]"
            context_str += f"\n\nContext {i}{source_info}:\n{ctx}"
        
        # System prompt for legal RAG
        system_prompt = """You are a real estate assistant AI specializing in real estate law. 
Your role is to provide accurate, helpful information based on the legal documents provided.

Guidelines:
- Answer based ONLY on the provided context
- If the context doesn't contain enough information, say so clearly
- Cite specific sections, articles, or case names when mentioned in context
- Use clear, precise legal language
- If asked about legal advice, remind that this is informational only
- Never make up information not present in the context"""

        # User prompt with query and context
        user_prompt = f"""Question: {query}

Retrieved Legal Documents:{context_str}

Based on the above legal documents, please provide a comprehensive answer to the question. 
If the documents don't contain relevant information, state that clearly.

Answer:"""

        # Generate response
        response = self.llm.generate(
            prompt=user_prompt,
            system_prompt=system_prompt,
            temperature=0.2  # Lower temperature for factual responses
        )
        
        return response
    
    def extract_key_points(self, text: str) -> str:
        """Extract key legal points from text."""
        prompt = f"""Extract the key legal points from the following text. 
List them as concise bullet points.

Text: {text}

Key Points:"""
        
        return self.llm.generate(prompt, temperature=0.2)
    
    def summarize_judgment(self, judgment: str) -> str:
        """Summarize a legal judgment."""
        system = "You are an expert at summarizing real estate documents clearly and accurately."

        prompt = f"""Summarize this real estate documents, including:
1. Case name and citation (if mentioned)
2. Key facts
3. Real estate clauses
4. Important real estate principles

Judgment:
{judgment}

Summary:"""
        
        return self.llm.generate(prompt, system_prompt=system, temperature=0.2)