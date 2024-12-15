from .gemini import GeminiConnector
from typing import List, Dict, Any, Optional
import asyncio
from uuid import uuid4

from elasticsearch import Elasticsearch
from langchain.schema import BaseRetriever, Document
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ElasticsearchChatMessageHistory
from langchain.llms.base import LLM
from langchain.callbacks.manager import CallbackManagerForLLMRun, Callbacks
from langchain.pydantic_v1 import Field, root_validator
from dotenv import load_dotenv
import os
import requests
from langdetect import detect

# Load environment variables
load_dotenv()

class GeminiLLM(LLM):
    client: Any = Field(description="Gemini API client")
    model: str = Field(default="gemini-1.5-flash", description="Model name to use")
    
    class Config:
        """Configuration for this pydantic object."""
        arbitrary_types_allowed = True

    @root_validator()
    def validate_environment(cls, values: Dict) -> Dict:
        """Validate that api key and python package exists in environment."""
        try:
            client = values["client"]
            if not client:
                raise ValueError(
                    "Could not initialize Gemini Client"
                )
        except Exception as e:
            raise ValueError(
                "Could not initialize Gemini Client. "
                f"Error: {e}"
            )
        return values

    @property
    def _llm_type(self) -> str:
        return "gemini"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        """Execute the chat completion."""
        try:
            response = self.client.generate_content(
                prompt
            )
            return response.text if hasattr(response, 'text') else str(response)
        except Exception as e:
            print(f"Error generating content: {e}")
            return ""

    @property
    def _identifying_params(self) -> Dict[str, Any]:
        """Get the identifying parameters."""
        return {
            "model": self.model
        }

class CustomElasticsearchStore:
    def __init__(
        self,
        es_client: Elasticsearch,
        index_name: str = "knowledge-puu"
    ):
        self.client = es_client
        self.current_index = index_name
        # Define both indices
        self.indices = {
            "legal": "knowledge-puu",
            "general": "knowledge-base"
        }
        
    def is_legal_question(self, query: str) -> bool:
        """Determine if the question is about legal/regulatory content."""
        legal_keywords = [
            'peraturan', 'regulasi', 'pergub', 'perda', 'keputusan', 
            'gubernur', 'undang-undang', 'hukum', 'legal', 'sanksi',
            'regulation', 'law', 'decree', 'governor', 'penalty',
            'kebijakan', 'policy', 'perundang-undangan', 'ketentuan',
            'sk', 'surat keputusan', 'instruksi', 'instruction',
            'directive', 'circular', 'surat edaran', 'se', 'ingub'
        ]
        
        # Convert query to lowercase for case-insensitive matching
        query_lower = query.lower()
        
        # Check if any legal keyword is in the query
        return any(keyword in query_lower for keyword in legal_keywords)
    
    def set_index(self, query: str):
        """Set the appropriate index based on query type."""
        if self.is_legal_question(query):
            self.current_index = self.indices["legal"]
        else:
            self.current_index = self.indices["general"]

    def similarity_search(self, query: str, k: int = 10) -> List[Dict]:
        """Search for documents similar to query using existing text_vector field."""
        try:
            # Set appropriate index based on query
            self.set_index(query)
            
            query_embedding = self.get_embedding(query)
            
            script_query = {
                "script_score": {
                    "query": {"match_all": {}},
                    "script": {
                        "source": "cosineSimilarity(params.query_vector, 'text_vector') + 1.0",
                        "params": {"query_vector": query_embedding}
                    }
                }
            }
            
            response = self.client.search(
                index=self.current_index,
                body={
                    "size": k,
                    "query": script_query,
                    "_source": ["text"]
                }
            )
            
            results = []
            for hit in response['hits']['hits']:
                doc = {
                    'content': hit['_source']['text'],
                    'metadata': {
                        'score': hit['_score'],
                        'index': self.current_index
                    },
                    'score': hit['_score']
                }
                results.append(doc)
            
            return results
            
        except Exception as e:
            print(f"Error in similarity search: {e}")
            return []

    def get_embedding(self, text: str) -> List[float]:
        """Get embeddings from custom API."""
        response = requests.post(
            url="http://52.230.101.0:1013/embeddings",
            json={
                "text": text,
                "token": "givemeazurecredsboys!"
            }
        )
        return response.json()['detail']['vector']

class CustomRetriever(BaseRetriever):
    """Custom retriever that wraps our Elasticsearch store."""
    
    store: Any = Field(description="The vector store to use for retrieval")

    class Config:
        arbitrary_types_allowed = True
    
    def _get_relevant_documents(
        self, query: str, *, run_manager: Callbacks = None
    ) -> List[Document]:
        """Get documents relevant for a query."""
        try:
            results = self.store.similarity_search(query)
            return [
                Document(
                    page_content=result['content'],
                    metadata=result.get('metadata', {})
                )
                for result in results
            ]
        except Exception as e:
            print(f"Error retrieving documents: {e}")
            return []
    
    async def _aget_relevant_documents(
        self,
        query: str,
        *,
        run_manager: Callbacks = None,
    ) -> List[Document]:
        """Async version of document retrieval."""
        return await asyncio.to_thread(self._get_relevant_documents, query)

class ChatSystem:
    def __init__(self):
        # Initialize Elasticsearch
        self.es_client = Elasticsearch(
            cloud_id=os.getenv('ES_CLOUD_ID'),
            http_auth=(os.getenv('ES_USERNAME'),os.getenv('ES_PASSWORD'))
        )
        
        # Initialize Gemini
        self.gemini = GeminiConnector()
        
        # Initialize components with existing store
        self.vector_store = CustomElasticsearchStore(es_client=self.es_client)
        self.chat_chain = None
        self.session_id = str(uuid4())
        self.chat_history = None
        
        # Persona configuration
        self.persona = {
            'name': 'Jaksee',
            'role': 'Virtual Assistant for Jakarta Information',
            'personality': {
                'id': """
                Kamu adalah Jaksee, asisten virtual yang memiliki pengetahuan mendalam tentang Jakarta. 
                Kamu selalu menjawab dengan ramah, sopan, dan informatif menggunakan bahasa Indonesia yang baik dan benar.
                
                Jika Kamu tidak memiliki informasi yang tepat, kamu akan mengatakan:
                "Maaf, untuk pertanyaan tersebut saya belum memiliki informasi yang akurat. Namun, saya bisa membantu Anda dengan informasi lain seputar Jakarta. Apa yang ingin Anda ketahui?"
                """,
                'en': """
                You are Jaksee, a virtual assistant with deep knowledge about Jakarta. 
                You always respond in a friendly, polite, and informative manner.
                
                If You don't have accurate information, I will say:
                "I apologize, I don't have accurate information for that question at the moment. However, I'd be happy to help you with other information about Jakarta. What would you like to know?"
                """
            }
        }
        
        # Language mapping
        self.language_prompts = {
            'id': f"{self.persona['personality']['id']}\n\nJawablah dalam Bahasa Indonesia yang formal dan sopan.",
            'en': f"{self.persona['personality']['en']}\n\nPlease answer in proper English.",
            'default': f"I am {self.persona['name']}, Jakarta's virtual assistant. Please answer appropriately."
        }

    def detect_language(self, text: str) -> str:
        """Detect the language of input text."""
        try:
            return detect(text)
        except:
            return 'en'  # Default to English if detection fails

    def get_language_prompt(self, lang_code: str) -> str:
        """Get appropriate language instruction based on detected language."""
        return self.language_prompts.get(lang_code, self.language_prompts['default'])

    def initialize_chat(self):
        """Initialize the chat system with LLM and retriever."""
        try:
            # Initialize LLM
            llm = GeminiLLM(client=self.gemini, model="gemini-1.5-flash")
            
            # Create retriever from vector store
            retriever = CustomRetriever(store=self.vector_store)
            
            # Create chat history index if it doesn't exist
            if not self.es_client.indices.exists(index="workplace-docs-chat-history"):
                self.es_client.indices.create(index="workplace-docs-chat-history")
            
            # Setup chat history
            self.chat_history = ElasticsearchChatMessageHistory(
                es_cloud_id=os.getenv('ES_CLOUD_ID'),
                es_user=os.getenv('ES_USERNAME'),
                es_password=os.getenv('ES_PASSWORD'),
                index="workplace-docs-chat-history",
                session_id=self.session_id
            )
            
            # Create chat chain
            self.chat_chain = ConversationalRetrievalChain.from_llm(
                llm=llm,
                retriever=retriever,
                return_source_documents=True,
            )
            
            print(f"Chat session initialized with ID: {self.session_id}")

            
        except Exception as e:
            print(f"Error initializing chat system: {e}")
            raise

    def ask(self, question: str):
        """Ask a question and get a response."""
        if not self.chat_chain:
            raise ValueError("Chat system not initialized. Call initialize_chat() first.")
        
        # Detect language and get appropriate prompt
        detected_lang = self.detect_language(question)
        system_prompt = self.get_language_prompt(detected_lang)
        
        # Create the full prompt with persona
        full_prompt = f"""
        {system_prompt}

        Rules: 
            Respond naturally and in a friendly manner.
            Never say phrases like "The provided text..." or "Based on the provided context...".
            If you don't have accurate information, use the predefined answer templates provided above.
                    
        -----------------------
        Context History:
        {self.chat_history.messages if self.chat_history.messages else 'Ini adalah awal percakapan.'}

        -----------------------
        Question: {question}
        
        Hard Rules:
        If the question is a greeting (e.g., "hi," "hello" or say his name) or a farewell (e.g., "thank you," "goodbye"), respond friendly without using the Context.
        Otherwise, please answer politely and clearly in the language the user is using.

        
        """
        

        print(full_prompt)

        # Get response with persona context
        result = self.chat_chain({
            "question": full_prompt,
            "chat_history": self.chat_history.messages,
        })
        
        # Print response with index information
        source_index = 'unknown'
        if result.get('source_documents'):
            first_doc = result['source_documents'][0]
            if hasattr(first_doc, 'metadata'):
                source_index = first_doc.metadata.get('index', 'unknown')
  
        
        # Update chat history
        self.chat_history.add_user_message(question)
        self.chat_history.add_ai_message(result["answer"])

        
        return {'question':question,'answer':result["answer"]}