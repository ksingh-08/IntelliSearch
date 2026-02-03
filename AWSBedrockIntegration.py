"""
AWS Bedrock Integration Module for IntelliSearch AI
Provides embeddings and LLM capabilities using Amazon Bedrock
"""

import os
import json
import base64
from typing import List, Dict, Any, Optional
from pathlib import Path
import numpy as np

try:
    import boto3
    from botocore.exceptions import ClientError, NoCredentialsError
    BOTO3_AVAILABLE = True
except ImportError:
    BOTO3_AVAILABLE = False
    print("Warning: boto3 not installed. AWS Bedrock features will be disabled.")

# Constants for Bedrock model IDs
BEDROCK_EMBEDDING_MODELS = {
    "titan-embed-text-v1": "amazon.titan-embed-text-v1",
    "titan-embed-text-v2": "amazon.titan-embed-text-v2:0",
    "titan-embed-image-v1": "amazon.titan-embed-image-v1",
    "cohere-embed-english-v3": "cohere.embed-english-v3",
    "cohere-embed-multilingual-v3": "cohere.embed-multilingual-v3"
}

BEDROCK_LLM_MODELS = {
    "claude-3-5-sonnet": "anthropic.claude-3-5-sonnet-20241022-v2:0",
    "claude-3-5-haiku": "anthropic.claude-3-5-haiku-20241022-v1:0",
    "claude-3-opus": "anthropic.claude-3-opus-20240229-v1:0",
    "claude-3-sonnet": "anthropic.claude-3-sonnet-20240229-v1:0",
    "claude-3-haiku": "anthropic.claude-3-haiku-20240307-v1:0",
    "titan-text-express": "amazon.titan-text-express-v1",
    "titan-text-lite": "amazon.titan-text-lite-v1",
    "llama3-70b": "meta.llama3-70b-instruct-v1:0",
    "llama3-8b": "meta.llama3-8b-instruct-v1:0",
    "mistral-7b": "mistral.mistral-7b-instruct-v0:2",
    "mixtral-8x7b": "mistral.mixtral-8x7b-instruct-v0:1"
}

def _log(msg: str, log_callback=None):
    """Send log message to UI if available, else fallback to print."""
    if log_callback:
        log_callback(msg)
    else:
        print(msg)

class BedrockEmbeddings:
    """Wrapper for Amazon Bedrock Embeddings to match SentenceTransformer interface"""
    
    def __init__(self, model_name: str = "titan-embed-text-v2", region_name: str = "us-east-1", 
                 aws_access_key: str = None, aws_secret_key: str = None, log_callback=None):
        """
        Initialize Bedrock embeddings client.
        
        Args:
            model_name: One of the BEDROCK_EMBEDDING_MODELS keys
            region_name: AWS region (default: us-east-1)
            aws_access_key: AWS access key (optional, will use environment/IAM if not provided)
            aws_secret_key: AWS secret key (optional, will use environment/IAM if not provided)
            log_callback: Function to call for logging
        """
        if not BOTO3_AVAILABLE:
            raise ImportError("boto3 is required for Bedrock integration. Install with: pip install boto3")
        
        self.log_callback = log_callback
        self.model_name = model_name
        self.model_id = BEDROCK_EMBEDDING_MODELS.get(model_name, model_name)
        
        # Initialize Bedrock client
        try:
            if aws_access_key and aws_secret_key:
                self.client = boto3.client(
                    service_name='bedrock-runtime',
                    region_name=region_name,
                    aws_access_key_id=aws_access_key,
                    aws_secret_access_key=aws_secret_key
                )
            else:
                # Use default credentials (environment variables, IAM role, or credentials file)
                self.client = boto3.client(
                    service_name='bedrock-runtime',
                    region_name=region_name
                )
            _log(f"✓ Bedrock client initialized with model: {self.model_id}", log_callback)
        except NoCredentialsError:
            _log("[ERROR] AWS credentials not found. Please configure AWS credentials.", log_callback)
            raise
        except Exception as e:
            _log(f"[ERROR] Failed to initialize Bedrock client: {e}", log_callback)
            raise
    
    def encode(self, texts, convert_to_numpy=True, batch_size=1, normalize_embeddings=True, **kwargs):
        """
        Encode texts into embeddings (matches SentenceTransformer interface).
        
        Args:
            texts: String or list of strings to embed
            convert_to_numpy: If True, return numpy array
            batch_size: Batch size for encoding (currently processes one at a time)
            normalize_embeddings: If True, normalize embeddings to unit length
            
        Returns:
            Embeddings as numpy array or list
        """
        # Handle single string input
        if isinstance(texts, str):
            texts = [texts]
        
        embeddings = []
        for text in texts:
            try:
                embedding = self._get_embedding(text)
                embeddings.append(embedding)
            except Exception as e:
                _log(f"[ERROR] Failed to embed text: {e}", self.log_callback)
                # Return zero vector on error
                embeddings.append([0.0] * 1024)  # Default dimension for Titan
        
        # Convert to numpy if requested
        if convert_to_numpy:
            embeddings = np.array(embeddings)
            
            # Normalize if requested
            if normalize_embeddings:
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                norms[norms == 0] = 1  # Avoid division by zero
                embeddings = embeddings / norms
        
        return embeddings
    
    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for a single text using Bedrock."""
        try:
            # Prepare request based on model type
            if "titan" in self.model_id:
                body = json.dumps({
                    "inputText": text
                })
            elif "cohere" in self.model_id:
                body = json.dumps({
                    "texts": [text],
                    "input_type": "search_document"
                })
            else:
                body = json.dumps({"inputText": text})
            
            # Invoke model
            response = self.client.invoke_model(
                modelId=self.model_id,
                body=body,
                contentType='application/json',
                accept='application/json'
            )
            
            # Parse response
            response_body = json.loads(response['body'].read())
            
            # Extract embedding based on model type
            if "titan" in self.model_id:
                embedding = response_body.get('embedding', [])
            elif "cohere" in self.model_id:
                embedding = response_body.get('embeddings', [[]])[0]
            else:
                embedding = response_body.get('embedding', [])
            
            return embedding
            
        except ClientError as e:
            error_code = e.response['Error']['Code']
            error_message = e.response['Error']['Message']
            _log(f"[ERROR] Bedrock API error ({error_code}): {error_message}", self.log_callback)
            raise
        except Exception as e:
            _log(f"[ERROR] Failed to get embedding: {e}", self.log_callback)
            raise


class BedrockImageEmbeddings:
    """Wrapper for Amazon Bedrock Image Embeddings"""
    
    def __init__(self, model_name: str = "titan-embed-image-v1", region_name: str = "us-east-1",
                 aws_access_key: str = None, aws_secret_key: str = None, log_callback=None):
        """Initialize Bedrock image embeddings client."""
        if not BOTO3_AVAILABLE:
            raise ImportError("boto3 is required for Bedrock integration.")
        
        self.log_callback = log_callback
        self.model_name = model_name
        self.model_id = BEDROCK_EMBEDDING_MODELS.get(model_name, "amazon.titan-embed-image-v1")
        
        try:
            if aws_access_key and aws_secret_key:
                self.client = boto3.client(
                    service_name='bedrock-runtime',
                    region_name=region_name,
                    aws_access_key_id=aws_access_key,
                    aws_secret_access_key=aws_secret_key
                )
            else:
                self.client = boto3.client(
                    service_name='bedrock-runtime',
                    region_name=region_name
                )
            _log(f"✓ Bedrock image client initialized with model: {self.model_id}", log_callback)
        except Exception as e:
            _log(f"[ERROR] Failed to initialize Bedrock image client: {e}", log_callback)
            raise
    
    def encode(self, images, convert_to_numpy=True, batch_size=1, normalize_embeddings=True, **kwargs):
        """
        Encode images into embeddings (matches SentenceTransformer interface).
        
        Args:
            images: PIL Image or list of PIL Images
            convert_to_numpy: If True, return numpy array
            batch_size: Batch size for encoding
            normalize_embeddings: If True, normalize embeddings
            
        Returns:
            Embeddings as numpy array or list
        """
        from PIL import Image
        import io
        
        # Handle single image input
        if isinstance(images, Image.Image):
            images = [images]
        
        embeddings = []
        for img in images:
            try:
                # Convert PIL Image to base64
                img_buffer = io.BytesIO()
                img.save(img_buffer, format='PNG')
                img_base64 = base64.b64encode(img_buffer.getvalue()).decode('utf-8')
                
                # Get embedding
                embedding = self._get_image_embedding(img_base64)
                embeddings.append(embedding)
            except Exception as e:
                _log(f"[ERROR] Failed to embed image: {e}", self.log_callback)
                embeddings.append([0.0] * 1024)  # Default dimension
        
        if convert_to_numpy:
            embeddings = np.array(embeddings)
            
            if normalize_embeddings:
                norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
                norms[norms == 0] = 1
                embeddings = embeddings / norms
        
        return embeddings
    
    def _get_image_embedding(self, image_base64: str) -> List[float]:
        """Get embedding for a single image using Bedrock."""
        try:
            body = json.dumps({
                "inputImage": image_base64
            })
            
            response = self.client.invoke_model(
                modelId=self.model_id,
                body=body,
                contentType='application/json',
                accept='application/json'
            )
            
            response_body = json.loads(response['body'].read())
            embedding = response_body.get('embedding', [])
            
            return embedding
            
        except Exception as e:
            _log(f"[ERROR] Failed to get image embedding: {e}", self.log_callback)
            raise


class BedrockLLM:
    """Wrapper for Amazon Bedrock LLM to match OpenAI/LMStudio interface"""
    
    def __init__(self, model_name: str = "claude-3-5-haiku", region_name: str = "us-east-1",
                 aws_access_key: str = None, aws_secret_key: str = None, 
                 temperature: float = 0.7, max_tokens: int = 2000, log_callback=None):
        """Initialize Bedrock LLM client."""
        if not BOTO3_AVAILABLE:
            raise ImportError("boto3 is required for Bedrock integration.")
        
        self.log_callback = log_callback
        self.model_name = model_name
        self.model_id = BEDROCK_LLM_MODELS.get(model_name, model_name)
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        try:
            if aws_access_key and aws_secret_key:
                self.client = boto3.client(
                    service_name='bedrock-runtime',
                    region_name=region_name,
                    aws_access_key_id=aws_access_key,
                    aws_secret_access_key=aws_secret_key
                )
            else:
                self.client = boto3.client(
                    service_name='bedrock-runtime',
                    region_name=region_name
                )
            _log(f"✓ Bedrock LLM initialized with model: {self.model_id}", log_callback)
        except Exception as e:
            _log(f"[ERROR] Failed to initialize Bedrock LLM: {e}", log_callback)
            raise
    
    def create_completion(self, messages: List[Dict[str, Any]], stream: bool = False, 
                          temperature: Optional[float] = None, max_tokens: Optional[int] = None):
        """
        Create a completion (matches OpenAI chat completion interface).
        
        Args:
            messages: List of message dicts with 'role' and 'content'
            stream: If True, stream the response
            temperature: Override default temperature
            max_tokens: Override default max_tokens
            
        Returns:
            Completion response or generator for streaming
        """
        temp = temperature if temperature is not None else self.temperature
        max_tok = max_tokens if max_tokens is not None else self.max_tokens
        
        if stream:
            return self._stream_completion(messages, temp, max_tok)
        else:
            return self._complete(messages, temp, max_tok)
    
    def _complete(self, messages: List[Dict[str, Any]], temperature: float, max_tokens: int):
        """Non-streaming completion."""
        try:
            # Format request based on model type
            if "claude" in self.model_id:
                body = self._format_claude_request(messages, temperature, max_tokens)
            elif "titan" in self.model_id:
                body = self._format_titan_request(messages, temperature, max_tokens)
            elif "llama" in self.model_id or "mistral" in self.model_id:
                body = self._format_llama_request(messages, temperature, max_tokens)
            else:
                body = self._format_generic_request(messages, temperature, max_tokens)
            
            response = self.client.invoke_model(
                modelId=self.model_id,
                body=json.dumps(body),
                contentType='application/json',
                accept='application/json'
            )
            
            response_body = json.loads(response['body'].read())
            
            # Extract text based on model type
            if "claude" in self.model_id:
                text = response_body.get('content', [{}])[0].get('text', '')
            elif "titan" in self.model_id:
                text = response_body.get('results', [{}])[0].get('outputText', '')
            else:
                text = response_body.get('generation', '')
            
            # Return in OpenAI-compatible format
            return {
                'choices': [{
                    'message': {
                        'role': 'assistant',
                        'content': text
                    }
                }]
            }
            
        except Exception as e:
            _log(f"[ERROR] Bedrock completion failed: {e}", self.log_callback)
            raise
    
    def _stream_completion(self, messages: List[Dict[str, Any]], temperature: float, max_tokens: int):
        """Streaming completion generator."""
        try:
            if "claude" in self.model_id:
                body = self._format_claude_request(messages, temperature, max_tokens)
            elif "titan" in self.model_id:
                body = self._format_titan_request(messages, temperature, max_tokens)
            else:
                body = self._format_generic_request(messages, temperature, max_tokens)
            
            response = self.client.invoke_model_with_response_stream(
                modelId=self.model_id,
                body=json.dumps(body),
                contentType='application/json',
                accept='application/json'
            )
            
            # Stream the response
            for event in response.get('body'):
                chunk = json.loads(event['chunk']['bytes'])
                
                # Extract text based on model type
                if "claude" in self.model_id:
                    if chunk.get('type') == 'content_block_delta':
                        delta = chunk.get('delta', {})
                        text = delta.get('text', '')
                        if text:
                            yield {
                                'choices': [{
                                    'delta': {'content': text}
                                }]
                            }
                elif "titan" in self.model_id:
                    text = chunk.get('outputText', '')
                    if text:
                        yield {
                            'choices': [{
                                'delta': {'content': text}
                            }]
                        }
                
        except Exception as e:
            _log(f"[ERROR] Bedrock streaming failed: {e}", self.log_callback)
            raise
    
    def _format_claude_request(self, messages: List[Dict], temperature: float, max_tokens: int) -> Dict:
        """Format request for Claude models."""
        # Extract system message if present
        system_msg = ""
        filtered_messages = []
        for msg in messages:
            if msg['role'] == 'system':
                system_msg = msg['content']
            else:
                filtered_messages.append(msg)
        
        body = {
            "anthropic_version": "bedrock-2023-05-31",
            "max_tokens": max_tokens,
            "temperature": temperature,
            "messages": filtered_messages
        }
        
        if system_msg:
            body["system"] = system_msg
        
        return body
    
    def _format_titan_request(self, messages: List[Dict], temperature: float, max_tokens: int) -> Dict:
        """Format request for Titan models."""
        # Combine all messages into a single prompt
        prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
        
        return {
            "inputText": prompt,
            "textGenerationConfig": {
                "temperature": temperature,
                "maxTokenCount": max_tokens,
                "topP": 0.9
            }
        }
    
    def _format_llama_request(self, messages: List[Dict], temperature: float, max_tokens: int) -> Dict:
        """Format request for Llama/Mistral models."""
        # Combine messages into chat format
        prompt = ""
        for msg in messages:
            role = msg['role']
            content = msg['content']
            if role == 'system':
                prompt += f"[INST] {content} [/INST]\n"
            elif role == 'user':
                prompt += f"[INST] {content} [/INST]\n"
            else:
                prompt += f"{content}\n"
        
        return {
            "prompt": prompt,
            "temperature": temperature,
            "max_gen_len": max_tokens,
            "top_p": 0.9
        }
    
    def _format_generic_request(self, messages: List[Dict], temperature: float, max_tokens: int) -> Dict:
        """Generic request format."""
        prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
        return {
            "prompt": prompt,
            "temperature": temperature,
            "max_tokens": max_tokens
        }


def test_bedrock_connection(region_name: str = "us-east-1", 
                           aws_access_key: str = None, 
                           aws_secret_key: str = None,
                           log_callback=None) -> bool:
    """
    Test if Bedrock connection is working.
    
    Returns:
        True if connection successful, False otherwise
    """
    try:
        if not BOTO3_AVAILABLE:
            _log("[ERROR] boto3 not installed", log_callback)
            return False
        
        if aws_access_key and aws_secret_key:
            client = boto3.client(
                service_name='bedrock',
                region_name=region_name,
                aws_access_key_id=aws_access_key,
                aws_secret_access_key=aws_secret_key
            )
        else:
            client = boto3.client(
                service_name='bedrock',
                region_name=region_name
            )
        
        # Try to list foundation models
        response = client.list_foundation_models()
        _log(f"✓ Successfully connected to Bedrock in {region_name}", log_callback)
        _log(f"  Found {len(response.get('modelSummaries', []))} available models", log_callback)
        return True
        
    except NoCredentialsError:
        _log("[ERROR] AWS credentials not found", log_callback)
        return False
    except Exception as e:
        _log(f"[ERROR] Bedrock connection test failed: {e}", log_callback)
        return False
