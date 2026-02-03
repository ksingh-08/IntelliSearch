# IntelliSearch AI - Enterprise Knowledge Base with AWS Bedrock
## Intelligent RAG System for Enhanced Learning and Productivity

---

## Features

- **AWS Bedrock Integration**: Enterprise-grade embeddings using Amazon Titan and Cohere models
- **Claude 3.5 AI Models**: Advanced language understanding via AWS Bedrock (Haiku, Sonnet, Opus)
- **Semantic Search**: Searches based on deep semantic meaning using vector embeddings
- **Lexical Search**: Finds exact keywords using BM25 algorithm
- **Hybrid Search Architecture**: Combines semantic and lexical approaches with MMR reranking
- **Multimodal Support**: 
  * Embed and search both text and images
  * Attach documents and images for contextual search
  * Vision-enabled AI models for image understanding
- **Multiple AI Backends**:
  * **AWS Bedrock**: Claude 3.5, Titan, Llama 3, Mistral models
  * **Local Models**: LM Studio for private, offline operation
  * **OpenAI**: GPT models via API
- **RAG (Retrieval-Augmented Generation)**: AI responses grounded in your knowledge base
- **Supported Files**: .txt, .pdf, .docx, .gdoc, .png, .jpg, .jpeg, .gif, .webp
- **Scalable**: Sync 100,000+ files efficiently
- **Flexible Deployment**: Local embeddings or AWS Bedrock cloud processing
- **Privacy Options**: Local-only mode or secure AWS VPC deployment
- **Google Drive Integration**: Sync and search Google Docs
- **Offline Capable**: Works with downloaded models when internet unavailable

---

## Architecture

### System Overview

```
Documents + Images
        |
        v
Text Chunking & Processing
        |
        v
Embedding Generation (AWS Bedrock Titan / Local Models)
        |
        v
ChromaDB Vector Store (Local)
        |
        v
Hybrid Search (Semantic + Lexical BM25)
        |
        v
MMR Reranking (Relevance + Diversity)
        |
        v
AI Synthesis (AWS Bedrock Claude / OpenAI / Local)
        |
        v
Results + Insights
```

## How It Works

### Backend (SecondBrainBackend.py)

**Syncing:** Scans directories for new, updated, or deleted files. Extracts text and splits into chunks. Generates vector embeddings using AWS Bedrock Titan models or local SentenceTransformers. Stores embeddings in ChromaDB vector database. Creates BM25 lexical index for keyword search. Generates image captions for multimodal search.

**Retrieval:** Executes hybrid search combining semantic (vector similarity) and lexical (BM25 keyword) approaches. Reranks results using Maximal Marginal Relevance (MMR) for optimal relevance and diversity. Optional AI filtering and synthesis using AWS Bedrock Claude models.

### Frontend (SecondBrainFrontend.py)

Modern UI built with Flet framework. Manages threading for non-blocking operations. Handles AWS Bedrock, OpenAI, and local LLM integrations. Supports text and image attachments for contextual queries. Interactive settings for model selection and configuration. Real-time streaming of AI responses. Attachment system for continuous, context-aware searches.

### Configuration Files

- **config.json**: Auto-generated configuration for AWS Bedrock settings, model selection, and search parameters
- **AWSBedrockIntegration.py**: AWS Bedrock wrapper classes for embeddings and LLMs
- **credentials.json**: (Optional) Google Drive OAuth credentials
- **image_labels.csv**: Image classification labels from Google Open Images dataset
- **icon.ico**: Application icon

---

## Getting Started

### 1. Prerequisites
- Python 3.9 or higher
- AWS Account with Bedrock access (for AWS features)
- AWS CLI configured (run `aws configure`)
- (Optional) OpenAI API key
- (Optional) LM Studio for local models
- GPU recommended but not required (CPU supported)

### 2. Installation
```bash
# Clone repository
git clone <repository-url>
cd <repository-folder>

# Install dependencies
pip install -r requirements.txt
```

### 3. AWS Bedrock Setup
```bash
# Configure AWS credentials
aws configure

# Test Bedrock integration
python test_bedrock_integration.py
```

Enable these models in AWS Bedrock Console:
- Amazon Titan Embed Text v2
- Amazon Titan Embed Image v1
- Claude 3.5 Haiku or Sonnet

### 4. Configuration
Run the application and configure in Settings:
- **Sync Directory**: Your knowledge base folder
- **Use AWS Bedrock**: Enable for AWS embeddings
- **LLM Backend**: Choose "AWS Bedrock", "OpenAI", or "LM Studio"
- **AWS Region**: Select closest region (e.g., us-east-1)
- **AWS Credentials**: Enter keys or leave empty to use AWS CLI

### 5. Running the Application
```bash
python IntelliSearchFrontend.py
```

For Google Drive (optional):
- Follow [Google Cloud API](https://developers.google.com/workspace/drive/api/guides/about-sdk) instructions
- Add credentials.json to project folder
- Authorize on first sync

---

## Usage Guide

### Syncing Files
Click **Sync Directory** to embed files using AWS Bedrock or local models. Process runs incrementally - cancel and resume anytime. GPU acceleration available for faster processing.

### Searching
- Type query and press Enter
- Results show text chunks and related images
- Click results to open, attach, or view source
- Hybrid search combines semantic and keyword matching

### AI Mode
Toggle **AI Mode** to enable:
- AWS Bedrock Claude models for advanced reasoning
- Query expansion for comprehensive results
- AI-powered relevance filtering
- Synthesized insights with source citations
- Vision support for image understanding

Configure in Settings:
- LLM Query Expansion (0-10 queries)
- LLM Filtering (enable/disable)
- Choose backend (AWS Bedrock, OpenAI, LM Studio)

### Attaching Files
- Click attachment icon to add context
- Documents: Full text if under size limit, relevant chunks otherwise
- Images: Find similar images and related documents
- Send attachments without messages for pure similarity search

### Saving Insights
Click **Save Insight** to store AI responses as embedded documents. Creates searchable memory of previous interactions stored in saved_insights folder.

---

## Configuration Details

Settings can be modified through the UI Settings page or by editing config.json directly.

### AWS Bedrock Settings
| Parameter | Description | Default |
|----|----|----|
| use_bedrock | Enable AWS Bedrock integration | false |
| bedrock_region | AWS region for Bedrock | us-east-1 |
| bedrock_text_model | Bedrock text embedding model | titan-embed-text-v2 |
| bedrock_image_model | Bedrock image embedding model | titan-embed-image-v1 |
| bedrock_llm_model | Bedrock LLM for AI mode | claude-3-5-haiku |
| aws_access_key_id | AWS access key (or use AWS CLI) | "" |
| aws_secret_access_key | AWS secret key (or use AWS CLI) | "" |

### Core Settings

| Parameter | Description | Range | Default |
|-----------|-------------|-------|---------|
| target_directory | Knowledge base directory path | Any directory | "C:\\Users\\user\\Documents" |
| text_model_name | Local text embedding model (when use_bedrock=false) | bge-small/large/m3 | "BAAI/bge-small-en-v1.5" |
| image_model_name | Local image embedding model (when use_bedrock=false) | clip-ViT-B-32/B-16/L-14 | "clip-ViT-B-32" |
| use_cuda | Enable GPU acceleration | true/false | true |
| batch_size | Parallel embedding batch size | 1-64 | 16 |
| chunk_size | Text chunk size in tokens | 100-2000 | 200 |
| chunk_overlap | Token overlap between chunks | 0-200 | 0 |
| mmr_lambda | Relevance (1.0) vs diversity (0.0) | 0.0-1.0 | 0.5 |
| mmr_alpha | Semantic (1.0) vs lexical (0.0) | 0.0-1.0 | 0.5 |
| search_multiplier | Results per query multiplier | 1-20 | 5 |
| max_results | Maximum search results | 1-30 | 6 |
| llm_backend | AI backend selection | "AWS Bedrock"/"OpenAI"/"LM Studio" | "LM Studio" |
| llm_filter_results | Enable AI result filtering | true/false | false |
| query_multiplier | AI query expansion count | 0-10 | 4 |
| max_attachment_size | Max attachment tokens | 200-8000 | 1000 |
| lms_model_name | LM Studio model name | Any model | "unsloth/gemma-3-4b-it" |
| openai_model_name | OpenAI model name | Any model | "gpt-4.1" |
| OPENAI_API_KEY | OpenAI API key | Any key | "" |
| use_drive | Enable Google Drive sync | true/false | true |
| credentials_path | Google Drive credentials path | Any path | "credentials.json" |

---

## Dependencies

Install all dependencies:
```bash
pip install -r requirements.txt
```

Key packages:
- **boto3** - AWS SDK for Bedrock integration
- **chromadb** - Vector database
- **sentence-transformers** - Local embedding models
- **flet** - UI framework
- **langchain** - Text processing
- **torch** - Deep learning (GPU support)
- **rank_bm25** - Lexical search
- **openai** - OpenAI API integration
- **lmstudio** - Local model integration

## Cost Comparison

| Backend | Setup | Cost per 1000 queries | Privacy | Offline |
|---------|-------|----------------------|---------|---------|
| AWS Bedrock | AWS account | ~$10 | AWS VPC | No |
| Local (LM Studio) | GPU recommended | Free | Fully private | Yes |
| OpenAI | API key | ~$500 | Cloud | No |

AWS Bedrock provides the best balance of cost, performance, and scalability for production use.

## Resources

- [AWS Bedrock Documentation](https://docs.aws.amazon.com/bedrock/)
- [ChromaDB Documentation](https://docs.trychroma.com/)
- [LangChain Documentation](https://python.langchain.com/)
- [AWS Setup Guide](AWS_SETUP_GUIDE.md)
- [AWS Credentials Guide](AWS_CREDENTIALS_GUIDE.md)

---

**IntelliSearch AI** - Enterprise-grade AI-powered knowledge management with AWS Bedrock  
