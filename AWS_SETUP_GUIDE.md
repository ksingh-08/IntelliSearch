# Second Brain - AWS Bedrock Integration Setup Guide

## Overview
Second Brain now supports **Amazon Bedrock** and **Amazon Q** for enterprise-grade AI capabilities. This integration allows you to use AWS's powerful embedding models and language models for your knowledge base.

## What's New
- **Amazon Bedrock Embeddings**: Use Titan and Cohere embedding models for text and image understanding
- **Amazon Bedrock LLMs**: Access Claude 3.5, Titan, Llama 3, and Mistral models for AI-powered search
- **Scalable & Secure**: Enterprise-grade infrastructure with AWS security
- **Cost-Effective**: Pay-per-use pricing with no upfront costs

## Installation

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

This will install the core dependencies plus AWS-specific packages:
- `boto3` - AWS SDK for Python
- `botocore` - Low-level AWS client
- `langchain-aws` - LangChain AWS integrations
- `langchain-community` - Additional LangChain tools

### 2. AWS Account Setup

#### Option A: AWS Console (Recommended)
1. Go to [AWS Console](https://console.aws.amazon.com/)
2. Sign up for a free tier account if you don't have one
3. Navigate to **IAM** (Identity and Access Management)
4. Create a new user with programmatic access
5. Attach the policy: `AmazonBedrockFullAccess`
6. Save your **Access Key ID** and **Secret Access Key**

#### Option B: AWS CLI
```bash
# Install AWS CLI
pip install awscli

# Configure credentials
aws configure
# Enter your Access Key ID
# Enter your Secret Access Key
# Default region: us-east-1
# Default output format: json
```

### 3. Enable Bedrock Models
1. Go to [Amazon Bedrock Console](https://console.aws.amazon.com/bedrock/)
2. Click on **Model access** in the left sidebar
3. Click **Manage model access**
4. Enable the following models (recommended):
   - **Amazon Titan Embed Text v2** (for text embeddings)
   - **Amazon Titan Embed Image v1** (for image embeddings)
   - **Claude 3.5 Haiku** (for fast AI responses)
   - **Claude 3.5 Sonnet** (for high-quality AI responses)
5. Click **Save changes** and wait for approval (usually instant)

### 4. Configure IntelliSearch AI

Run IntelliSearch AI and go to **Settings**:

#### For Embeddings:
1. **Use AWS Bedrock**: Toggle ON
2. **AWS Bedrock Text Model**: Select `titan-embed-text-v2`
3. **AWS Bedrock Image Model**: Select `titan-embed-image-v1`
4. **AWS Region**: Select your closest region (e.g., `us-east-1`)
5. **AWS Access Key ID**: Enter your access key (or leave empty to use AWS CLI credentials)
6. **AWS Secret Access Key**: Enter your secret key (or leave empty to use AWS CLI credentials)

#### For AI Mode:
1. **LLM Backend**: Select `AWS Bedrock`
2. **AWS Bedrock LLM Model**: Select `claude-3-5-haiku` (or `claude-3-5-sonnet` for better quality)

#### Save Settings:
1. Click **Save & Close**
2. The backend will reload with Bedrock integration

### 5. Sync Your Directory
1. Click **Sync Directory** to start embedding your files with AWS Bedrock
2. Files will be embedded using Titan models and stored in ChromaDB
3. You can now search with enterprise-grade embeddings!

### 6. Enable AI Mode (Optional)
1. Toggle **AI Mode** ON
2. IntelliSearch AI will load the Claude model from Bedrock
3. Enjoy AI-powered search with query expansion, result filtering, and insights!

## Architecture

### How It Works:
```
┌─────────────────┐
│   Your Files    │
│  (PDFs, docs,   │
│    images)      │
└────────┬────────┘
         │
         v
┌─────────────────┐
│  Text Splitter  │  ← Chunks documents
└────────┬────────┘
         │
         v
┌─────────────────┐
│ AWS Bedrock     │
│ Titan Embeddings│  ← Converts to vectors
└────────┬────────┘
         │
         v
┌─────────────────┐
│   ChromaDB      │  ← Stores vectors locally
│  Vector Store   │
└────────┬────────┘
         │
         v
┌─────────────────┐
│  Hybrid Search  │  ← Semantic + Lexical
│  (BM25 + MMR)   │
└────────┬────────┘
         │
         v
┌─────────────────┐
│ AWS Bedrock     │
│ Claude 3.5      │  ← AI insights
└─────────────────┘
```

## Cost Estimates (AWS Bedrock Pricing)

### Embeddings:
- **Titan Embed Text v2**: $0.0001 per 1K tokens (~$0.10 per 1M tokens)
- **Titan Embed Image v1**: $0.00006 per image

### LLMs (per 1K tokens):
- **Claude 3.5 Haiku**: Input $0.80 / Output $4.00 per 1M tokens
- **Claude 3.5 Sonnet**: Input $3.00 / Output $15.00 per 1M tokens
- **Claude 3 Opus**: Input $15.00 / Output $75.00 per 1M tokens

### Example Usage Cost:
- Embedding 10,000 documents (avg 500 tokens each): **~$0.50**
- 100 AI queries with Claude 3.5 Haiku: **~$0.05**
- **Total estimated cost: < $1** for typical usage

## Troubleshooting

### Error: "Access Denied" or "ModelNotFound"
- Make sure you've enabled model access in the Bedrock console
- Check that your IAM user has `AmazonBedrockFullAccess` permissions
- Verify your AWS region matches where models are enabled

### Error: "NoCredentialsError"
- Ensure AWS credentials are configured via AWS CLI or in settings
- Try running `aws configure` to set up credentials
- Check that your access keys are valid and not expired

### Error: "ThrottlingException"
- AWS has rate limits on Bedrock API calls
- Reduce batch size in settings (default is 16, try 8 or 4)
- Add delays between requests if needed

### Slow Performance
- Use a closer AWS region (check latency)
- Enable GPU acceleration for local processing
- Use smaller embedding models (Titan v1 instead of v2)

## Features Comparison

| Feature | Local Models | OpenAI | AWS Bedrock |
|---------|--------------|--------|-------------|
| **Privacy** | Fully Local | Cloud-based | Your AWS VPC |
| **Cost** | Free | ~$0.50/query | ~$0.01/query |
| **Setup** | Requires GPU | API Key only | AWS Account |
| **Speed** | Depends on HW | Fast | Very Fast |
| **Quality** | Varies | Excellent | Excellent |
| **Offline** | Yes | No | No |
| **Scalability** | Limited | High | Enterprise |

## Key Benefits

### Why AWS Bedrock
1. **Enterprise-Ready**: Scales beyond personal use
2. **Security**: Data stays in your AWS account
3. **Cost-Effective**: Pay-per-use with no infrastructure overhead
4. **Flexibility**: Support multiple providers (Local, OpenAI, AWS)
5. **Best Models**: Access to Claude 3.5 and Titan optimized embeddings

### Core Features
- **RAG (Retrieval-Augmented Generation)**: Grounds AI responses in your data
- **Hybrid Search**: Combines semantic (meaning) and lexical (keywords) search
- **MMR Reranking**: Ensures diverse, relevant results
- **Multimodal**: Works with text, PDFs, images, and more
- **Learning Aid**: Perfect for students, researchers, and professionals
- **Productivity Tool**: Find information 10x faster than manual search

## Amazon Q Integration (Future Enhancement)

While this version focuses on Amazon Bedrock, here's how to extend it with **Amazon Q**:

### Amazon Q Use Cases:
1. **Code Assistance**: Help developers understand codebases
2. **Document Q&A**: Chat with your documents
3. **Business Intelligence**: Query structured data

### Integration Steps:
1. Enable Amazon Q in your AWS account
2. Create a Q application
3. Add a `AmazonQIntegration.py` module
4. Update frontend to support Q-specific features

## Resources

- [Amazon Bedrock Documentation](https://docs.aws.amazon.com/bedrock/)
- [Amazon Q Documentation](https://docs.aws.amazon.com/amazonq/)
- [AWS Free Tier](https://aws.amazon.com/free/)
- [Boto3 Documentation](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html)
- [LangChain AWS Integration](https://python.langchain.com/docs/integrations/platforms/aws)

## Support

If you encounter issues:
1. Check the logs (enable "Log Messages" in settings)
2. Verify AWS credentials with `aws sts get-caller-identity`
3. Test Bedrock connection: `aws bedrock list-foundation-models --region us-east-1`
4. Review the [AWS Bedrock Troubleshooting Guide](https://docs.aws.amazon.com/bedrock/latest/userguide/troubleshooting.html)

## License

IntelliSearch AI with AWS Bedrock integration for enhanced AI-powered knowledge management.
