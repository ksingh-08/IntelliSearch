# AWS Credentials Configuration for IntelliSearch AI

This file provides instructions for setting up AWS credentials for Amazon Bedrock integration.

## Option 1: AWS CLI (Recommended)

The easiest way to configure AWS credentials is using the AWS CLI:

```bash
# Install AWS CLI
pip install awscli

# Configure credentials
aws configure
```

You'll be prompted for:
- **AWS Access Key ID**: Your AWS access key
- **AWS Secret Access Key**: Your AWS secret key
- **Default region name**: `us-east-1` (or your preferred region)
- **Default output format**: `json`

Your credentials will be stored in:
- **Linux/Mac**: `~/.aws/credentials`
- **Windows**: `C:\Users\USERNAME\.aws\credentials`

## Option 2: Environment Variables

Set environment variables in your shell:

### Linux/Mac:
```bash
export AWS_ACCESS_KEY_ID=your_access_key_here
export AWS_SECRET_ACCESS_KEY=your_secret_key_here
export AWS_DEFAULT_REGION=us-east-1
```

Add these to your `~/.bashrc`, `~/.zshrc`, or `~/.profile` to make them permanent.

### Windows (PowerShell):
```powershell
$env:AWS_ACCESS_KEY_ID="your_access_key_here"
$env:AWS_SECRET_ACCESS_KEY="your_secret_key_here"
$env:AWS_DEFAULT_REGION="us-east-1"
```

### Windows (Command Prompt):
```cmd
setx AWS_ACCESS_KEY_ID "your_access_key_here"
setx AWS_SECRET_ACCESS_KEY "your_secret_key_here"
setx AWS_DEFAULT_REGION "us-east-1"
```

## Option 3: IntelliSearch AI Settings

You can enter credentials directly in IntelliSearch AI settings:

1. Open IntelliSearch AI
2. Click **Settings**
3. Find the AWS section:
   - **AWS Access Key ID**: Enter your access key
   - **AWS Secret Access Key**: Enter your secret key
   - **AWS Region**: Select your region
4. Click **Save & Close**

**Note**: Credentials entered in settings are stored in `config.json`. For better security, use AWS CLI or environment variables instead.

## Option 4: IAM Role (For EC2/ECS)

If running IntelliSearch AI on AWS infrastructure (EC2, ECS, Lambda), use IAM roles:

1. Create an IAM role with `AmazonBedrockFullAccess` policy
2. Attach the role to your EC2 instance or ECS task
3. Leave AWS credentials empty in IntelliSearch AI settings
4. Boto3 will automatically use the IAM role

## Getting AWS Access Keys

### Step 1: Sign in to AWS Console
Go to [AWS Console](https://console.aws.amazon.com/)

### Step 2: Navigate to IAM
- Search for "IAM" in the AWS Console search bar
- Click on **IAM** (Identity and Access Management)

### Step 3: Create a User
1. Click **Users** in the left sidebar
2. Click **Add users**
3. Enter a username (e.g., `intellisearch-ai-user`)
4. Select **Access key - Programmatic access**
5. Click **Next: Permissions**

### Step 4: Set Permissions
1. Click **Attach existing policies directly**
2. Search for `AmazonBedrockFullAccess`
3. Check the box next to the policy
4. Click **Next: Tags** (optional)
5. Click **Next: Review**
6. Click **Create user**

### Step 5: Save Your Keys
1. **Access key ID**: Copy and save this
2. **Secret access key**: Click "Show" and copy this
3. **⚠️ Important**: This is the only time you can view the secret key. Save it securely!

## Security Best Practices

### 1. Never Commit Credentials to Git
Add to your `.gitignore`:
```
config.json
.aws/
*.pem
*.key
```

### 2. Use IAM Roles When Possible
IAM roles provide temporary credentials and are more secure than long-term access keys.

### 3. Rotate Keys Regularly
Change your access keys every 90 days:
```bash
aws iam create-access-key --user-name intellisearch-ai-user
aws iam delete-access-key --access-key-id OLD_KEY_ID --user-name intellisearch-ai-user
```

### 4. Restrict Permissions
Create a custom policy with minimal permissions:

```json
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Action": [
        "bedrock:InvokeModel",
        "bedrock:InvokeModelWithResponseStream",
        "bedrock:ListFoundationModels"
      ],
      "Resource": "*"
    }
  ]
}
```

### 5. Enable MFA
Add multi-factor authentication to your AWS account for extra security.

### 6. Monitor Usage
Check AWS CloudTrail for API calls and set up billing alerts.

## Troubleshooting

### Error: "Unable to locate credentials"
**Solution**: 
- Run `aws configure` to set up credentials
- Or set environment variables
- Or enter credentials in IntelliSearch AI settings

### Error: "An error occurred (InvalidClientTokenId)"
**Solution**: 
- Your access key ID is incorrect
- Verify the key in AWS IAM console
- Regenerate keys if needed

### Error: "An error occurred (SignatureDoesNotMatch)"
**Solution**: 
- Your secret access key is incorrect
- Check for extra spaces or characters
- Regenerate keys if needed

### Error: "An error occurred (AccessDeniedException)"
**Solution**: 
- Your user doesn't have Bedrock permissions
- Add `AmazonBedrockFullAccess` policy to your IAM user
- Or create a custom policy with required permissions

### Check Current Credentials
```bash
# Verify your credentials are configured
aws sts get-caller-identity

# List available Bedrock models
aws bedrock list-foundation-models --region us-east-1
```

## AWS Free Tier

AWS Free Tier includes:
- **No free tier for Bedrock** (pay-per-use only)
- But costs are very low: ~$0.0001 per 1K tokens
- First-time users get $300 in credits for 12 months (may vary)

## Cost Management

### Set Up Billing Alerts
1. Go to [AWS Billing Console](https://console.aws.amazon.com/billing/)
2. Click **Billing Preferences**
3. Enable **Receive Billing Alerts**
4. Create a CloudWatch alarm for spending > $10

### Monitor Bedrock Usage
1. Go to [AWS Cost Explorer](https://console.aws.amazon.com/cost-management/)
2. Filter by service: **Amazon Bedrock**
3. View daily/monthly costs

### Estimate Costs
Use the IntelliSearch AI cost calculator:
- 10,000 documents × 500 tokens = 5M tokens
- 5M tokens × $0.0001 per 1K tokens = **$0.50**

## Additional Resources

- [AWS IAM Best Practices](https://docs.aws.amazon.com/IAM/latest/UserGuide/best-practices.html)
- [AWS Security Credentials](https://docs.aws.amazon.com/general/latest/gr/aws-sec-cred-types.html)
- [Boto3 Credentials Guide](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html)
- [AWS Bedrock Pricing](https://aws.amazon.com/bedrock/pricing/)

---

**Need Help?** Check the logs in IntelliSearch AI (enable "Log Messages" in settings) for detailed error messages.
