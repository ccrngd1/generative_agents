import boto3
import os
from litellm import completion
 
sts_client = boto3.client('sts') 

role_name = "arn:aws:iam::791580863750:role/EpoxyChronicleInstanceRole"
rolesession_name = "bedrockSession"

assumed_role_object = sts_client.assume_role(
    RoleArn=role_name,
    RoleSessionName=rolesession_name
)

credentials = assumed_role_object['Credentials'] 

os.environ["AWS_ACCESS_KEY_ID"] = aws_access_key_id=credentials['AccessKeyId']
os.environ["AWS_SECRET_ACCESS_KEY"] = aws_secret_access_key=credentials['SecretAccessKey']
os.environ["AWS_REGION_NAME"] = aws_region = "us-east-1"
aws_sessiontoken=credentials['SessionToken']

bedrock = boto3.client(
            service_name="bedrock-runtime",
            region_name="us-east-1",
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            aws_session_token=aws_sessiontoken,
)

response = completion(
  model="bedrock/anthropic.claude-3-5-sonnet-20240620-v1:0",
  messages=[{ "content": "Hello, how are you?","role": "user"}],
    aws_bedrock_client=bedrock,
)

#print(response)

print(response['choices'][0].message.content)