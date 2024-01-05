"""
Author: Nick Lawson (github.com/ccrngd1)

FIle: bedrock_structure.py
Description: Wrapper functions for calling Bedrock APIs
"""
import json
import random 
import time 
import os
import sys
import json 
import boto3 
import re
import traceback  
from typing import List, Union, Type

import numpy as np
from pydantic import BaseModel, Field  
from io import BytesIO 

from langchain.llms.bedrock import Bedrock
from langchain.embeddings import BedrockEmbeddings 
from langchain.prompts import StringPromptTemplate
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent, AgentOutputParser,initialize_agent
from langchain.schema import AgentAction, AgentFinish
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.document_loaders import PyPDFLoader, PyPDFDirectoryLoader
from langchain.chains.question_answering import load_qa_chain
from langchain.vectorstores import FAISS
from langchain.indexes import VectorstoreIndexCreator
from langchain.indexes.vectorstore import VectorStoreIndexWrapper
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain import  LLMChain
from langchain.docstore.base import Docstore
from langchain.docstore.document import Document 
from langchain.memory import ConversationBufferWindowMemory
from langchain.agents.react.base import DocstoreExplorer 
from langchain.tools import BaseTool

from bedrock import * 
from refreshableSession import *

if __name__ == "__main__":
    print('test')
    
    titanID = 'amazon.titan-tg1-large' 
    claudeID = 'anthropic.claude-v1'
    
    accept = 'application/json'
    contentType = 'application/json'

    #session = BotoSession(profile_name='xacct').refreshable_session()
    
    #sts_client = session.client("sts")
    #print(sts_client.get_caller_identity())
    
    #response = sts_client.assume_role(
    #    RoleArn='arn:aws:iam::567516816488:role/central-bedrock-access',
    #    RoleSessionName='test'
    #) 

    #boto3_bedrock = session.client(
    # service_name='bedrock', 
     #aws_access_key_id=response['Credentials']['AccessKeyId'],
     #aws_secret_access_key=response['Credentials']['SecretAccessKey'],
     #aws_session_token=response['Credentials']['SessionToken']
    #)  
    
    
    
    session = boto3.Session(profile_name='xacct', region_name='us-east-1')
    #session = BotoSession(profile_name='xacct', region_name='us-east-1').refreshable_session()
    
    sts_client = session.client('sts')

    response = sts_client.assume_role(
        RoleArn='arn:aws:iam::567516816488:role/central-bedrock-access',
        RoleSessionName='test'
    ) 
    
    #print(response)

    boto3_bedrock = session.client(
     service_name='bedrock', 
     #aws_access_key_id=response['Credentials']['AccessKeyId'],
     #aws_secret_access_key=response['Credentials']['SecretAccessKey'],
     #aws_session_token=response['Credentials']['SessionToken']
    )  
    
    print(boto3_bedrock.list_foundation_models())
    
    prompt = "what is best programming language?"
    
    inference_modifier_titan={
        "maxTokenCount":50,
        "stopSequences":[],
        "temperature":0.5,
        "topP":0.3
        }

    inference_modifier_claude = {
                        "prompt":prompt,
                        "max_tokens_to_sample":50, 
                      "temperature":0.5,
                      "top_k":250,
                      "top_p":1,
                      "stop_sequences": []
                     }

    print('TITAN')
    body = json.dumps({"inputText": prompt , 
                       "textGenerationConfig":inference_modifier_titan});
    
    
    response = boto3_bedrock.invoke_model(body=body, modelId=titanID, accept=accept, contentType=contentType)
    response_body = json.loads(response.get('body').read())
    
    outputText = response_body.get('results')[0].get('outputText')
    
    print(outputText)
    
    
    print('CLAUDE')
    body = json.dumps(inference_modifier_claude);
    response = boto3_bedrock.invoke_model(body=body, modelId=claudeID, accept=accept, contentType=contentType)
    response_body = json.loads(response.get('body').read())
    #print(response_body)
    outputText = response_body.get('completion')
    
    print(outputText)