"""
Author: Nick Lawson (github.com/ccrngd1)

FIle: bedrock_claude_structure.py
Description: Wrapper functions for calling Bedrock APIs using Claude
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

from persona.prompt_template.bedrock import * 
from persona.prompt_template.refreshableSession import *

class BedrockClaudeLLM: 
  
  modelId = 'anthropic.claude-v2' 
  accept = 'application/json'
  contentType = 'application/json' 
  
  _session = ""
  _client=""
  
  default_params = {
    "prompt":"",
    "max_tokens_to_sample":1000, 
    "temperature":0.7,
    "top_k":250,
    "top_p":0.8,
    "stop_sequences": ['\n']
 }
    
  def __init__(self):
    pass
    
  # ============================================================================
  # #####################[SECTION 1: private functions ######################
  # ============================================================================
  
  def _boto3_bedrock(self, renew=False):
    
    if False:
      session = BotoSession().refreshable_session()
  
      return session.client(
       service_name='bedrock',
       region_name='us-west-2'
      )
    elif False:
      if renew or not self._session:
        print('bedrock_claude_structure renewing session')
        self._session = boto3.Session(profile_name='xacct', region_name='us-east-1')
        
        self._client = self._session.client(
         service_name='bedrock-runtime'
        ) 
    else:
      if renew or not self._session:
        print('bedrock_claude_structure renewing session')
        self._session = boto3.Session()
        
        self._client = self._session.client(
         service_name='bedrock-runtime',
         region_name='us-east-1'
        ) 
        
    return self._client
    
  def _boto3_bedrock_embedding(self): 
    return BedrockEmbeddings(client=self._boto3_bedrock)
  
  def __func_validate(self, gpt_response): 
    if len(gpt_response.strip()) <= 1:
      return False
    if len(gpt_response.strip().split(" ")) <= 1: 
      return False
    return True
    
  def __func_clean_up(self,gpt_response):
    cleaned_response = gpt_response.strip()
    return cleaned_response
    
  def temp_sleep(self, seconds=0.1):
    time.sleep(seconds)
    pass
    
  
  def _llm_request(self,prompt, llm_parameter=0, verbose=True): 
    """
    Given a prompt and a dictionary of GPT parameters, make a request to OpenAI
    server and returns the response. 
    ARGS:
      prompt: a str prompt
      gpt_parameter: a python dictionary with the keys indicating the names of  
                     the parameter and the values indicating the parameter 
                     values.   
    RETURNS: 
      a str of GPT-3's response. 
    """
    end_on_newline = False
    
    prompt = "Human:\r"+prompt+"\rAssistant:"
    
    if llm_parameter==0:
      llm_parameter = self.default_params
    else:
      #check to see if params passed in are for gpt
      #if so - adapt to Claude
      if 'engine' in llm_parameter:
        titan_params = {}
        titan_params['max_tokens_to_sample'] = llm_parameter['max_tokens']
        titan_params['stop_sequences'] = llm_parameter['stop']
        titan_params['temperature'] = llm_parameter['temperature']
        titan_params['top_p'] = llm_parameter['top_p']
        titan_params['top_k']=250
        llm_parameter = titan_params
        
    if llm_parameter['stop_sequences'] is None:
      llm_parameter['stop_sequences']=[]
    
    if '\n' in llm_parameter['stop_sequences']:
      end_on_newline=True
    
    self.temp_sleep()
    body = ""
    response = ""
    
    retVal =""
    retry_attempts=2
    renew_session =False
    
    while retry_attempts>0:
      retry_attempts-=1
      try: 
        body = json.dumps({
              "prompt": prompt, 
              "max_tokens_to_sample":llm_parameter["max_tokens_to_sample"],
              #"stop_sequences":llm_parameter["stop_sequences"],
              "temperature":llm_parameter["temperature"],
              "top_p":llm_parameter["top_p"],
              #"top_k":llm_parameter["top_k"]
            }) 
        
        response = self._boto3_bedrock(renew_session).invoke_model(body=body, modelId=self.modelId, accept=self.accept, contentType=self.contentType)
        response_body = json.loads(response.get('body').read())
        
        retVal = response_body.get('completion').strip()
        
        if verbose:
            print('llm_request body')
            print(body)
            print('llm_request repsonse')
            print(retVal)
        
        if end_on_newline:
          v = retVal.split('\n')
          
          for a in v:
            if a:
              retVal = a
              break
        break
      except Exception as e: 
        print ("Bedrock error")
        print (body)
        print (response)
        print (e)
        renew_session = True
        self.temp_sleep(10)
    
    return retVal
      
  # ============================================================================
  # #####################[SECTION 2: Public functions] ######################
  # ============================================================================
  def LLM_single_request(self,prompt, params = default_params): 
    return self._llm_request(prompt,params)
    
  def LLM_safe_generate_response(self,
                                    prompt, 
                                     example_output="",
                                     special_instruction="",
                                     repeat=5,
                                     fail_safe_response="error",
                                     func_validate=__func_validate,
                                     func_clean_up=__func_clean_up,
                                     verbose=True,
                                     params=default_params): 
    prompt = 'Human:' + prompt + '\n'
    #prompt += f"Output the response to the prompt above in json. {special_instruction}\n"
    if special_instruction:
      prompt += f"{special_instruction}\n"
    
    if example_output:
      prompt += "Example output :\n"
      prompt += str(example_output) 
      
    prompt += '\nAssistant:'
  
    i = 0
    while i < repeat: 
      i+=1
      curr_gpt_response = ""
      
      try: 
        curr_gpt_response = self._llm_request(prompt, params).strip()
        
        #stop asking for resposne to be in json - we can do that programattically and Titan doesn't return json when asked
        #end_index = curr_gpt_response.rfind('}') + 1
        #curr_gpt_response = curr_gpt_response[:end_index] 
        
        if not curr_gpt_response:
          curr_gpt_response = '{"output":""}'
          if repeat<10:
            repeat+=1
        else:
          curr_gpt_response = '{"output":"'+ curr_gpt_response.strip() +'"}'
          
        print (curr_gpt_response)
        
        curr_gpt_response = json.loads(curr_gpt_response)["output"]
  
        if verbose:
          print ("---ashdfaf")
          print (curr_gpt_response)
          print ("000asdfhia")
        
        if func_validate(curr_gpt_response, prompt=prompt): 
          return func_clean_up(curr_gpt_response, prompt=prompt)
        
        if verbose: 
          print ("---- repeat count: \n", i, curr_gpt_response)
          print (curr_gpt_response)
          print ("~~~~")
          
        if repeat<10 and not curr_gpt_response:
          repeat+=1
  
      except Exception as e: 
        print ('Bedrock exception LLM_safe_generate_response prompt')
        print(prompt)
        print ('Bedrock exception LLM_safe_generate_response')
        print(curr_gpt_response)
        
        print(e)
        traceback.print_exc()
        
        pass
  
    raise Exception("Titan not responding")
    
    return False
  
  def LLM_safe_generate_response_OLD(self,prompt, 
                                     repeat=3,
                                     fail_safe_response="error",
                                     func_validate=None,
                                     func_clean_up=None,
                                     verbose=True): 
    if verbose: 
      print ("LLM PROMPT")
      print (prompt)
  
    for i in range(repeat): 
      try: 
        curr_gpt_response = self._llm_request(prompt).strip()
        if func_validate(curr_gpt_response, prompt=prompt): 
          return func_clean_up(curr_gpt_response, prompt=prompt)
        if verbose: 
          print (f"---- repeat count: {i}")
          print (curr_gpt_response)
          print ("~~~~")
  
      except: 
        pass
    print ("FAIL SAFE TRIGGERED") 
    return fail_safe_response
    
    
  def generate_prompt(self,curr_input, prompt_lib_file, verbose=True): 
    """
    Takes in the current input (e.g. comment that you want to classifiy) and 
    the path to a prompt file. The prompt file contains the raw str prompt that
    will be used, which contains the following substr: !<INPUT>! -- this 
    function replaces this substr with the actual curr_input to produce the 
    final promopt that will be sent to the GPT3 server. 
    ARGS:
      curr_input: the input we want to feed in (IF THERE ARE MORE THAN ONE
                  INPUT, THIS CAN BE A LIST.)
      prompt_lib_file: the path to the promopt file. 
    RETURNS: 
      a str prompt that will be sent to  server.  
    """
    
    if type(curr_input) == type("string"): 
      curr_input = [curr_input]
    curr_input = [str(i) for i in curr_input]
  
    f = open(prompt_lib_file, "r")
    prompt = f.read()
    f.close()
    for count, i in enumerate(curr_input):   
      prompt = prompt.replace(f"!<INPUT {count}>!", i)
    if "<commentblockmarker>###</commentblockmarker>" in prompt: 
      prompt = prompt.split("<commentblockmarker>###</commentblockmarker>")[1]
    return prompt.strip()
    
    
  def safe_generate_response(self,prompt, 
                             gpt_parameter,
                             repeat=5,
                             fail_safe_response="error",
                             func_validate=__func_validate,
                             func_clean_up=__func_clean_up,
                             verbose=True): 
    if verbose: 
      print ('bedrock safe_generate_response prompt')
      print (prompt)
  
    for i in range(repeat): 
      curr_gpt_response = self._llm_request(prompt, gpt_parameter)
        
      if verbose: 
        print ('bedrock safe_generate_response response')
        print ("---- repeat count: ", i, curr_gpt_response)
        print (curr_gpt_response)
        print ("~~~~")
      
      if func_validate(curr_gpt_response, prompt=prompt): 
        return func_clean_up(curr_gpt_response, prompt=prompt)
        
    return fail_safe_response
    
  def get_embedding(self,text):
    inp = json.dumps({"inputText": text})
    response = self._boto3_bedrock().invoke_model(body=inp, modelId="amazon.titan-e1t-medium", accept=self.accept, contentType=self.contentType)
    response_body = json.loads(response.get('body').read())
    embedding = response_body.get('embedding')
    return embedding