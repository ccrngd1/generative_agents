"""
Author: Joon Sung Park (joonspk@stanford.edu)

File: gpt_structure.py
Description: Wrapper functions for calling OpenAI APIs.
"""
import time 
import json
from pathlib import Path 
from litellm import completion
import boto3

from utils import * 

modelId = "bedrock/meta.llama3-70b-instruct-v1:0" 
#modelId = "bedrock/anthropic.claude-3-sonnet-20240229-v1:0" 
sess = None
client=""

default_params = {
  "prompt":"",
  "max_tokens_to_sample":1000, 
  "temperature":0.7,
  "top_k":250,
  "top_p":0.8,
  "stop_sequences": ['\n']
}
  
def __func_validate(gpt_response): 
  if len(gpt_response.strip()) <= 1:
    return False
  if len(gpt_response.strip().split(" ")) <= 1: 
    return False
  return True
  
def __func_clean_up(gpt_response):
  cleaned_response = gpt_response.strip()
  return cleaned_response
  
def temp_sleep(seconds=0.5):
  print('sleeping')
  time.sleep(seconds)
  print('resuming from sleep')
  pass

def _boto3_bedrock(renew=False):
  global sess, client, modelId
  if renew or not sess:
    print('bedrock_claude_structure renewing session')
    sess = boto3.Session()
  
  client = sess.client(
    service_name='bedrock-runtime',
    region_name='us-east-1'
  ) 
    
  return client 


def ChatGPT_single_request(prompt): 
  global sess, client, modelId
  temp_sleep()
  #resp = client.chat.completions.create(
  #  model=modelId,
  #  messages=[{"role": "user", "content": prompt}]
  #)
  resp = completion(
    model=modelId,
    messages=[{"role": "user", "content": prompt}],
      aws_bedrock_client=client,
  )
  return resp.choices[0].message.content


def ChatGPT_request(prompt, llm_parameter=0):   
  global sess, client, modelId 
    
  #prompt = "Human:\r"+prompt+"\rAssistant:"
  
  if llm_parameter==0:
    llm_parameter = default_params
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
  
  temp_sleep() 
  resp= ""
  try: 
    #resp = client.chat.completions.create(
    #  model=modelId,
    #  messages=[{"role": "user", "content": prompt}],
    #  temperature=llm_parameter["temperature"],
    #  max_tokens=llm_parameter["max_tokens_to_sample"],
    #  top_p=llm_parameter["top_p"],  
    #  stop=llm_parameter["stop_sequences"])
       
    resp = completion(
      model=modelId,
      messages=[{"role": "user", "content": prompt}],
      aws_bedrock_client=client,
      temperature=llm_parameter["temperature"],
      max_tokens=llm_parameter["max_tokens_to_sample"],
      top_p=llm_parameter["top_p"],  
      #stop=llm_parameter["stop_sequences"]
    ) 

    return resp.choices[0].message.content
  
  except Exception as e: 
    print(f"Error: {e}")
    print(f"REQUEST: {prompt}")
    print(f"RESPONSE: {resp}")
    return "ChatGPT ERROR"

'''
def ChatGPT_safe_generate_response(prompt, 
                                   example_output,
                                   special_instruction,
                                   repeat=5,
                                   fail_safe_response="error",
                                   func_validate=__func_validate,
                                   func_clean_up=__func_clean_up,
                                   verbose=False): 
  global sess, client, modelId 
  prompt = '"""\n' + prompt + '\n"""\n'
  prompt += f"Output the response to the prompt above in json. {special_instruction}\n"
  prompt += "So for example do not output 5 output {'output': 5}"
    
  if example_output:
    prompt += "Example output :\n"
    prompt += str(example_output) 

  if verbose: 
    print ("CHAT GPT PROMPT")
    print (prompt)

  for i in range(repeat): 

    try: 
      curr_gpt_response = ChatGPT_request(prompt).strip()
      end_index = curr_gpt_response.rfind('}') + 1
      curr_gpt_response = curr_gpt_response[:end_index]
      curr_gpt_response = json.loads(curr_gpt_response)["output"]
      
      if func_validate(curr_gpt_response, prompt=prompt): 
        return func_clean_up(curr_gpt_response, prompt=prompt)
      
      if verbose: 
        print ("---- repeat count: \n", i, curr_gpt_response)
        print (curr_gpt_response)
        print ("~~~~")

    except: 
      pass

  return False


def ChatGPT_safe_generate_response_OLD(prompt, 
                                   repeat=3,
                                   fail_safe_response="error",
                                   func_validate=None,
                                   func_clean_up=None,
                                   verbose=False): 
  global sess, client, modelId
  if verbose: 
    print ("CHAT GPT PROMPT")
    print (prompt)

  for i in range(repeat): 
    try: 
      curr_gpt_response = ChatGPT_request(prompt).strip()
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
'''

def GPT_request(prompt, gpt_parameter): 
  global sess, client, modelId  
  temp_sleep()
  response=""
  try:      
    response = ChatGPT_request(prompt,gpt_parameter)
      
    return response

  except Exception as e:
    print(f"Error: {e}")
    print(f"REQUEST: {prompt}")
    print(f"RESPONSE: {response}")
    return "GPT REQUEST FAILED" 

def generate_prompt(curr_input, prompt_lib_file): 
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
    a str prompt that will be sent to OpenAI's GPT server.  
  """
  global sess, client, modelId
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


# this function is frustratingly similar to CHATGPT_safe_generate_response
def safe_generate_response(prompt, 
                           gpt_parameter,
                           repeat=5,
                           fail_safe_response="error",
                           prompt_input=[],
                           prompt_template="",
                           func_validate=None,
                           func_clean_up=None,
                           verbose=False): 
  if debug:
    verbose = True

  for i in range(repeat): 
    if i >1:
      temp_sleep()

    if verbose: 
      print("------- BEGIN SAFE GENERATE --------")
      print ("---- repeat count: ", i)
      for j, prompt_j in enumerate(prompt_input):
        print("---- prompt_input_{}".format(j), prompt_j)
      print("---- prompt: ", prompt)
      print("---- prompt_template: ", prompt_template)
      print("---- gpt_parameter: ", gpt_parameter)

    curr_gpt_response = GPT_request(prompt, gpt_parameter)

    if verbose:  
      print("---- curr_gpt_response: ", curr_gpt_response)

    try: 
      response_cleanup = func_clean_up(curr_gpt_response, prompt=prompt)
    except Exception as e:
      if verbose:
        print("----  func_clean_up:  ERROR", e)
        print("---- func_validate: ", False)
        print(f"------- END TRIAL {i} --------")
      continue

    if curr_gpt_response == "GPT REQUEST FAILED":
      continue

    try: 
      response_valid = func_validate(curr_gpt_response, prompt=prompt)
    except Exception as e:
      if verbose:
        print("----  func_clean_up: ", response_cleanup)
        print("---- func_validate: ERROR", e)
        print(f"------- END TRIAL {i} --------")
      continue

    if verbose:
      print("----  func_clean_up: ", response_cleanup)
      print("---- func_validate: ", response_valid)
      print(f"------- END TRIAL {i} --------")
    if response_valid:
      print("------- END SAFE GENERATE --------")
      return response_cleanup
  
  # behaviour if all retries are used up
  if EXCEPT_ON_FAILSAFE:
    raise Exception("Too many retries and failsafes are disabled!")
  else:
    print("ERROR fail to succesfully retrieve response")
    print("ERROR using fail_safe: ", fail_safe_response)
    print("------- END SAFE GENERATE --------")
    return fail_safe_response


def get_embedding(text):
    global sess, client, modelId
    inp = json.dumps({"inputText": text})
    response = _boto3_bedrock().invoke_model(body=inp, modelId="amazon.titan-embed-text-v1")
    response_body = json.loads(response.get('body').read())
    embedding = response_body.get('embedding')
    return embedding




















