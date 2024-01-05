from enum import Enum
#from gpt_structure import *
from persona.prompt_template.bedrock_structure import *
from persona.prompt_template.bedrock_claude_structure import *

_llm= BedrockClaudeLLM()
 
class LLMType(Enum):
    OPENAI = 1
    BEDROCK = 2
    BEDROCKCLAUDE = 3

def set_llm(model):
    if model == LLMType.OPENAI:
        _llm= ChatGPTLLM()
    elif model == LLMType.BEDROCK:
        _llm = BedrockLLM()
    elif model == LLMType.BEDROCKCLAUDE:
        _llm = BedrockClaudeLLM()
        
    return _llm
    
def get_llm():
    return _llm