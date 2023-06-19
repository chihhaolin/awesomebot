from langchain.chat_models import ChatOpenAI
from langchain.schema import AIMessage, HumanMessage, SystemMessage
from langchain import PromptTemplate
from langchain.prompts.chat import PromptTemplate, ChatPromptTemplate, SystemMessagePromptTemplate, AIMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain import LLMChain
from langchain.memory import ConversationBufferMemory, ReadOnlySharedMemory
import json
from langchain.memory import ChatMessageHistory
from langchain.schema import messages_from_dict, messages_to_dict


import config
from enum import Enum

class BotAction(Enum):
    INFO_EXTRACT = 1
    WHAT_Q = 2
    CHIT_CHAT = 3
    INTENT_ACTION = 4 
    COPYWRITER = 5

class TaskBOT:
    """
        create stateless chat_model, 
        設計三種類型: (1) 信息提取 (2)針對特定主題，提出開放性問題[What-Q]  (3)閒聊[chitchat]
    """    
    def __init__(self, bot_type: BotAction, system_message: str, 
                Extract_key=None, Extract_description=None, WHAT_Q=None, temperature = 0.0):
        self.SystemMessage = SystemMessage(content=system_message)
        self.output_parser = None
        self.extract_key = None
        self.bot_type = bot_type
        self.chat_model = ChatOpenAI(openai_api_key = config.OPENAI_KEY,temperature = temperature, max_tokens=500, 
                                        model='gpt-3.5-turbo', max_retries = 3, request_timeout = 20)
        
        if self.bot_type == BotAction.INFO_EXTRACT:
            if (Extract_key is None) or (Extract_description is None):
                raise Exception(" INFO_EXTRACT setting error")
            self.set_ResponseFormat(Extract_key, Extract_description)
            self.extract_key = Extract_key
        
        if self.bot_type == BotAction.WHAT_Q:
            if WHAT_Q is None:
                raise Exception(" WHAT_Q setting needs sample question")
            self.what_q = WHAT_Q 

    def get_chains(self, messages : list, next_text:str):
        """ messages: [] 是過去的聊天記錄，由 AIMessage, HumanMessage 組成, messages[-1] = HumanMessage """

        if self.bot_type == BotAction.INFO_EXTRACT:
            chain = self._get_info_extract_chain(messages, next_text)
            return chain

        elif self.bot_type == BotAction.WHAT_Q:
            chain = self._get_what_q_chain()
            return chain
        elif self.bot_type == BotAction.CHIT_CHAT:
            chain = self._get_chitchat_chain(messages, next_text)
            return chain

    def postprocess_api_response(self, ai_text: str):
      
        if self.bot_type == BotAction.INFO_EXTRACT:
            ## output: dict, 裡面只有一組 (key, value) pair, 
            ## 其中 key = self.extract_key
            res = self._postprocess_info_extract(ai_text)
            return res

        elif self.bot_type == BotAction.WHAT_Q:
            ## output: string
            res = self._postprocess_what_q(ai_text)
            return res

        elif self.bot_type == BotAction.CHIT_CHAT:
            ## output: string
            res = self._postprocess_chitchat(ai_text)
            return res



    def set_SystemMessage(self, system_message: str) -> None:
        self.SystemMessage = SystemMessage(content=system_message)

    def set_ResponseFormat(self, name: str, description: str) -> None:
        """ 
        Example:
            name: 興趣
            description: 請提取聊天過程中，出現的「興趣」訊息。 如果沒提取到，請回傳 ***
        """

        response_schemas = [ 
            ResponseSchema(name = name, description = description)
        ]
        self.output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
        
    def _get_info_extract_chain(self, messages:list, next_text:str) -> LLMChain:
        """ messages: [] 是過去的聊天記錄，由 AIMessage, HumanMessage 組成, messages[-1] = AIMessage """
        """ next_text = user 最近傳入的字串 """

        bot_messages = [self.SystemMessage]        
        bot_messages.extend(messages)
        
        ## For 信提提取 taskBOT        
        ## 對 bot_messages 的最後一個 HumanMessage 最後面加入  \n{format_instructions}\n
        _human_massage = self.SystemMessage.content   + "\n{format_instructions}\n\n" + next_text 
        format_instructions = self.output_parser.get_format_instructions()
        human_message_prompt = HumanMessagePromptTemplate.from_template(_human_massage)
        message = human_message_prompt.format_messages(format_instructions = format_instructions)
        bot_messages.extend(message)
        prompt = ChatPromptTemplate.from_messages(messages=bot_messages)        
        return LLMChain(llm=self.chat_model, prompt=prompt)

        #res = chat_model(prompt.format_prompt().to_messages()) # json, with key  自己在 set_Response Format 定義的 "name"
        ## 這邊要實際跑來測試 ##
        #return {"type": "json", "res": res}

    def _get_what_q_chain(self) -> LLMChain:
        bot_messages = [self.SystemMessage] 
        text= """ 請參考下面句子的語意，寫一句一樣的語意但更 口語化且穩重 的問句，20個字以內。 
        
        句子:
        ```{what_q}```
        """
        human_message_prompt = HumanMessagePromptTemplate.from_template(text)
        message = human_message_prompt.format_messages(what_q = self.what_q)
        bot_messages.extend(message)
        prompt = ChatPromptTemplate.from_messages(messages=bot_messages)

        return LLMChain(llm=self.chat_model, prompt=prompt)
        #res = self.chat_model(message) # AIMessage
        #return {"type": "str_AIrespone", "res": res.content}
      
    def _get_chitchat_chain(self,  messages : list, next_text:str) -> LLMChain:
        """ messages: [] 是過去的聊天記錄，由 AIMessage, HumanMessage 組成, messages[-1] = AIMessage """
        """ next_text = user 最近傳入的字串 """
        bot_messages = [self.SystemMessage]        
        bot_messages.extend(messages)     

        human_message_prompt = HumanMessagePromptTemplate.from_template(next_text)
        message = human_message_prompt.format_messages()
        bot_messages.extend(message)
        prompt = ChatPromptTemplate.from_messages(messages=bot_messages)

        return LLMChain(llm=self.chat_model, prompt=prompt)
        # res = self.chat_model(bot_messages) # AIMessage
        # return {"type": "str_AIrespone", "res": res.content}        

    def _postprocess_info_extract(self, res: str) -> dict:
        """
            output: dict, 裡面只有一組 (key, value) pair, 
            其中 key = self.extract_key
        """        
        r = self.output_parser.parse(res) 
        return r

    def _postprocess_what_q(self, res) -> str:
        return res

    def _postprocess_chitchat(self, res: str) -> str:
        return res

class CopyWriter(TaskBOT):
    def get_copywriting_chain(self, chitchat: str, what_q:str) -> LLMChain:
        bot_messages = [self.SystemMessage]
        
        text= """ 幫我結合下面兩句話的語意後，寫一段70字以內 口語化且穩重 句子。 這個句子裡，最多只可以有一個問句。
        
        句子1:
        ```{chitchat}```

        句子2:
        ```{what_q}```        
        """        
        human_message_prompt = HumanMessagePromptTemplate.from_template(text)
        message = human_message_prompt.format_messages(chitchat = chitchat, what_q = what_q)
        bot_messages.extend(message)
        prompt = ChatPromptTemplate.from_messages(messages=bot_messages)
        
        return LLMChain(llm=self.chat_model, prompt=prompt)
        # res = self.chat(message) # AIMessage
        # return {"type": "str_AIrespone", "res": res.content}

    def postprocess_copywriter(self, res) -> dict:
        ## 這邊要實際跑來測試, 待處理 ##
        return {"type": "str_AIrespone", "res": res.content}