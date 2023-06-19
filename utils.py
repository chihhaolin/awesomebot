
from langchain.prompts.chat import AIMessagePromptTemplate, HumanMessagePromptTemplate
from enum import Enum


class MessageType(Enum):
    AI_MESSAGE = 1
    HUMAN_MESSAGE = 2



def text_to_message(text: str, message_type: MessageType) -> list:
    """ 回傳 message list: [HumanMessage] or [AIMessage]"""
    
    if message_type == MessageType.HUMAN_MESSAGE:
        prompt = HumanMessagePromptTemplate.from_template(text)
        message = prompt.format_messages()
        return message
    
    elif message_type == MessageType.AI_MESSAGE:
        prompt = AIMessagePromptTemplate.from_template(text)
        message = prompt.format_messages()   
        return message


