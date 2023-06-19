import config
from langchain.memory import ConversationBufferMemory, ReadOnlySharedMemory
import json
from langchain.memory import ChatMessageHistory
from langchain.schema import messages_from_dict, messages_to_dict
import asyncio
import time

from bot_factory import BotAction, TaskBOT

from langchain.memory import ConversationBufferMemory
from langchain.memory import ChatMessageHistory

import utils
from utils import MessageType

class ChatRouteInJupyter:
    """
    Message:
        1. asyncio 在 jupyter 上的 使用方式不一樣， 這個 class 把 asyncio 的部分，全部移到 Jupyter上，以方便互動介面上的 開發與測試

        2. ChatRoute 繼承 ChatRouteInJupyter, 會把 asyncio 包進 class for prodution 使用
        
    Flow 功能:
        1. 定義好，各種不同種類的 TaskBots
        2. User Input:  str: next_text
        3. 把 next_text 結合 前 Ｎ 輪的 memory (message)，用 asyncio 送入 TaskBots to LLM
        4. 把 step3 返回的結果，送回 TaskBots 去 parse 出 「合適的format」
        5. 把 step4 的結果，送進去 「出話策略模塊」決定 response_text
        6. 把 (HUMAN)next_text 和 (AI) response_text，存入 memory 裡

    步驟:
        1. 先去 _create_taskbots() 定義 需要的 taskbots

    
    先來測試上下文聊天機器人。
        提取任務: 
            1. 提取使用者的興趣
            2. 追問使用者興趣
        聊天: 
            1. 上下文聊天
        出話策略: 
            1. 上下文聊天當基礎，適時追問使用者的興趣，如果提取到使用者興趣後，就不再進行追問。

    """

    def __init__(self, memory_length = 3, banned_N = 3):
        self.bots, self.bot_relation = self._create_taskbots()        
        self.next_message_pair = []  
        self.memory_length =  memory_length     
        
        self.task_collection = self._create_task_collection() # 收集任務資訊 { key:[] }
        self.whatQ_banned = banned_N # N輪內，不可以追問同樣主題的提問
        self.whatQ_banned_queue = [None for _ in range(banned_N)]
        
        self.memory = ConversationBufferMemory(return_messages=True, memory_key=config.MEMORY_KEY)  
        #self.history = ChatMessageHistory()
        # 如果要存DB 可以考慮這隻 api: message_history = MongoDBChatMessageHistory...    

    def _create_taskbots(self) -> list:
        """
            回傳這個任務需要的 bots, 假設這個任務，是收集使用者興趣
        """
        
        
        chitchat = TaskBOT(BotAction.CHIT_CHAT, temperature = 1.0, 
            system_message="你是一個有情感、有溫度的繁體中文聊天機器人，講話穩重且用詞非常口語化簡單好懂，回話不會超過70個字")

        
        interest_bot = TaskBOT(BotAction.INFO_EXTRACT, temperature = 0.0, 
            system_message="你是一隻任務型機器人，負責提取 使用者輸入的對話文字關於使用者的「興趣」，的內容。",
            Extract_key="興趣", Extract_description="把使用者對話內容，跟「興趣」有關的提取出來")

        what_q_intBot = TaskBOT(BotAction.WHAT_Q, temperature = 1.0, 
            system_message="你是一隻提出 what 問句 的繁體中文機器人，專門問別人的「興趣」是什麼，個性穩重且口語化",
            WHAT_Q="你平常的興趣是什麼")

        bots = [chitchat, interest_bot,  what_q_intBot]
        relation = {
            what_q_intBot: interest_bot
        }
        return [bots, relation]


    def _create_task_collection(self):
        r = {}
        for bot in self.bots:
            if bot.bot_type == BotAction.INFO_EXTRACT:
                r[bot.extract_key] = []
        return r

    def my_awesome_response_strategy(self, res: list) -> str:
        """
            res 是由 get_step_taskchains 的 chain 經 LLM api 產出的結果。
            直接回傳，接著，後處理
        """        
        candidate_results = self._postprosess_jupyter_ai_responses(res)
        
        chitchat_candidates = [] #[ [str, bot_id]]  
        what_q_candidates = [] #[ [str, bot_id]]

        ## 處理 bot_type = INFO_EXTRACT 和 CHIT_CHAT
        for i, t in enumerate(candidate_results):
            if self.bots[i].bot_type == BotAction.INFO_EXTRACT:
                key = self.bots[i].extract_key
                value = candidate_results[i].get(key, None)
                if value:                     
                    self.task_collection[key].append(value)
                
            if self.bots[i].bot_type == BotAction.CHIT_CHAT:
                _res = [candidate_results[i], self.bots[i]]
                chitchat_candidates.append(_res)            

        ## 處理 追問 what_q (for INFO_EXTRACT )
        for i, t in enumerate(candidate_results):
            if self.bots[i].bot_type == BotAction.WHAT_Q:
                info_extract_bot = self.bot_relation[self.bots[i]]
                tmp_extract_values = self.task_collection[info_extract_bot.extract_key]
                ## 沒抽到東西，就追問
                if not tmp_extract_values: 
                    ## 如果追問機器人沒有被ban, 就追問
                    if self.bots[i] not in self.whatQ_banned_queue:
                        _res = [candidate_results[i], self.bots[i]]
                        what_q_candidates.append(_res)
        
        ## 我的神奇策略2， 
        ## 選chitcat_candidates 最後一個，選what_q_candidates最後一個
        ## concat在一起後，直接回傳
        response = ""
        if chitchat_candidates:
            _chat, _ = chitchat_candidates.pop()
            response = response + _chat
        
        ## 這邊可以 再打一次 LLM 進行二次篩選
        if what_q_candidates:
            _what_q, _id = what_q_candidates.pop()
            response = response + " " + _what_q
            self.whatQ_banned_queue.append(_id)
        else:
            self.whatQ_banned_queue.append(None)
        self.whatQ_banned_queue.pop(0)

         
        if response == "":
            response = "我是機器人，我不知道跟你說什麼比較好。"

        # ## 我的神奇策略1，選第一隻bot (chitchat bot)的結果，直接回傳
        # ai_response_string = canditate_results[0]


        ai_response_string = response
        ## update memory & return ai_response_string
        self._create_next_ai_message(ai_response_string)        
        message_pair = self.next_message_pair
        self.update_memory(message_pair)
        self.next_message_pair = []

        return ai_response_string



    def _postprosess_jupyter_ai_responses(self, res: list) -> list[str]:
        """
            res 是由 get_step_taskchains 的 chain 經 LLM api 產出的結果。
            直接回傳，接著，後處理
        """        
        thinking_stage_canditate_results = []
        
        for i, ai_text in enumerate(res):
            res = self.bots[i].postprocess_api_response(ai_text)
            thinking_stage_canditate_results.append(res)
        
        return thinking_stage_canditate_results

    def get_step_taskchains(self, next_text:str ) -> list:
        """
            return 準備好的 chains, 去 asyncio run 即可
        """

        if not isinstance(next_text, str):
            raise Exception("user input must be string")

        chains = []
        messages = self.get_messages(N=self.memory_length)

        for bot in self.bots:
            chains.append(bot.get_chains(messages, next_text))
        
        ## 把 input text 存起來
        self._create_next_human_message(next_text)

        return chains


    def _create_next_human_message(self, next_text: str) -> None:
        if len(self.next_message_pair) != 0:
            raise Exception("Internel Memory Error") 
        
        human_message = utils.text_to_message(next_text, message_type = MessageType.HUMAN_MESSAGE)
        self.next_message_pair.extend(human_message)

    def _create_next_ai_message(self, next_ai_text: str) -> None:
        if len(self.next_message_pair) != 1:
            raise Exception("Internel Memory Error") 
        
        ai_message = utils.text_to_message(next_ai_text, message_type = MessageType.AI_MESSAGE)
        self.next_message_pair.extend(ai_message)

    def get_messages(self, N=2) -> list:
        """
            回傳 messages: [HumanMessage, AIMessage, HumanMessage, AIMessage...] 
            
            前 N 輪對話， 1輪有兩個 messages，順序如下， [HumanMessage, AIMessage]

        """
        r = self.memory.load_memory_variables({})
        messages = r.get(config.MEMORY_KEY, None) #如果沒有 memory，api 會回傳 []
        if not messages:
            return []
        length = N*2*(-1)
        messages = messages[length:]
        return messages[:]

    def update_memory(self, messages: list) -> None:
        """
            input messages: [HumanMessage, AIMessage]

            ps: 可以使用 utils.text_to_message 的 funtion 把 字串轉換成 message
            
            一定要是一對 message pair，而且順序不能變， 先human 再 ai

        """
        user = messages[0].content
        ai = messages[1].content
        self.memory.chat_memory.add_user_message(user)
        self.memory.chat_memory.add_ai_message(ai)






class ChatRoute(ChatRouteInJupyter):
    def step(self, chains):
        pass


    def async_run_cahins(self, chains):
        # If running this outside of Jupyter, use asyncio.run(generate_concurrently())
        #res = await generate_concurrently(chains)
        res = asyncio.run(self.generate_concurrently(chains))
        return res

    async def async_generate(self, chain):
        resp = await chain.arun(dummy="dummy")
        #print(resp)
        return resp

    async def generate_concurrently(self, chains):
        tasks = [self.async_generate(chain) for chain in chains]
        res = await asyncio.gather(*tasks)
        return res


    def update_memory(self):
        pass


    def response_strategy(self):
        pass

    def set_slot_goal(self):
        pass


if __name__ ==  "__main__":
    chitchat = TaskBOT(BotAction.CHIT_CHAT, system_message="你是聊天機器人")
    texts = ["你好", "今天天氣怎樣", "夏天適合游泳嗎", "知名的籃球球星有誰", "知名的歌手有誰" ]
    chains = [chitchat.get_chitchat_chain([], next_text=t) for t in texts]
    cr = ChatRoute()
    res = cr.async_run_cahins(chains)
    print(res)