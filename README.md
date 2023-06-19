
### My Awesome Chatbot
利用 LangChain 快速搭建聊天機器人的核心部件  
https://python.langchain.com/docs/get_started/introduction.html


### LangChain 概念 與 Chatbot 架構圖
https://app.heptabase.com/w/05a6bdcfff20d205a8243a3ebdecd7674cc656efcc0caf50609e66f4d993e9b3


### 範例1  
https://github.com/chihhaolin/awesomebot/blob/master/ChatBot_Test2.ipynb  
flow.py: 因為在 jupyter 環境，使用 asyncio 封裝會出問題，因此把 asyncio 的部分留在外部  

**流程描述**  
1. 利用 bot_factory.py， 定義出三隻bot，chitchat 聊天, info_extraction信息提取, what_q追問問題。 其中 [info_extraction, what_q] 是一個 pair 組合
2. User Input:  str: next_text
3. 把 next_text 結合 前 Ｎ 輪的 memory (message)，用 asyncio 送入 TaskBots to LLM
4. 把 step3 返回的結果，送回 TaskBots 去 parse 出 「合適的format」
5. 把 step4 的結果，送進去 「出話策略模塊」決定 response_text
6. 把 (HUMAN)next_text 和 (AI) response_text，存入 memory 裡

**任務描述:上下文聊天機器人**  
1. 提取任務: (a) 提取使用者的興趣 (b)追問使用者興趣. 
2. 聊天:上下文聊天  
3. 出話策略: 上下文聊天當基礎，適時追問使用者的興趣，如果提取到使用者興趣後，就不再進行追問。  


### 範例2 信息提取  
https://github.com/chihhaolin/awesomebot/blob/master/InfoExtraction.ipynb  
1. 利用 bot_factory.py 定義好實體提取的機器人，每一隻機器人只提取一個實體  
2. 取得 LLMChain & 用 asyncio 打 LLM 模型，並取回結果  


### ToDo (Maybe)
**bot_factory.py:**  
1. CopyWriter: 第二次打LLM的機制，優化 response 的品質

**flow.py:**  
1. ChatRoute: 把 asyncio 封裝進來， for server 使用
