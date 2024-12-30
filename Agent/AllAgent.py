from langchain.memory import ConversationBufferMemory
from langchain_community.llms import OpenAI
import openai
import config
import time 

class Agent:
    def __init__(self):
        # LangChain의 ConversationBufferMemory를 사용하여 대화 히스토리를 관리
        self.memory = ConversationBufferMemory(return_messages=True)  # return_messages=True 옵션으로 메시지를 리스트로 반환
        self.conversation_question = []
        openai.api_key = config.api_key

    async def generate_response(self, question, context_list=None, continuous=0, history=None):
        start_time = time.time()
        self.conversation_question.append(question)
        messages = []
        if context_list is None:
            messages = self.normal_message(continuous, history)
        else:
            messages = self.using_context_list_message(context_list, continuous, history, question)

        messages.append({
            "role": "user",
            "content": f"Question: {question}\n\nAnswer:"
        })

        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=messages,
            n=1,
            max_tokens=1500,
            temperature=0,
            stop=None,
            stream=True
        )
        
        response_content = ""
        # 스트리밍된 응답을 비동기적으로 처리
        for chunk in response:
            content_chunk = chunk['choices'][0].get('delta', {}).get('content', '')
            if content_chunk:
                response_content += content_chunk
                for char in content_chunk:
                    yield char  # 한 글자씩 반환

        end_time = time.time()  # 응답 생성 종료 시간
        print(f"응답 생성 시간: {(end_time - start_time) * 1000:.2f} ms")  # 응답 생성 시간 계산


    def using_context_list_message(self, context_list, continuous, history, question):
        if continuous == 1:
            messages = [
                {
                    "role": "system",
                    "content": f"You are a helpful and kind assistant. "
                               f"Your answer must be made by only utilizing {question}, {context_list} and {history}. "
                               "Do not use your own knowledge."
                               "If there are multiple possible answers, provide all of them."
                },
            ]
        else:
            messages = [
                {
                    "role": "system",
                    "content": f"You are a helpful and kind assistant.\n"
                               f"Your answer must be made by only utilizing {question} and {context_list}. "
                               "Do not use your own knowledge."
                               "If there are multiple possible answers, provide all of them."
                },
            ]
        return messages

    def normal_message(self, continuous, history):
        if continuous == 1:
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                               f"If you can't make a clear response, utilize {history}."
                }
            ]
        else:
            messages = [
                {
                    "role": "system",
                    "content": "You are a helpful assistant."
                }
            ]
        return messages
