from langchain.memory import ConversationBufferMemory
from langchain_community.llms import OpenAI
import DatabaseConnection
import openai
import config
import AllAgent
import NormalAgent
import CurriculumAgent
import time

db_connection = DatabaseConnection.DatabaseConnection(
    host=config.host,
    port=config.port,
    user=config.user,
    password=config.password,
    database=config.database
)


class QuestionProcessor:
    def __init__(self, db_connection):
        self.db_connection = db_connection
        self.memory = ConversationBufferMemory(return_messages=True)
        self.conversation_question = []
        self.failed_directagent = False  # directagent 실패 여부를 저장하는 플래그
        openai.api_key = config.api_key
    async def process_questions(self, question, history):
        start_time = time.time()
        assignment, continuous = self.classify_question(question, history)
        print(f"배정 결과: {assignment}, {continuous}")
        if assignment == '1':
            async for partial_response in self.directagent(question, continuous, history):
                yield partial_response
            # result = await self.directagent(question, continuous, history)
            end_time = time.time()
            print(f"총 응답 시간: {(end_time - start_time) * 1000:.2f} ms")
            print("-" * 50)            
        elif assignment == '2':
            async for partial_response in self.searchagent(question, continuous, history):
                yield partial_response
            end_time = time.time()
            print(f"총 응답 시간: {(end_time - start_time) * 1000:.2f} ms")
            print("-" * 50)            
        elif assignment == '3':
            async for partial_response in self.normalagent(question, continuous, history):
                yield partial_response
            end_time = time.time()
            print(f"총 응답 시간: {(end_time - start_time) * 1000:.2f} ms")
            print("-" * 50)
        elif assignment == '4':
            # `curriculumagent`가 여러 개의 결과를 `yield`로 반환
            async for partial_response in self.curriculumagent(question, continuous, history = None):
                yield partial_response
            end_time = time.time()
            print(f"총 응답 시간: {(end_time - start_time) * 1000:.2f} ms")
            print("-" * 50)
        elif assignment == '5':
            # `curriculumagent`가 여러 개의 결과를 `yield`로 반환
            async for partial_response in self.dep_curriculumagent(question, continuous, history = None):
                yield partial_response
            end_time = time.time()
            print(f"총 응답 시간: {(end_time - start_time) * 1000:.2f} ms")
            print("-" * 50)
        elif assignment == '6':
            # `curriculumagent`가 여러 개의 결과를 `yield`로 반환
            async for partial_response in self.search_curriculumagent(question, continuous, history = None):
                yield partial_response
            end_time = time.time()
            print(f"총 응답 시간: {(end_time - start_time) * 1000:.2f} ms")
            print("-" * 50)

    
    def classify_question(self, question, history, failed=False):
        print("Failed: ", failed)
        start_time = time.time()
        # 프롬프트를 작성하여 LLM에 질문을 전달하고 응답을 받음
        if failed == 1:
            return "2",0
        elif failed == 2:
            return "6",0

        # LLM 프롬프트 작성: 이전 대화와 연관성을 평가
        prompt = f"""
        You are an intelligent agent that classifies questions into one of five categories:
        1. When the question requests a simple data search, or there is any conjunction or directive pronoun (such as '그럼', '그', '이런') appearing in the question. This includes questions asking about departments, professors, or specific courses. (e.g., "컴퓨터인공지능학부 2학년 때 무슨 과목을 들어야할까?", "골프는 언제 개설돼?", "송현제 교수님은 뭘 가르치셔?", "그럼 그 과목을 가르치는 교수님은 누구야?", "동역학은 어느 학과에서 배워?").
        3. Assign to the agent for general or casual questions that do not involve specific department or lecture (e.g., "저녁 추천해줘", "오늘 날씨 어때?", "날씨 좋다", "고마워", "잘헸어!", "졸려", "배고파", "심심해").
        4. Assign questions related to careers or educational curriculum to this agent (e.g., "AI 기계결함 전문가가 되기 위해 2학년부터 커리큘럼을 짜줘" or "과학선생님이 되기 위한 과목을 추천해줘" or "의료인공지능에 대해 공부하고 싶어").
        5. Assign to this agent for questions specifically asking about the curriculum of a particular department, the question must include "커리큘럼" (e.g., "컴퓨터공학부 커리큘럼 알려줘", "기계공학부 커리큘럼은 뭐야?", "경제부 커리큘럼은 어떻게 돼?").

        Classify the issue as category 1, 3, 4, or 5 only. Keep in mind that there is no 2.
        Also, determine if the current question is not relevant with the previous conversation or a new topic comes up, return '1' if it is continuous, otherwise return '0'.

        Question: "{question}"
        Previous conversation: "{history}"


        Return the answer in the following format:
        Category: [classification]
        Continuation: [0 or 1]
        """
        # LLM에게 질문 분류 요청
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=20,  
            n=1,
            stop=None,
            temperature=0
        )

        response_text = response['choices'][0]['message']['content'].strip()
        classification = None
        continuous = 0

        if "Category:" in response_text:
            classification = response_text.split("Category:")[1].split()[0].strip()

        if "Continuation:" in response_text:
            continuous = response_text.split("Continuation:")[1].split()[0].strip()
        end_time = time.time()
        print(f"질문: {question}")
        print(f"에이전트 배정 시간: {(end_time - start_time) * 1000:.2f} ms")
        return classification, int(continuous)

    async def directagent(self, question, continuous, history):
        agent = AllAgent.AllAgent()
        agent.memory = history  # 히스토리 공유
        agent.conversation_question = self.conversation_question
        result = ""
        async for partial_response in agent.directagent(question, continuous, history):
            result += partial_response
            yield partial_response 
        if result == "No result":  # directagent 결과가 없으면 classify_question을 통해 2번으로 유도
            print("DirectAgent의 결과가 없으므로 질문을 다시 분류합니다.")
            self.failed_directagent = True  # 실패 플래그 설정
            assignment, continuous = self.classify_question(question, history, failed=1)  # 분류에서 2번으로 유도
            if assignment == '2':
                print("2번으로 분류되어 searchagent 호출")
                async for partial_response in agent.searchagent(question, continuous, history):
                    yield partial_response 
            else:
                print(f"재분류 결과: {assignment}. 예상과 다르게 분류되었습니다.")
    
    async def searchagent(self, question, continuous, history):
        agent = AllAgent.AllAgent()
        agent.memory = history # 히스토리 공유
        agent.conversation_question = self.conversation_question
        async for partial_response in agent.searchagent(question, continuous, history):
            yield partial_response 

    async def normalagent(self, question, continuous, history):
        agent = NormalAgent.NormalAgent()
        agent.memory = history  # 히스토리 공유
        agent.conversation_question = self.conversation_question
        async for partial_response in agent.normalagent(question, continuous, history):
            yield partial_response 

    async def curriculumagent(self, question, continuous, history):
        agent = CurriculumAgent.CurriculumAgent()
        agent.memory = history  # 히스토리 공유
        agent.conversation_question = self.conversation_question
        # CurriculumAgent의 `curriculumagent` 메서드에서 `yield`되는 값을 받아서 하나씩 `yield`
        async for partial_response in agent.curriculumagent(question, continuous, history):
            yield partial_response 
            
    async def dep_curriculumagent(self, question, continuous, history):
        agent = CurriculumAgent.CurriculumAgent()
        agent.memory = history  # 히스토리 공유
        agent.conversation_question = self.conversation_question
        result = ""
        # CurriculumAgent의 `curriculumagent` 메서드에서 `yield`되는 값을 받아서 하나씩 `yield`
        async for partial_response in agent.direct_curriculumagent(question, continuous, history):
            result += partial_response
            yield partial_response 
        if result == "No result":  # directagent 결과가 없으면 classify_question을 통해 2번으로 유도
            print("dep_curriculumagent의 결과가 없으므로 질문을 다시 분류합니다.")
            self.failed_directagent = True  # 실패 플래그 설정
            assignment, continuous = self.classify_question(question, history, failed=2)  # 분류에서 2번으로 유도
            if assignment == '6':
                print("6번으로 분류되어 searchagent 호출")
                async for partial_response in agent.dep_curriculumagent(question, continuous, history):
                    result += partial_response
                    yield partial_response 
            else:
                print(f"재분류 결과: {assignment}. 예상과 다르게 분류되었습니다.")
    async def search_curriculumagent(self, question, continuous, history):
        agent = CurriculumAgent.CurriculumAgent()
        agent.memory = history  # 히스토리 공유
        agent.conversation_question = self.conversation_question
        # CurriculumAgent의 `curriculumagent` 메서드에서 `yield`되는 값을 받아서 하나씩 `yield`
        async for partial_response in agent.dep_curriculumagent(question, continuous, history):
            yield partial_response 