
import asyncio
import numpy as np
import openai
import time
import AllAgent
import random
import ast
from langchain.memory import ConversationBufferMemory

np.random.seed(42)

class CurriculumAgent(AllAgent.AllAgent):
    def __init__(self):
        super().__init__()
        self.memory = ConversationBufferMemory(return_messages=True)
        
    async def curriculumagent(self, question, continuous, history):
        start_time = time.time()
        key_skills = ""  # key_skills를 빈 문자열로 초기화
        department_names = ""         #추천 학과 명
        response_curriculum = ""      #커리큘럼 추천 강의 명
        num_departments = 0
        yield "curriculum_start"
        # identify_key_skill이 async generator를 반환하는 경우, async for 사용
        async for part in self.identify_key_skill(question):
            key_skills += part  # 각 부분을 key_skills에 추가
            for char in part:
                yield char  # 받은 부분을 즉시 한 글자씩 yield
        prompt =  "\n\n해당 분야와 파악된 핵심 기술과 관련된 학과입니다.\n"
        for char in prompt:
            yield char  # 받은 부분을 즉시 한 글자씩 yield
        async for part in self.recommend_relevant_department(question, key_skills):
            if isinstance(part, (int, float, np.floating)):
                print(part)
                if num_departments <= 3:
                    department_names += ", "
                    yield ", "
                    
            else:
                department_names += part
                num_departments += 1
                print(len(department_names))
                for char in part:
                    yield char  # 받은 부분을 즉시 한 글자씩 yield
        prompt = "\n\n위에서 언급된 핵심 기술과 관련된 학과를 기반으로 해당 학기에 대한 추천 커리큘럼입니다.\n\n"
        for char in prompt:
            yield char
        # department_names = "회계학과, 농경제유통학부, 공공인재학부, 도시공학과, 통계학과, 환경공학과"
        # print_department_str = f"해당 분야와 파악된 핵심 기술과 관련된 학과는 {department_str}입니다"
        result = self.search_db(question + " 관련학과: " + department_names, continuous, history)
        async for part in self.generate_curriculum(question, department_names, key_skills, result, history):
            if isinstance(part, list):
                # 리스트 그대로 yield
                yield part
                yield "\n\n"
                await asyncio.sleep(0.01)
            else:
                response_curriculum += part  # 각 부분을 key_skills에 추가
                for char in part:
                    yield char  # 받은 부분을 즉시 한 글자씩 yield
        prompt = "이와 같이 추천된 강의들은 해당 분야에서 요구되는 핵심 기술과 지식을 체계적으로 습득할 수 있도록 구성되었습니다."
        for char in prompt:
            yield char
        end_time = time.time()
        print(f"총 커리큘럼 생성 시간: {(end_time - start_time) * 1000:.2f} ms")
        
    async def dep_curriculumagent(self, question, continuous, history):
        start_time = time.time()
        relevant_department = self.recommend_relevant_department_all(question, None)
        response_curriculum = ""
        result = self.search_db(question + " 관련학과: " + ", ".join(dept[0] for dept in relevant_department), continuous, history)
        async for part in self.generate_curriculum(question, relevant_department, None, result, history):
            response_curriculum += part  # 각 부분을 key_skills에 추가
            for char in part:
                yield char  # 받은 부분을 즉시 한 글자씩 yield
                
        end_time = time.time()
        print(f"총 커리큘럼 생성 시간: {(end_time - start_time) * 1000:.2f} ms")
        
    async def direct_curriculumagent(self, question, continuous, history):
        start_time = time.time()
        filtered_data = self.search_db(question, continuous, history)
        if not filtered_data or filtered_data == "None" or filtered_data is None:
            yield "No result"
        else:
            response_curriculum = ""
            async for part in self.generate_curriculum(question, None, None, filtered_data, history):
                response_curriculum += part  # 각 부분을 key_skills에 추가
                for char in part:
                    yield char  # 받은 부분을 즉시 한 글자씩 yield
        end_time = time.time()
        print(f"총 커리큘럼 생성 시간: {(end_time - start_time) * 1000:.2f} ms")
        
    async def identify_key_skill(self, question):
        start_time = time.time()
        prompt = f""" 
            "When a user expresses interest in a specific career, follow these steps:
            Career Definition: Define the career specified by the user in a concise and clear manner. The definition should include the primary role, responsibilities, and objectives in the field. 
            List and Output Required Technical Skills:
            Technical Skills: List the key technical skills required to succeed in this career. Include relevant technologies, tools, processes, and software.
            Output Format:
            진로 정의:
            [Career Name] is a professional who [Career Definition: Primary role, responsibilities, and key technologies used].
            필요한 기술적 정의:
            [Technology/Tool 1]
            [Technology/Tool 2]
            [Technology/Tool 3]
            [Technology/Tool 4]
            [Technology/Tool 5]
            Use this structure to provide relevant information based on the user's query."
            Example Outputs:
            User Input: "I want to become an AI Machine Failure Analysis Expert."
            Career Definition:
            An AI Machine Failure Analysis Expert specializes in utilizing artificial intelligence technologies to identify, diagnose, and propose solutions for mechanical failures. This role focuses on applying AI-based tools and algorithms to enhance the accuracy and efficiency of failure diagnostics and implement preventive measures.
            Technical Skills: The following technical skills are required:
            Proficiency in machine learning frameworks (e.g., TensorFlow, PyTorch)
            Data analysis and visualization tools (e.g., Python, MATLAB, Tableau)
            Knowledge of predictive maintenance systems and techniques
            Expertise in IoT-based monitoring systems
            Experience with mechanical simulation and modeling software (e.g., ANSYS, SolidWorks)
            user query :{question}
            Please print the output in Korean
        """
        response = openai.ChatCompletion.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a helpful and kind assistant."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=1500,
            n=1,
            stop=None,
            temperature=0.5,
            seed = 1234,
            stream=True  # 스트리밍 활성화
        )
        
        response_content = ""
        # 스트리밍된 응답을 비동기적으로 처리
        for chunk in response:
            content_chunk = chunk['choices'][0].get('delta', {}).get('content', '')
            if content_chunk:
                response_content += content_chunk
                yield content_chunk  # 한 글자씩 반환

        end_time = time.time()
        print(f"\n핵심 기술 파악 시간: {(end_time - start_time) * 1000:.2f} ms")
          
    async def generate_curriculum(self, question, department_list, key_skill, contexts_list, history):
        start_time = time.time()
        recommended_courses_set_name = set()     # 이전학기 까지 추천된 학과명 집합
        added_prerequisites_set = set()          # 선수과목에 이미 추가가되어있는지 확인을 위한 집합
        recommended_courses_set_name_department = set()
        self.conversation_question.append({question})
        categorized_courses = self.classify_courses(contexts_list)
        recommended_prerequisites = []
        target_year = None
        for year in range(2, 5):  # 2학년부터 4학년까지 확인
            for semester in ["1학기", "2학기"]:
                flag = False
                year_semester_key = f"{year}학년 {semester}"
                if categorized_courses.get(year_semester_key):  # 해당 학기 데이터가 존재할 경우
                    target_year = year
                    flag = True
                    break  # 첫 번째로 발견된 학년으로 설정하고 반복 종료
            if flag is True:
                break
            if target_year is None:
                target_year = 2

        # 입력된 학년부터 4학년까지의 모든 학기 강의를 각각 처리
        for year in range(5, target_year-1, -1):
            for semester in ["2학기", "1학기"]:
                matching_courses = []  # 선수과목 중 현재 학기에 존재하는 과목
                semester_start_time = time.time()
                response = ""           # 응답 
                year_semester_key = f"{year}학년 {semester}"
                relevant_courses = categorized_courses.get(year_semester_key, [])
                if relevant_courses:
                    for prerequisite_info in recommended_prerequisites:  # 리스트로 변경하여 수정 가능하도록
                        # 권장선수과목과 학과명을 key-value 쌍으로 분리
                        prerequisite = list(prerequisite_info.keys())[0]  # 권장선수과목
                        department = prerequisite_info[prerequisite]  # 학과

                        # relevant_courses에서 강의명과 학과명이 일치하는 항목을 찾음
                        for course_info in relevant_courses:
                            if course_info["교과목명"] == prerequisite and course_info["학과"] == department:
                                # 일치하는 경우 강의명과 학과명을 리스트에 추가
                                matching_courses.append((prerequisite, department))

                                # 일치하는 항목을 recommended_prerequisites에서 삭제
                                recommended_prerequisites.remove(prerequisite_info)
                                break  # 하나를 찾으면 더 이상 찾을 필요 없음
                    # print("권장선수과목 목록 : ", recommended_prerequisites)
                    print("이번 학기에 존재하는 권장선수과목 : ", matching_courses)
                    response_content = ""
                    count = 0
                    relevant_courses_without_prerequisites = [
                        {key: course_info[key] for key in course_info if key != "권장선수과목"}
                        for course_info in relevant_courses
                    ]
                    random.shuffle(relevant_courses_without_prerequisites)
                    context_str = "\n\n".join([f"Context {i+1}: {str(ctx)}" for i, ctx in enumerate(relevant_courses_without_prerequisites)])
                    print("이번 학기에 추천 할 수 있는 과목 : ", context_str)
                    for i, _ in enumerate(relevant_courses):
                        count+=1
                    if key_skill is None:
                        prompt = f"""
                            Please list all lectures in the given context and ensure that no duplicate lectures are included. 
                            You MUST answer only using the following contexts:\n\n{context_str} in Korean.
                            Format for query and answer:\n
                            {year_semester_key}에 대한 모든 강의입니다.\n\n
                            - 교과목1 
                            - 교과목2 
                            - 교과목3 
                            - 교과목4 
                            - 교과목5
                            - 교과목6
                            ...\n
                        """                        
                    else:
                        prompt = f"""
                            You are an expert educational planner tasked with creating a curriculum for students in reverse semester order, starting from 4th grade.
                            **IMPORTANT:** You must NOT list {context_str} under any circumstances. Instead, you should strictly recommend lectures based on the specific instructions given below:
                            
                            ### Requirements:
                            1. **Curriculum in Reverse Order**:
                            Recommended Course list from Previous Semester: {', '.join(recommended_courses_set_name)}.
                            You should check this list and only select courses that are **NOT** already listed.
                            
                            2. **Curriculum based on '수업목표'**:
                            Your course recommendations must be based exclusively on the ‘수업목표’ found in 'context_str'. Do **NOT** include or list 'context_str' itself. 
                            Identify and recommend courses that help build the skills and knowledge relevant to {key_skill} and support the objectives related to {question}.
                            
                            3. **Number of Lectures**:
                            Select **exactly FIVE courses per semester**. If there are fewer than five available, recommend all of the courses in 'context_str', but ensure they are not duplicated across semesters.
                            
                            4. **No Duplication of Course Names**:
                            Do NOT repeat any course name across semesters. Even if a course appears under different professors or departments, it must be treated as a duplicate and excluded from further recommendations.
                            
                            5. **Prerequisite Lecture {matching_courses}**:
                            "The list of {matching_courses} represents the courses that are recommended to be considered when selecting recommended courses, as they have been identified as prerequisites for the previously recommended courses. However, their inclusion is not mandatory, and if there are two or more courses in {matching_courses}, the last course is not recommended for inclusion."
                            
                            6. **Content Limitations**:
                            You must **ONLY** use the contents from 'context_str' to make your recommendations. Under no circumstances should you introduce or list any course, department, or data that is **NOT** already in 'context_str'.
                            
                            7. **Output Language**:
                            All responses must be provided in Korean.
                            
                            8. **No Additional Explanations**:
                            You must not provide any additional explanations or recommendations. Only the courses and departments listed in 'context_str' should be included. Do not include any extra commentary, analysis, or background information.
                                                    
                            9. **Output Format**:
                            Please format your recommendations as follows for each semester:
                            {year_semester_key}에 대한 추천 강의입니다.\n\n
                            - 교과목1 (학과)
                            - 교과목2 (학과)
                            - 교과목3 (학과)
                            - 교과목4 (학과)
                            - 교과목5 (학과)\n
                        """
                    ############################기존 프롬프트#########################
                    # prompt = f"""
                    #     You are currently recommending the curriculum in reverse, starting from the 4th year.
                    #     Review the ‘수업목표’ in the list of lessons provided and select lessons that related with {question}.
                    #     You MUST answer only using the following contexts:\n\n{context_str} in Korean.
                    #     We will distribute the selected lectures, with 5 lectures per semester. For each semester,
                    #     You must make sure to list each '교과목명' in other department only once and avoid including the same course multiple times.
                    #     Format for query and answer:\n
                    #     {year_semester_key}에 대한 추천 강의입니다\n\n
                    #     - 교과목1 (학과): Summarize '수업목표' of this lecture in korean and explain the reason why you select this lecture\n
                    #     - 교과목2 (학과): Summarize '수업목표' of this lecture in korean and explain the reason why you select this lecture\n
                    #     - 교과목3 (학과): Summarize '수업목표' of this lecture in korean and explain the reason why you select this lecture\n
                    #     - 교과목4 (학과): Summarize '수업목표' of this lecture in korean and explain the reason why you select this lecture\n
                    #     - 교과목5 (학과): Summarize '수업목표' of this lecture in korean and explain the reason why you select this lecture\n\n
                    # """

                    # GPT 모델에 요청
                    response = openai.ChatCompletion.create(
                        model="gpt-4o",
                        messages=[
                            {"role": "system", "content": "You are a helpful and kind assistant."},
                            {"role": "user", "content": prompt}
                        ],
                        max_tokens=1500,
                        n=1,
                        stop=None,
                        temperature=0
                    )

                    # 응답이 제대로 되었는지 확인
                    if "choices" in response and len(response["choices"]) > 0:
                        # 응답을 문자열로 처리
                        new_courses = response['choices'][0]['message']['content'].strip().split("\n")
                    else:
                        print("Invalid response structure:", response)
                        return  # 응답이 올바르지 않으면 함수 종료

                    recommended_courses_set = set()  # 강의명과 학과명이 함께 저장될 set

                    # 추천된 과목 처리 부분
                    for course in new_courses:
                        # 강의명만 추출 (괄호 앞 부분)
                        course_name = course.split(" (")[0].strip()
                        print(course_name)
                        department_name = course.split(" (")[1].split(")")[0].strip() if "(" in course else ""  # 학과명 추출
                        # 중복되지 않는 강의만 추가
                        if (course_name, department_name) not in recommended_courses_set:
                            if "-" in course_name:
                                course_name = course_name.lstrip('-').strip()  # 강의명에서 '-' 제거
                            print(course_name)
                            if course_name not in recommended_courses_set_name:
                                recommended_courses_set.add((course_name, department_name))  # 강의명과 학과명 추가
                                recommended_courses_set_name.add(course_name)
                                recommended_courses_set_name_department.add((course_name, department_name))
                                response_content += course  # 전체 형식을 유지하며 추가
                            if not response_content.endswith("\n"):
                                response_content += "\n"
                    print(recommended_courses_set_name)
                    response_content += "\n"
                    yield response_content
                    if key_skill:
                        result = self.extract_course_details(relevant_courses, recommended_courses_set)
                        yield result
                        # 추천된 과목의 권장 선수과목을 가져오기

                        print(recommended_courses_set_name_department)
                        for course, department in recommended_courses_set:
                            # relevant_courses에서 강의명과 학과명을 함께 찾음
                            for course_info in relevant_courses:
                                if course_info["교과목명"] == course and course_info["학과"] == department:
                                    prerequisites = course_info["권장선수과목"]
                                    department_info = course_info["학과"]
                                    # 권장선수과목이 있다면 쉼표로 나누어 리스트로 분리
                                    if prerequisites:
                                        prerequisites_list = prerequisites.split(',')
                                        # 각 권장선수과목과 학과명을 결합하여 recommended_prerequisites에 저장
                                        for prerequisite in prerequisites_list:
                                            # 'prerequisite.strip()'을 사용하여 불필요한 공백 제거
                                            prerequisite_cleaned = prerequisite.strip()
                                            if prerequisite_cleaned:
                                                # (선수과목, 학과명) 조합이 이미 추가된 것인지 확인
                                                if (prerequisite_cleaned, department_info) not in added_prerequisites_set:
                                                    recommended_prerequisites.append({prerequisite_cleaned: department_info})
                                                    added_prerequisites_set.add((prerequisite_cleaned, department_info))  # 추가된 조합을 기록
                    semester_end_time = time.time()
                    print(f"{year_semester_key} 커리큘럼 구성 시간: {(semester_end_time - semester_start_time) * 1000:.2f} ms")
                    # 결과 출력
        
        end_time = time.time()
        print(f"커리큘럼 구성 시간: {(end_time - start_time) * 1000:.2f} ms")

    
    def classify_courses(self, contexts):
        """
        강의 정보를 학년과 학기에 따라 분류.
        """
        categorized_courses = {
            "1학년 1학기": [],
            "1학년 2학기": [],
            "2학년 1학기": [],
            "2학년 2학기": [],
            "3학년 1학기": [],
            "3학년 2학기": [],
            "4학년 1학기": [],
            "4학년 2학기": []
        }
        for context in contexts:
            course_details = self.parse_course_info(context)
            year_semester = f"{course_details.get('학년')}학년 {course_details.get('학기')}"
            if year_semester in categorized_courses:
                categorized_courses[year_semester].append(context)
        return categorized_courses
    
    def parse_course_info(self, course_info):
        """
        학년과 학기 정보를 추출하여 반환.
        """
        course_details = {}
        if isinstance(course_info, dict):
            course_details['학년'] = course_info.get('학년', None)
            course_details['학기'] = course_info.get('학기', None)
        return course_details
    
    def extract_course_details(self, relevant_courses, recommended_courses_set):
        """
        Extract course name, department name, and class objectives for courses in relevant_courses
        that match course names and departments in recommended_courses_set.

        Args:
            relevant_courses (list of dict): List of courses with details such as course name, department, and objectives.
            recommended_courses_set (set of tuple): Set containing tuples of (course_name, department_name).

        Returns:
            list of dict: List of matching courses with their course name, department name, and objectives.
        """
        matching_courses = []

        for course in relevant_courses:
            course_name = course.get("교과목명")
            department_name = course.get("학과")
            objectives = course.get("수업목표")
            objectives_no_newline = objectives.replace("\n", "").replace("\r", "")
            grade = course.get("학년")
            semester = course.get("학기")

            if (course_name, department_name) in recommended_courses_set:
                matching_courses.append({
                    "교과목명": course_name,
                    "학과": department_name,
                    "수업목표": objectives_no_newline,
                    "학년": grade,
                    "학기": semester
                })

        return matching_courses

    
                        # response = openai.ChatCompletion.create(
                    #     model="gpt-4o-mini",
                    #     messages=[
                    #         {"role": "system", "content": "You are a helpful and kind assistant."},
                    #         {"role": "user", "content": prompt}
                    #     ],
                    #     max_tokens=1500,
                    #     n=1,
                    #     stop=None,
                    #     temperature=0.5,
                    #     seed = 1234,
                    #     stream = True
                    # )
                    #         # 스트리밍된 응답을 비동기적으로 처리
                    # for chunk in response:
                    #     content_chunk = chunk['choices'][0].get('delta', {}).get('content', '')
                    #     if content_chunk:
                    #         print(content_chunk)
                    #         if content_chunk == "(":
                    #             response_lecture.append(response_content)
                    #         response_content += content_chunk
                    #         recommended_courses_set.add(content_chunk)
                    #         yield content_chunk  # 한 글자씩 반환
                    # print("--------------------------------------", response_lecture)
                    # for lecture in response_lecture:
                    #     print(lecture)
                    #     prerequisite = ""
                    #     for course in context_list:
                    #         if course["교과목명"] == lecture:
                    #             prerequisite += context_list[lecture]['권장선수과목']
                    #     print("권장선수과목------------------------------------------------------------------------", prerequisite)
                    
                        #                     3. Curriculum using {department_list}:
                        # `{department_list}` contains a list of departments ranked by their similarity to the given question and key_skill, sorted in descending order of similarity. Make sure that courses in departments with higher similarity in `{department_list}` are prioritized in the curriculum.
                        # For example, if `{department_list}` contains departments A, B, and C (with A being the most similar), the curriculum should 'prioritize' courses in `{context_str}` if they are included. Do not create a curriculum where courses are recommended only in departments with lower similarity (e.g. B, C) and not in departments with higher similarity (e.g. A). If there are no courses in departments with higher similarity, proceed to the next most similar department.