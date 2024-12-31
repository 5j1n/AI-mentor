const chatInput = document.querySelector("#chat-input");
const sendButton = document.querySelector("#send-btn");
const chatContainer = document.querySelector(".chat-container");
const themeButton = document.querySelector("#theme-btn");
const deleteButton = document.querySelector("#delete-btn");
const eventSource = new EventSource('/events');

let userText = null;

const loadDataFromLocalstorage = () => {
    const themeColor = localStorage.getItem("themeColor");
    document.body.classList.toggle("light-mode", themeColor === "light_mode");
    themeButton.innerText = document.body.classList.contains("light-mode") ? "dark_mode" : "light_mode";

    const defaultText = `<div class="default-text">
                            <h1>전북대 AI 멘토</h1>
                            <p>궁금한 것을 물어보세요.</p>
                        </div>`

    chatContainer.innerHTML = localStorage.getItem("all-chats") || defaultText;
    chatContainer.scrollTo(0, chatContainer.scrollHeight);
}

const createChatElement = (content, className) => {
    const chatDiv = document.createElement("div");
    chatDiv.classList.add("chat", className);
    chatDiv.innerHTML = content;
    return chatDiv;
}

const loadHistoryFromLocalstorage = () => {
    return JSON.parse(localStorage.getItem('history')) || [];
};

// 로컬 스토리지에 질문과 응답을 저장하는 함수
const saveQA = (question, response) => {
    let history = loadHistoryFromLocalstorage();
    history.push({ question, response });
    console.log(history)
    // 히스토리가 5개 이상이면 첫 번째 항목을 제거
    if (history.length > 5) {
        history.shift();
    }

    localStorage.setItem('history', JSON.stringify(history));
};

const clearHistory = () => {
    // 로컬 스토리지에서 history를 삭제합니다. 서버 재시작 시 이를 호출하여 초기화합니다.
    localStorage.removeItem('history');
};

function extractCourseNames(text) {
    // '학기:' 또는 '학기'가 있는 곳에서부터 추출
    const startIndex = text.indexOf("4학년 2학기");
    const relevantText = text.slice(startIndex);
  
    // 정규 표현식을 사용하여 강의명만 추출
    const coursePattern = /([가-힣0-9\s]+) \(/g;
    const courseNames = [];
    let match;
  
    while ((match = coursePattern.exec(relevantText)) !== null) {
      courseNames.push(match[1].trim());
    }
  
    return courseNames;
};

const getChatResponse = (incomingChatDiv, question) => {
    const pElement = document.createElement("p");
    curriculum_start = false;
    pElement.textContent = "Connecting...";
    incomingChatDiv.querySelector(".chat-details").appendChild(pElement);

    const history = loadHistoryFromLocalstorage();
    const historyParam = encodeURIComponent(JSON.stringify(history));
    console.log("History loaded:", history);
    const url = `http://210.117.181.104:8080/get-curriculum?question=${encodeURIComponent(question)}&history=${historyParam}`;

    const eventSource = new EventSource(url);
    let historyText = "";
    let curriculumText = ""
    eventSource.onmessage = (event) => {
        // 받은 데이터를 콘솔에 출력
        console.log("Received data:", event.data);
        if (event.data === "END") {
            // 타이핑 애니메이션 끝내기
            incomingChatDiv.querySelector('.typing-animation').style.display = 'none';
            saveQA(question, historyText); // 응답을 히스토리에 저장
            if (curriculum_start === true){
                let regex = /'교과목명':\s*'([^']+)',\s*'학과':\s*'([^']+)',\s*'수업목표':\s*'([^']+)',\s*'학년':\s*(\d+),\s*'학기':\s*'([^']+)'/g;
                let matches;
                let data = [];
                while ((matches = regex.exec(curriculumText)) !== null) {
                    let course = {
                        교과목명: matches[1],
                        학과: matches[2],
                        수업목표: matches[3],
                        학년: matches[4],
                        학기: matches[5]
                    };
                    data.push(course);
                }
                
                console.log(data);
                populateCourseButtons(data);
                curriculum_start = false
            }
            console.log(question, historyText)
            return;
        }
        
        let responseText = event.data.trim();
        if (responseText === "") {
            responseText = " "; // 데이터가 비어있다면 띄어쓰기로 간주
        }
        if (responseText === "curriculum_start") {
            curriculum_start = true;
            responseText = responseText.replaceAll("curriculum_start","");
        }
        if (responseText.length > 400 && curriculum_start === true) {
            let clean_curri = responseText.replaceAll("[","");
            clean_curri = clean_curri.replaceAll("]","");
            clean_curri = clean_curri.replaceAll("data: <br><br>","")
            curriculumText += clean_curri;
            responseText = "";
            console.log(curriculumText)
        }

        let cleanText = responseText.replace(/<br>/g, "\n"); // <br> 태그를 제거
    
        cleanText = cleanText.replaceAll("*","");
        cleanText = cleanText.replaceAll("#","");
        cleanText = cleanText.replaceAll("-","");
        cleanText = cleanText.replaceAll("No result","");
        historyText += cleanText
        // 타이핑 효과
        let index = 0;
        
        // setInterval을 사용하여 한 글자씩 출력
        const typingInterval = setInterval(() => {
            if (index < cleanText.length) {
                pElement.textContent += cleanText.charAt(index);  // 한 글자씩 추가
                index++;
            } else {
                clearInterval(typingInterval);  // 모든 글자가 출력되면 타이핑 효과 종료
                incomingChatDiv.scrollTo(0, incomingChatDiv.scrollHeight);  // 스크롤을 맨 아래로 이동
            }
        }, 50);  // 타이핑 속도 (30ms 간격)
    };

    eventSource.onerror = (error) => {
        console.log("Error: ", error);
        eventSource.close();
    };

    eventSource.onopen = () => {
        pElement.textContent = "";  // 초기 텍스트 제거
    };
};

function convertToJSON(data) {
    return data.map(item => {
        // 각 객체의 키와 값을 이중 따옴표로 감싸서 변환
        let jsonObject = {};
        for (let key in item) {
            if (item.hasOwnProperty(key)) {
                // key와 value를 이중 따옴표로 감싸서 변환
                jsonObject[`"${key}"`] = `"${item[key]}"`;
            }
        }
        return jsonObject;
    });
}

function populateCourseButtons(data) {
    const curriculumInfo = document.getElementById("curriculum-prompts-btn");
    const table = document.getElementById("curriculum-table");
    if (table) {
        curriculumInfo.style.display = "block";
        table.style.display = "table"; // 테이블 표시
    }

    // 버튼들을 찾을 수 있는 선택자
    const buttons = [
        "btn1", "btn2", "btn3", "btn4", "btn5",
        "btn6", "btn7", "btn8", "btn9", "btn10",
        "btn11", "btn12", "btn13", "btn14", "btn15",
        "btn16", "btn17", "btn18", "btn19", "btn20",
        "btn21", "btn22", "btn23", "btn24", "btn25",
        "btn26", "btn27", "btn28", "btn29", "btn30"
    ];

    // 각 학년-학기별 버튼 범위 정의
    const buttonRanges = {
        "4-2학기": [0, 5],  // 4학년 2학기: 버튼 1~5
        "4-1학기": [5, 10], // 4학년 1학기: 버튼 6~10
        "3-2학기": [10, 15], // 3학년 2학기: 버튼 11~15
        "3-1학기": [15, 20], // 3학년 1학기: 버튼 16~20
        "2-2학기": [20, 25], // 2학년 2학기: 버튼 21~25
        "2-1학기": [25, 30] // 2학년 1학기: 버튼 26~30
    };

    // 학년-학기별 버튼 설정
    Object.keys(buttonRanges).forEach(key => {
        const [grade, semester] = key.split("-");
        const [start, end] = buttonRanges[key];
        console.log(parseInt(grade), semester)
        // 해당 학년, 학기에 해당하는 과목 필터링
        const filteredCourses = data.filter(course => course["학년"] === grade && course["학기"] === semester);
        console.log(filteredCourses)
        // 버튼에 과목명 설정
        for (let i = 0; i < filteredCourses.length && start + i < end; i++) {
            const course = filteredCourses[i];
            const button = document.getElementById(buttons[start + i]);

            if (button && course["교과목명"] && course["학과"]) {
                button.innerHTML = `${course["교과목명"]}<br>${course["학과"]}`; // 교과목명 + 학과명
                            // 버튼 클릭 이벤트 추가
                button.addEventListener("click", function() {
                    displayCourseInfo(course);
                });
            }
        }
    });
}

function displayCourseInfo(course) {
    // 교과목명, 수업목표, 학과 정보를 표시할 HTML 요소
    const courseNameElement = document.getElementById("course-name");
    const courseDepartmentElement = document.getElementById("course-department");
    const courseGoalElement = document.getElementById("course-goal");
    const displayInfo = document.getElementById("explain-prompts-btn");
    const courseInfo = document.getElementById("course-info");
    if (courseInfo) {
        courseInfo.style.display = "block";
        displayInfo.style.display = "block";
    }
    // 교과목 정보를 HTML 요소에 설정
    courseNameElement.textContent = course["교과목명"];
    courseDepartmentElement.textContent = "학과: " + course["학과"];
    let course_goal = course["수업목표"]
    course_goal = (course_goal || "").replace(/\n/g, "<br>");
    const courseGoalText = "수업목표: " + course_goal
    console.log(courseGoalText)
    courseGoalElement.innerHTML = courseGoalText;
}

const copyResponse = (copyBtn) => {
    const reponseTextElement = copyBtn.parentElement.querySelector("p");
    navigator.clipboard.writeText(reponseTextElement.textContent);
    copyBtn.textContent = "done";
    setTimeout(() => copyBtn.textContent = "content_copy", 1000);
}

const showTypingAnimation = (userText) => {
    const html = `<div class="chat-content">
                    <div class="chat-details">
                        <img src="static/images/chatbot.jpg" alt="chatbot-img">
                        <div class="typing-animation">
                            <div class="typing-dot" style="--delay: 0.2s"></div>
                            <div class="typing-dot" style="--delay: 0.3s"></div>
                            <div class="typing-dot" style="--delay: 0.4s"></div>
                        </div>
                    </div>
                    <span onclick="copyResponse(this)" class="material-symbols-rounded">content_copy</span>
                </div>`;
    const incomingChatDiv = createChatElement(html, "incoming");
    chatContainer.appendChild(incomingChatDiv);
    chatContainer.scrollTo(0, chatContainer.scrollHeight);

    // 이제 userText를 getChatResponse에 전달하여 실제 응답을 처리
    getChatResponse(incomingChatDiv, userText);  // userText를 전달하여 서버에 질문을 보냄
};

const handleOutgoingChat = () => {
    const userText = chatInput.value.trim();  // 사용자가 입력한 질문을 받아옵니다.
    if (!userText) return;

    chatInput.value = "";
    chatInput.style.height = `${initialInputHeight}px`;

    // 사용자 메시지 출력
    const html = `<div class="chat-content">
                    <div class="chat-details">
                        <img src="static/images/user.jpg" alt="user-img">
                        <p>${userText}</p>
                    </div>
                </div>`;

    const outgoingChatDiv = createChatElement(html, "outgoing");
    chatContainer.querySelector(".default-text")?.remove();
    chatContainer.appendChild(outgoingChatDiv);
    chatContainer.scrollTo(0, chatContainer.scrollHeight);

    setTimeout(() => showTypingAnimation(userText), 500);  // showTypingAnimation에 userText를 넘겨줍니다.
};

deleteButton.addEventListener("click", () => {
    if(confirm("Are you sure you want to delete all the chats?")) {
        localStorage.removeItem("all-chats");
        loadDataFromLocalstorage();
    }
});
themeButton.addEventListener("click", () => {
    document.body.classList.toggle("light-mode");
    localStorage.setItem("themeColor", themeButton.innerText);
    themeButton.innerText = document.body.classList.contains("light-mode") ? "dark_mode" : "light_mode";
});

const initialInputHeight = chatInput.scrollHeight;

chatInput.addEventListener("input", () => {   
    chatInput.style.height =  `${initialInputHeight}px`;
    chatInput.style.height = `${chatInput.scrollHeight}px`;
});

chatInput.addEventListener("keydown", (e) => {
    if (e.key === "Enter" && !e.shiftKey && window.innerWidth > 800) {
        e.preventDefault();
        handleOutgoingChat();
    }
});

eventSource.addEventListener('courses', function(event) {
    const data = JSON.parse(event.data);  // JSON 문자열을 배열로 복원
    console.log('Received courses:', data);  // 예: ["item1", "item2", "item3"]
  });
  
loadDataFromLocalstorage();
sendButton.addEventListener("click", handleOutgoingChat);


clearHistory();