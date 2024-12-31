const chatInput = document.querySelector("#chat-input");
const sendButton = document.querySelector("#send-btn");
const chatContainer = document.querySelector(".chat-container");
const themeButton = document.querySelector("#theme-btn");
const deleteButton = document.querySelector("#delete-btn");

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

const getChatResponse = (incomingChatDiv, question) => {
    const pElement = document.createElement("p");
    pElement.textContent = "Connecting...";
    incomingChatDiv.querySelector(".chat-details").appendChild(pElement);

    const history = loadHistoryFromLocalstorage();
    const historyParam = encodeURIComponent(JSON.stringify(history));
    console.log("History loaded:", history);
    const url = `http://127.0.0.1:5000/get-curriculum?question=${encodeURIComponent(question)}&history=${historyParam}`;

    const eventSource = new EventSource(url);
    let historyText = "";
    eventSource.onmessage = (event) => {
        // 받은 데이터를 콘솔에 출력
        console.log("Received data:", event.data);

        if (event.data === "END") {
            // 타이핑 애니메이션 끝내기
            incomingChatDiv.querySelector('.typing-animation').style.display = 'none';
            saveQA(question, historyText); // 응답을 히스토리에 저장
            console.log(question, historyText)
            return;
        }
        
        let responseText = event.data.trim();
        if (responseText === "") {
            responseText = " "; // 데이터가 비어있다면 띄어쓰기로 간주
        }

        historyText += responseText;
        console.log(responseText)
        let cleanText = responseText.replace(/<br>/g, "\n"); // <br> 태그를 제거
        console.log(cleanText)
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

const copyResponse = (copyBtn) => {
    const reponseTextElement = copyBtn.parentElement.querySelector("p");
    navigator.clipboard.writeText(reponseTextElement.textContent);
    copyBtn.textContent = "done";
    setTimeout(() => copyBtn.textContent = "content_copy", 1000);
}

const showTypingAnimation = (userText) => {
    const html = `<div class="chat-content">
                    <div class="chat-details">
                        <img src="images/chatbot.jpg" alt="chatbot-img">
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
                        <img src="images/user.jpg" alt="user-img">
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

loadDataFromLocalstorage();
sendButton.addEventListener("click", handleOutgoingChat);


clearHistory();