import DatabaseConnection
import QuestionProcessor
import config
import time
import logging
import asyncio
import torch
import numpy as np
import random
from quart import Quart, request, jsonify, render_template, Response
from quart_cors import cors  # CORS를 Quart에서 지원하기 위해서 `quart_cors` 사용
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"


# 데이터베이스 연결 설정
db_connection = DatabaseConnection.DatabaseConnection(
    host=config.host,
    port=config.port,
    user=config.user,
    password=config.password,
    database=config.database
)

def seed_everything(seed):
    torch.manual_seed(seed) #torch를 거치는 모든 난수들의 생성순서를 고정한다
    torch.cuda.manual_seed(seed) #cuda를 사용하는 메소드들의 난수시드는 따로 고정해줘야한다 
    torch.cuda.manual_seed_all(seed)  # if use multi-GPU
    torch.backends.cudnn.deterministic = True #딥러닝에 특화된 CuDNN의 난수시드도 고정 
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed) #numpy를 사용할 경우 고정
    random.seed(seed) #파이썬 자체 모듈 random 모듈의 시드 고정
seed_everything(42)

# Quart 애플리케이션 설정
app = Quart(__name__)
logging.basicConfig(level=logging.INFO)
app = cors(app, allow_origin="*")
cors(app)  # CORS 설정

@app.route("/")
async def index():
    return await render_template("index.html")

@app.route("/get-curriculum", methods=["GET"])
async def get_curriculum():
    question = request.args.get('question')
    history_param = request.args.get('history')
    print("history: ", history_param)
    db_connection.connect()  
    try:
        question_processor = QuestionProcessor.QuestionProcessor(db_connection)
        # 비동기 스트리밍 생성기
        async def generate():
            # process_questions 메서드에서 비동기적으로 응답을 받아 스트리밍
            async for response_text in question_processor.process_questions(question, history_param):
                if isinstance(response_text, list):
                # 리스트 그대로 yield
                    for text in response_text:
                        for text2 in text:
                            formatted_response = text2.replace("\n", "")
                    yield f"data: {response_text}"
                else:
                    formatted_response = response_text.replace("\n", "<br>")
                    yield f"data: {formatted_response}\n\n"
                await asyncio.sleep(0.01)
            yield "data: END\n\n"
        
        # 서버 전송 스트리밍 응답 설정
        response = Response(generate(), mimetype='text/event-stream')
        
        db_connection.commit()  
        db_connection.handle_unread_result()

    except Exception as e:
        print(f"Error: {e}")
        response = Response(f"data: Error: {str(e)}\n\n", mimetype='text/event-stream')
    finally:
        db_connection.close()
    
    return response

if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)
