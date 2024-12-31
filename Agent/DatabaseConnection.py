import mysql.connector
import config

connection = mysql.connector.connect(
    host=config.host,       # 예: 'localhost'
    user=config.user,   # MySQL 사용자 이름
    password=config.password,  # MySQL 비밀번호
    database=config.database,   # 사용할 데이터베이스 이름
    port = config.port
)

class DatabaseConnection:
    def __init__(self, host, port, user, password, database):
        self.host = host
        self.port = port
        self.user = user
        self.password = password
        self.database = database
        self.connection = None
        self.cursor = None
        
    def is_connected(self):
        if self.connection:
            try:
                self.connection.ping(reconnect=True)  # 연결 상태 체크
                return True
            except mysql.connector.Error:
                return False
        return False
    
    def handle_unread_result(self):
        if self.cursor:
            while self.cursor.nextset():  # 이전 쿼리 결과를 처리
                pass
    def connect(self):
        self.connection = mysql.connector.connect(
            host=self.host,
            port=self.port,
            user=self.user,
            password=self.password,
            database=self.database
        )
        self.cursor = self.connection.cursor()

    def close(self):
        if self.cursor:
            self.cursor.close()
        if self.connection:
            self.connection.close()

    def get_cursor(self):
        return self.cursor
    def commit(self):
        # 트랜잭션 커밋
        if self.connection:
            self.connection.commit()
    def fetch_all(self):
        # 결과를 가져오기
        if self.cursor:
            return self.cursor.fetchall()

    def execute(self, query, params=None):
        self.cursor.execute(query, params)
        self.handle_unread_result()  # 미사용된 결과를 처리
