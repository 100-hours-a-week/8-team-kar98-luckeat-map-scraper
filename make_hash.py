import pymysql
import hashlib

from dotenv import load_dotenv
import os

load_dotenv()

# 데이터베이스 연결 정보
db_config = {
    'host': os.getenv('DB_HOST'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'db': "dev",
    'charset': 'utf8mb4',
    'port': int(os.getenv('DB_PORT'))
}

# SHA-256 해시 생성 함수
def generate_sha256_hash(input_str):
    try:
        # SHA-256 해시 생성
        sha256_hash = hashlib.sha256(input_str.encode('utf-8')).hexdigest()
        # 첫 8자리만 반환
        return sha256_hash[:8]
    except Exception as e:
        # 예외 발생 시 hashCode() 대체 로직과 유사하게 구현
        hash_code = hash(input_str) & 0xffffffff  # Java의 hashCode()와 유사한 결과를 위해 32비트로 제한
        hash_code_hex = format(hash_code, '08x')  # 8자리 16진수로 변환
        return hash_code_hex[:8]  # 8자리 반환

# 데이터베이스 연결
connection = pymysql.connect(**db_config)

try:
    with connection.cursor() as cursor:
        # store 테이블 데이터 조회
        cursor.execute("SELECT id, store_name, google_place_id FROM store")
        stores = cursor.fetchall()

        # 각 행에 대해 SHA-256을 계산하여 store_url 필드 업데이트
        for store in stores:
            store_id, store_name, google_place_id = store
            concat_str = f"{store_name}{google_place_id}"
            
            # 새로운 해시 함수 사용
            hash_value = generate_sha256_hash(concat_str)

            # store_url 필드 업데이트
            update_query = "UPDATE store SET store_url = %s WHERE id = %s"
            cursor.execute(update_query, (hash_value, store_id))

    # 변경사항을 데이터베이스에 커밋
    connection.commit()
finally:
    connection.close()
