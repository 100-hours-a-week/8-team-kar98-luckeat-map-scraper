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
    'db': os.getenv('DB_NAME'),
    'charset': 'utf8mb4',
    'port': int(os.getenv('DB_PORT'))
}

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
            sha256_hash = hashlib.sha256(concat_str.encode('utf-8')).hexdigest()

            # store_url 필드 업데이트
            update_query = "UPDATE store SET store_url = %s WHERE id = %s"
            cursor.execute(update_query, (sha256_hash, store_id))

    # 변경사항을 데이터베이스에 커밋
    connection.commit()
finally:
    connection.close()
