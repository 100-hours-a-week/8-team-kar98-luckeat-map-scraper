import pymysql
import hashlib

from dotenv import load_dotenv
import os

# tqdm 임포트
try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

load_dotenv()

# 데이터베이스 연결 정보
db_config = {
    'host': os.getenv('DB_HOST'),
    'user': os.getenv('DB_USER'),
    'password': os.getenv('DB_PASSWORD'),
    'db': "prod",
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
        # 모든 store_url을 미리 조회해서 set에 저장
        cursor.execute("SELECT store_url FROM store WHERE store_url IS NOT NULL AND store_url != ''")
        existing_store_urls = set(row[0] for row in cursor.fetchall())

        # store 테이블에서 id, store_name, google_place_id, store_url 조회
        cursor.execute("SELECT id, store_name, google_place_id, store_url FROM store")
        stores = cursor.fetchall()

        # tqdm 상태바 사용 (없으면 enumerate로 대체)
        iterator = tqdm(stores, desc="진행상황", unit="건") if tqdm else enumerate(stores)

        for store in iterator:
            if tqdm:
                store_id, store_name, google_place_id, store_url = store
            else:
                _, (store_id, store_name, google_place_id, store_url) = store
            # 이미 store_url이 있으면 건너뜀
            if store_url and str(store_url).strip() != '':
                continue

            concat_str = f"{store_name}{google_place_id}"

            # store_url 충돌 방지: 최대 10회까지 시도
            for i in range(10):
                candidate = generate_sha256_hash(concat_str if i == 0 else concat_str + str(i))
                if candidate not in existing_store_urls:
                    break
            else:
                print(f"[경고] {store_name}({store_id}) store_url 해시 충돌로 생성 실패")
                continue

            # store_url 필드 업데이트
            update_query = "UPDATE store SET store_url = %s WHERE id = %s"
            cursor.execute(update_query, (candidate, store_id))
            existing_store_urls.add(candidate)

    # 변경사항을 데이터베이스에 커밋
    connection.commit()
finally:
    connection.close()
