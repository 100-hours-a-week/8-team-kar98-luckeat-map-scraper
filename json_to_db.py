import os
import json
import mysql.connector
from dotenv import load_dotenv
from datetime import datetime

# .env 파일 로드
load_dotenv()

# 데이터베이스 연결 설정
def get_db_connection():
    # .env 파일에서 DB 정보 가져오기
    db_host = os.getenv('DB_HOST')
    db_user = os.getenv('DB_USER')
    db_password = os.getenv('DB_PASSWORD')
    db_name = os.getenv('DB_NAME')
    db_port = os.getenv('DB_PORT')
    
    # 연결 생성
    connection = mysql.connector.connect(
        host=db_host,
        user=db_user,
        password=db_password,
        database=db_name,
        port=db_port
    )
    
    return connection

# JSON 파일에서 데이터 로드
def load_json_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

# 데이터를 DB에 저장
def save_to_database(connection, data):
    cursor = connection.cursor()
    
    # SQL 삽입 쿼리
    insert_query = """
    INSERT INTO store (
        id, store_name, address, latitude, longitude, 
        contact_number, business_hours, website, 
        avg_rating_google, google_place_id, store_img, review_summary
    ) VALUES (
        %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s
    ) ON DUPLICATE KEY UPDATE
        store_name = VALUES(store_name),
        address = VALUES(address),
        latitude = VALUES(latitude),
        longitude = VALUES(longitude),
        contact_number = VALUES(contact_number),
        business_hours = VALUES(business_hours),
        website = VALUES(website),
        avg_rating_google = VALUES(avg_rating_google),
        google_place_id = VALUES(google_place_id),
        store_img = VALUES(store_img),
        review_summary = VALUES(review_summary),
        updated_at = CURRENT_TIMESTAMP
    """
    
    # 데이터 처리 및 DB 저장
    for idx, store in enumerate(data, start=1):
        # 필요한 데이터 추출
        place_id = store.get('place_id', '')
        name = store.get('name', '')
        address = store.get('address', '')
        
        # 좌표 정보 - 직접 latitude, longitude 필드에서 가져옴
        lat = store.get('latitude', 0)
        lng = store.get('longitude', 0)
        
        # 연락처 - phone 필드에서 가져옴
        phone = store.get('phone', None)
        
        # 영업시간 - opening_hours 필드에서 가져옴
        opening_hours = store.get('opening_hours', None)
        
        # 웹사이트
        website = store.get('website', None)
        
        # 평점
        rating = store.get('rating', None)
        
        # 이미지 URL
        photo_url = store.get('photo_url', None)
        
        # 리뷰 요약
        review_summary = store.get('review_summary', None)
        
        # 데이터 삽입
        store_data = (
            idx,  # id
            name,  # store_name
            address,  # address
            lat,  # latitude
            lng,  # longitude
            phone,  # contact_number
            opening_hours,  # business_hours
            website,  # website
            rating,  # avg_rating_google
            place_id,  # google_place_id
            photo_url,  # store_img
            review_summary  # review_summary
        )
        
        try:
            cursor.execute(insert_query, store_data)
            print(f"가게 저장 완료: {name}")
        except mysql.connector.Error as err:
            print(f"오류 발생: {err}")
    
    # 변경사항 저장
    connection.commit()
    cursor.close()

def main():
    try:
        # 데이터베이스 연결
        connection = get_db_connection()
        
        # JSON 파일 경로
        json_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "구글맵_데이터", "제주_전체_빵집_가게정보.json")
        
        # JSON 데이터 로드
        store_data = load_json_data(json_file_path)
        
        # 데이터베이스에 저장
        save_to_database(connection, store_data)
        
        print("모든 가게 정보가 성공적으로 데이터베이스에 저장되었습니다.")
        
    except Exception as e:
        print(f"오류 발생: {e}")
    
    finally:
        if 'connection' in locals() and connection.is_connected():
            connection.close()
            print("데이터베이스 연결이 종료되었습니다.")

if __name__ == "__main__":
    main()
