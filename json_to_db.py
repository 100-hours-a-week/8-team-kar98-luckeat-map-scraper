import os
import json
import mysql.connector
from dotenv import load_dotenv
from pathlib import Path
import argparse  # 인자 처리를 위한 추가

# .env 파일 로드
load_dotenv()

# 카테고리 매핑
CATEGORY_MAP = {
    "한식": 1,
    "중식": 2,
    "양식": 3,
    "일식": 4,
    "베이커리": 5,
    "포케": 6
}

# 데이터베이스 연결 설정
def get_db_connection(db_name):
    # 스크립트가 있는 디렉토리의 .env 파일을 명시적으로 로드
    script_dir = Path(__file__).resolve().parent
    dotenv_path = script_dir / '.env'
    load_dotenv(dotenv_path=dotenv_path)

    # 또는 특정 경로 지정
    # .env 파일에서 DB 정보 가져오기
    db_host = os.getenv('DB_HOST')
    db_user = os.getenv('DB_USER')
    db_password = os.getenv('DB_PASSWORD')
    db_port = os.getenv('DB_PORT')
    
    print(f"DB Host: {db_host}")
    print(f"DB User: {db_user}")
    print(f"DB Name: {db_name}")  # 인자로 받은 DB 이름 사용
    print(f"DB Port: {db_port}")
    
    # 연결 생성
    try:
        connection = mysql.connector.connect(
            host=db_host,
            user=db_user,
            password=db_password,
            database=db_name,
            port=db_port
        )
        print(f"데이터베이스 '{db_name}'에 성공적으로 연결되었습니다.")
        return connection
    except mysql.connector.Error as err:
        print(f"데이터베이스 연결 오류: {err}")
        raise

# JSON 파일에서 데이터 로드
def load_json_data(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

# 데이터를 DB에 저장
def save_to_database(connection, data):
    cursor = connection.cursor()
    
    # 저장 성공/실패 개수 추적을 위한 카운터
    success_count = 0
    error_count = 0
    
    # 테이블 존재 여부 확인
    try:
        cursor.execute("SHOW TABLES LIKE 'store'")
        if not cursor.fetchone():
            print("주의: 'store' 테이블이 존재하지 않습니다.")
            print("테이블 구조를 확인하세요.")
            return
    except mysql.connector.Error as err:
        print(f"테이블 확인 오류: {err}")
        return
    
    # SQL 삽입 쿼리
    insert_query = """
    INSERT INTO store (
        store_name, address, latitude, longitude, 
        contact_number, business_hours, website, 
        avg_rating_google, google_place_id, store_img, review_summary, category_id
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
        category_id = VALUES(category_id),
        updated_at = CURRENT_TIMESTAMP
    """
    # 데이터 처리 및 DB 저장
    for store in data:
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
        original_photo_url = store.get('photo_url', None)
        
        # 리뷰 요약
        review_summary = store.get('review_summary', None)
        
        # 카테고리 ID 매핑
        search_query = store.get('search_query', None)
        category_id = CATEGORY_MAP.get(search_query)
        
        # 데이터 검증 - 필수 필드 확인
        if not place_id:
            print(f"오류: place_id가 없는 레코드 발견 - {name}")
            error_count += 1
            continue
            
        if not category_id:
            print(f"주의: 카테고리를 찾을 수 없음 - {name}, 검색 쿼리: {search_query}")

        print(f"이름: {name}")

        # 데이터 삽입 (store_img에 original_photo_url 사용)
        store_data = (
            name,  # store_name
            address,  # address
            lat,  # latitude
            lng,  # longitude
            phone,  # contact_number
            opening_hours,  # business_hours
            website,  # website
            rating,  # avg_rating_google
            place_id,  # google_place_id
            original_photo_url,  # store_img (원본 URL)
            review_summary,  # review_summary
            category_id  # category_id
        )
        
        try:
            # DB 저장 시도
            print(f"DB 저장 시도: {name} (Place ID: {place_id})")
            cursor.execute(insert_query, store_data)
            success_count += 1
            print(f"가게 저장 완료: {name} (카테고리: {category_id})")
        except mysql.connector.Error as err:
            error_count += 1
            print(f"DB 저장 오류 발생 ({name}): {err}")
            print(f"SQL 쿼리: {cursor.statement}")  # 실행된 SQL 쿼리 출력
        except Exception as e:
            error_count += 1
            print(f"처리 중 예상치 못한 오류 발생 ({name}): {e}")
    
    # 변경사항 저장
    try:
        connection.commit()
        print(f"\n저장 결과 요약:")
        print(f"성공: {success_count}개")
        print(f"실패: {error_count}개")
    except mysql.connector.Error as err:
        print(f"커밋 오류: {err}")
        connection.rollback()
        print("변경사항이 롤백되었습니다.")
    
    cursor.close()

def main():
    # 커맨드 라인 인자 파싱
    parser = argparse.ArgumentParser(description='JSON 데이터를 MySQL 데이터베이스에 저장')
    parser.add_argument('--db', '-d', default='prod', help='사용할 데이터베이스 이름 (기본값: prod)')
    args = parser.parse_args()
    
    # 인자에서 DB 이름 가져오기
    db_name = args.db
    print(f"사용할 데이터베이스: {db_name}")
    
    connection = None
    try:
        connection = get_db_connection(db_name)

        # 사용자에게 계속 진행할지 묻기
        while True:
            user_input = input("\n데이터베이스에 연결되었습니다. 계속 진행하시겠습니까? (y/n): ").strip().lower()
            if user_input in ['y', 'yes']:
                print("프로그램을 계속 진행합니다...")
                break
            elif user_input in ['n', 'no']:
                print("사용자 요청으로 프로그램을 종료합니다.")
                return
            else:
                print("잘못된 입력입니다. 'y' 또는 'n'을 입력해주세요.")

        # JSON 파일 경로를 새로 생성된 파일로 변경
        json_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "구글맵_데이터", "판교_전체_가게정보_processed.json")
        print(f"JSON 파일 경로: {json_file_path}")

        if not os.path.exists(json_file_path):
            print(f"오류: JSON 파일을 찾을 수 없습니다 - {json_file_path}")
            return # 파일이 없으면 종료

        store_data = load_json_data(json_file_path)
        print(f"로드된 가게 데이터 수: {len(store_data)}")

        # 데이터베이스 테이블 정보 확인
        try:
            cursor = connection.cursor()
            cursor.execute("DESCRIBE store")
            columns = cursor.fetchall()
            print("\n테이블 구조 확인:")
            for column in columns:
                print(f"  - {column[0]}: {column[1]}")
            cursor.close()
        except mysql.connector.Error as err:
            print(f"테이블 구조 확인 오류: {err}")

        # 다시 한번 사용자에게 데이터 저장을 진행할지 확인
        while True:
            user_input = input(f"\n총 {len(store_data)}개의 가게 데이터를 데이터베이스에 저장하시겠습니까? (y/n): ").strip().lower()
            if user_input in ['y', 'yes']:
                print("데이터 저장을 시작합니다...")
                break
            elif user_input in ['n', 'no']:
                print("사용자 요청으로 데이터 저장을 취소합니다.")
                return
            else:
                print("잘못된 입력입니다. 'y' 또는 'n'을 입력해주세요.")

        save_to_database(connection, store_data)

        print("모든 가게 정보 처리가 완료되었습니다.")
    except mysql.connector.Error as db_err:
        print(f"데이터베이스 관련 오류 발생: {db_err}")
        if connection and connection.is_connected():
            print("데이터베이스 오류로 인해 롤백이 필요할 수 있습니다.")
    except Exception as e:
        print(f"오류 발생: {e}")
    finally:
        if connection and connection.is_connected():
            # 커밋 또는 롤백 후 연결 종료
            connection.close()
            print("데이터베이스 연결이 종료되었습니다.")

if __name__ == "__main__":
    main()
