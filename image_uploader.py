import os
import json
import requests
import boto3
from io import BytesIO
from urllib.parse import urlparse
from dotenv import load_dotenv
from pathlib import Path
import time  # time 모듈 추가

# .env 파일 로드 (스크립트 위치 기준)
script_dir = Path(__file__).resolve().parent
dotenv_path = script_dir / '.env'
load_dotenv(dotenv_path=dotenv_path)

# AWS 설정 로드
aws_access_key_id = os.getenv('AWS_ACCESS_KEY_ID')
aws_secret_access_key = os.getenv('AWS_SECRET_ACCESS_KEY')
aws_region = os.getenv('AWS_REGION')
s3_bucket_name = os.getenv('S3_BUCKET_NAME')
cloudfront_domain = os.getenv('CLOUDFRONT_DOMAIN')

# S3 클라이언트 초기화
s3_client = boto3.client(
    's3',
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    region_name=aws_region
)

# JSON 파일에서 데이터 로드
def load_json_data(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
    except FileNotFoundError:
        print(f"Error: JSON file not found at {file_path}")
        return None
    except json.JSONDecodeError:
        print(f"Error: Could not decode JSON from {file_path}")
        return None

# 이미지를 S3에 업로드하고 CloudFront URL 반환
def upload_image_to_s3(image_url, place_id):
    if not image_url or not s3_bucket_name or not cloudfront_domain:
        print(f"필수 값 부족: image_url, s3_bucket_name, 또는 cloudfront_domain이 비어있습니다.")
        return None

    try:
        response = requests.get(image_url, stream=True, timeout=20) # 타임아웃 증가
        response.raise_for_status()

        content_type = response.headers.get('content-type')
        extension = '.jpg'
        if content_type and 'image/' in content_type:
            mime_type = content_type.split(';')[0]
            if '/' in mime_type:
                 ext_candidate = mime_type.split('/')[-1]
                 if ext_candidate.lower() in ['jpeg', 'jpg', 'png', 'gif', 'webp']:
                     extension = '.' + ext_candidate.lower()

        # S3 키 형식 수정: images/stores/place_id.확장자
        s3_key = f"images/stores/{place_id}{extension}"

        image_data = BytesIO(response.content)

        s3_client.upload_fileobj(
            image_data,
            s3_bucket_name,
            s3_key,
            ExtraArgs={'ContentType': content_type or 'image/jpeg'}
        )

        domain_part = cloudfront_domain
        if domain_part.startswith('https://'):
            domain_part = domain_part[len('https://'):]
        elif domain_part.startswith('http://'):
            domain_part = domain_part[len('http://'):]
            
        cloudfront_url = f"https://{domain_part.strip('/')}/{s3_key}"
            
        return cloudfront_url

    except requests.exceptions.Timeout:
        print(f"이미지 다운로드 시간 초과 ({place_id}, URL: {image_url})")
        return None
    except requests.exceptions.RequestException as e:
        print(f"이미지 다운로드 오류 ({place_id}, URL: {image_url}): {e}")
        return None
    except Exception as e:
        print(f"S3 업로드 또는 처리 오류 ({place_id}): {e}")
        return None

def process_images_and_update_json(json_file_path, output_json_path):
    """Loads JSON, uploads images to S3, updates image URLs in the data, and saves to a new JSON file."""
    store_data = load_json_data(json_file_path)
    if not store_data:
        return False # Indicate failure

    total_stores = len(store_data) # 총 가게 수
    print(f"로드된 가게 데이터 수: {total_stores}")
    processed_count = 0
    failed_count = 0
    skipped_missing_url_count = 0
    skipped_missing_id_count = 0

    updated_stores_data = [] # 수정된 데이터를 저장할 새 리스트

    for i, store in enumerate(store_data): # enumerate 사용
        place_id = store.get('place_id')
        original_photo_url = store.get('photo_url')
        name = store.get('name', 'N/A') # For logging

        # 원본 데이터를 복사하여 수정 (원본 리스트는 변경하지 않음)
        current_store_data = store.copy()

        # 진행 상황 출력 (현재 번호 / 전체 수)
        print(f"처리 중 [{i+1}/{total_stores}]: '{name}' (ID: {place_id})", end=' ... ')

        if not place_id:
            print(f"경고: place_id 없음. 건너<0xEB><0x9A><0x8D>니다.") # 더 간결하게
            skipped_missing_id_count += 1
            updated_stores_data.append(current_store_data) # 원본 데이터 추가
            continue

        if not original_photo_url:
            print(f"정보: photo_url 없음. 건너<0xEB><0x9A><0x8D>니다.") # 더 간결하게
            skipped_missing_url_count += 1
            updated_stores_data.append(current_store_data) # 원본 데이터 추가
            continue

        cloudfront_url = upload_image_to_s3(original_photo_url, place_id)

        if cloudfront_url:
            processed_count += 1
            print(f"성공 (URL: {cloudfront_url})") # 결과 출력
            # 복사된 데이터의 'photo_url'을 CloudFront URL로 업데이트
            current_store_data['photo_url'] = cloudfront_url
        else:
            failed_count += 1
            print(f"실패") # 결과 출력
            # 업로드 실패 시 photo_url은 원본 그대로 유지됨

        updated_stores_data.append(current_store_data) # 처리된 (또는 원본) 데이터 추가

    print(f"\n--- 이미지 처리 완료 ---") # 줄바꿈 추가
    print(f"S3 업로드 성공: {processed_count}")
    print(f"S3 업로드 실패: {failed_count}")
    print(f"ID 없음 건너<0xEB><0x9A><0x9C>: {skipped_missing_id_count}")
    print(f"URL 없음 건너<0xEB><0x9A><0x9C>: {skipped_missing_url_count}")

    # 수정된 데이터를 새 JSON 파일로 저장
    try:
        print(f"수정된 데이터를 '{output_json_path}' 파일에 저장 중...")
        # 출력 디렉토리가 없으면 생성
        output_dir = os.path.dirname(output_json_path)
        if output_dir: # 루트 디렉토리가 아닌 경우에만 생성
             os.makedirs(output_dir, exist_ok=True)

        with open(output_json_path, 'w', encoding='utf-8') as outfile:
            # updated_stores_data 리스트를 JSON 파일로 저장
            json.dump(updated_stores_data, outfile, ensure_ascii=False, indent=4)
        print(f"성공적으로 '{output_json_path}' 파일에 저장했습니다.")
        return True # 성공 반환
    except Exception as e:
        print(f"오류: 수정된 데이터를 JSON 파일에 저장하는 중 오류 발생 - {e}")
        return False # 실패 반환

def main():
    start_time = time.time()  # 시작 시간 기록

    script_dir = Path(__file__).resolve().parent
    # 입력 JSON 파일 경로 설정
    input_json_name = "제주_전체_가게정보.json"
    input_json_path = os.path.join(script_dir, "구글맵_데이터", input_json_name)

    # 출력 JSON 파일 경로 설정 (새 파일 이름)
    output_json_name = "제주_전체_빵집_가게정보_processed.json"
    output_json_path = os.path.join(script_dir, "구글맵_데이터", output_json_name)

    if not os.path.exists(input_json_path):
         print(f"오류: 입력 JSON 파일을 찾을 수 없습니다 - {input_json_path}")
         return

    # 이미지 처리 및 업데이트된 JSON 파일 저장 함수 호출
    success = process_images_and_update_json(input_json_path, output_json_path)

    if success:
        print("\n작업 완료: 이미지가 S3에 업로드되었고, URL이 업데이트된 JSON 파일이 생성되었습니다.") # 줄바꿈 추가
        print(f"다음 단계: '{output_json_name}' 파일을 사용하여 'json_to_db.py' 스크립트를 실행하세요.")
    else:
        print("\n작업 중 오류가 발생했습니다.") # 줄바꿈 추가

    end_time = time.time()  # 종료 시간 기록
    execution_time = end_time - start_time  # 실행 시간 계산
    print(f"최종 실행 시간: {execution_time:.2f}초")  # 실행 시간 출력

if __name__ == "__main__":
    main() 