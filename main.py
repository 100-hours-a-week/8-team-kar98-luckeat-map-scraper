#!/usr/bin/env python
# coding: utf-8

import requests
# import pandas as pd
import json
import time
from datetime import datetime
import os
from openai import OpenAI
import math
import re
import concurrent.futures # 추가

# .env 파일 로드
from dotenv import load_dotenv
load_dotenv(override=True)  # 기존 환경변수보다 .env 파일을 우선시

# API 키 로딩 확인
OPENAI_API_KEY = os.getenv('OPENAI_API_KEY')
API_KEY = os.getenv('GOOGLE_API_KEY')

# API 키 확인 및 로그 출력
print(f"OpenAI API 키: {OPENAI_API_KEY[:4]}...{OPENAI_API_KEY[-4:] if OPENAI_API_KEY else '로드 실패'}")
print(f"Google API 키: {API_KEY[:4]}...{API_KEY[-4:] if API_KEY else '로드 실패'}")

# .env 파일 내용 출력 (디버깅용)
print("\n.env 파일의 API 키 정보:")
with open('.env', 'r') as f:
    env_content = f.readlines()
    for line in env_content:
        if line.strip().startswith('GOOGLE_API_KEY='):
            key_value = line.strip().split('=', 1)[1]
            print(f"GOOGLE_API_KEY= {key_value[:4]}...{key_value[-4:]}")
        elif line.strip().startswith('OPENAI_API_KEY='):
            key_value = line.strip().split('=', 1)[1]
            print(f"OPENAI_API_KEY= {key_value[:4]}...{key_value[-4:]}")

if not API_KEY:
    raise ValueError("Google API 키가 설정되지 않았습니다. .env 파일을 확인하세요.")
if not OPENAI_API_KEY:
    raise ValueError("OpenAI API 키가 설정되지 않았습니다. .env 파일을 확인하세요.")

# 다시 환경 변수에서 API 키 할당 (확실히 하기 위해)
API_KEY = os.getenv('GOOGLE_API_KEY')
print(f"최종 확인된 Google API 키: {API_KEY[:4]}...{API_KEY[-4:]}")

# OpenAI 클라이언트 설정
client = OpenAI(api_key=OPENAI_API_KEY)

# 수집할 총 데이터 개수 제한
TOTAL_LIMIT = 1000

# 결과를 저장할 디렉토리 생성
output_dir = "구글맵_데이터"
if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# 검색어 설정
language = "ko"  # 결과 언어

# 제외할 키워드 목록
EXCLUDE_KEYWORDS = [
  "스타벅스",
  "이디야커피",
  "빽다방",
  "할리스커피",
  "투썸플레이스",
  "엔제리너스커피",
  "커피빈",
  "카페베네",
  "테라로사",
  "파스쿠찌",
  "커피스미스",
  "빈스빈스",
  "탐앤탐스",
  "드롭탑커피",
  "컴포즈커피",
  "커피에반하다",
  "메가커피",
  "팀홀튼",
  "블루보틀",
  "아몽즈커피",
  "바나타이거",
  "벤티프레소",
  "오슬랑커피",
  "고더커피",
  "이삐커피",
  "카페 마일로",
  "백억커피",
  "피카커피",
  "커피나무",
  "폴바셋"
]

def get_places(query, location, radius, api_key, language="ko", page_token=None):
    endpoint_url = "https://maps.googleapis.com/maps/api/place/textsearch/json"

    params = {
        "query": query,
        "key": api_key,
        "language": language
    }

    # 위치와 반경 매개변수 설정 (첫 번째 시도)
    if location and location.strip():
        params["location"] = location
        if radius:
            params["radius"] = radius

    # 페이지 토큰이 있으면 location, radius 제거 (API 정책)
    if page_token:
        params["pagetoken"] = page_token
        params.pop("location", None) # location 키가 있으면 제거
        params.pop("radius", None)   # radius 키가 있으면 제거
    # 페이지 토큰 없을 때만 location/radius 파라미터 유지됨
    # else:
    #     if location and location.strip():
    #         params["location"] = location
    #         if radius:
    #             params["radius"] = radius

    print(f"API 요청 (1차): 쿼리={query}, 위치={params.get('location', '지정 안 함')}, 반경={params.get('radius', '지정 안 함')}, 페이지토큰={page_token if page_token else '없음'}")

    try:
        response = requests.get(endpoint_url, params=params)
        response.raise_for_status() # HTTP 오류 발생 시 예외 발생
        result = response.json()
    except requests.exceptions.RequestException as e:
        print(f"API 요청 중 네트워크 오류 발생: {e}")
        return None
    except json.JSONDecodeError:
        print(f"API 응답 JSON 파싱 오류. 응답 내용: {response.text}")
        return None

    print(f"API 응답 상태 (1차): {result.get('status', '상태 없음')}")
    if result.get('status') == "INVALID_REQUEST":
        print(f"API 오류 메시지: {result.get('error_message', '없음')}")
        print("INVALID_REQUEST 오류입니다. API 키 설정이나 요청 파라미터를 확인하세요.")
        return result # 오류 상태 반환
    elif result.get('status') not in ["OK", "ZERO_RESULTS"]:
        print(f"API 오류 메시지: {result.get('error_message', '없음')}")
        # 다른 오류 상태 처리 (예: OVER_QUERY_LIMIT, REQUEST_DENIED 등)
        return result # 오류 상태 반환

    # 첫 번째 시도 결과가 'OK' 이지만 결과가 없는 경우, 위치/반경 없이 재시도
    # status가 "OK"이고 results가 비어있을 때만 재시도
    if result.get('status') == "OK" and not result.get("results"):
        print("위치/반경 제한으로 결과를 찾지 못했을 수 있습니다. 위치/반경 없이 다시 시도합니다.")
        # 두 번째 시도를 위한 파라미터 재설정 (location, radius 제외)
        params_retry = {
            "query": query, # '판교도'가 이미 포함된 쿼리
            "key": api_key,
            "language": language
        }
        if page_token:
            params_retry["pagetoken"] = page_token

        print(f"API 요청 (2차): 쿼리={query}, 페이지토큰={page_token if page_token else '없음'}")

        try:
            response_retry = requests.get(endpoint_url, params=params_retry)
            response_retry.raise_for_status()
            result_retry = response_retry.json()
        except requests.exceptions.RequestException as e:
            print(f"API 재시도 중 네트워크 오류 발생: {e}")
            return result # 이전 결과라도 반환하거나 None 반환
        except json.JSONDecodeError:
            print(f"API 재시도 응답 JSON 파싱 오류. 응답 내용: {response_retry.text}")
            return result

        print(f"API 응답 상태 (2차): {result_retry.get('status', '상태 없음')}")
        if result_retry.get('status') == "INVALID_REQUEST":
             print(f"API 오류 메시지 (2차): {result_retry.get('error_message', '없음')}")
             print("재시도에서도 INVALID_REQUEST 오류입니다.")
        # 재시도 결과를 최종 결과로 사용
        return result_retry

    # 첫 번째 시도 결과가 'OK'이고 결과가 있거나, 'ZERO_RESULTS'인 경우 그대로 반환
    return result

def get_place_details(place_id, api_key, language="ko"):
    endpoint_url = "https://maps.googleapis.com/maps/api/place/details/json"
    params = {
        "place_id": place_id,
        "key": api_key,
        "language": language,
        "fields": "name,geometry,rating,formatted_address,formatted_phone_number,website,review,opening_hours,photo"
    }
    response = requests.get(endpoint_url, params=params)
    if response.status_code != 200:
        print(f"에러: {response.status_code}, {response.text}")
        return None
    return response.json()

def get_photo_url(photo_reference, api_key, max_width=400):
    """
    Google Places Photo API를 사용하여 사진 URL을 가져옵니다.
    """
    photo_url = f"https://maps.googleapis.com/maps/api/place/photo?maxwidth={max_width}&photoreference={photo_reference}&key={api_key}"
    return photo_url

def reverse_geocode(lat, lng, api_key):
    """
    Google Geocoding API를 사용하여 위도, 경도 정보를 바탕으로 역지오코딩 후 도로명주소를 반환합니다.
    language=ko 옵션을 추가하여 한글 주소만 반환되도록 설정합니다.
    result_type 매개변수를 추가하여 더 정확한 주소를 요청합니다.
    """
    geocode_url = f"https://maps.googleapis.com/maps/api/geocode/json?latlng={lat},{lng}&key={api_key}&language=ko&result_type=street_address|sublocality|route"
    response = requests.get(geocode_url)
    if response.status_code == 200:
        data = response.json()
        if data.get("results"):
            # 첫 번째 결과 사용
            formatted_address = data["results"][0]["formatted_address"]
            
            # 주소가 너무 짧은경우 확인
            if len(formatted_address) < 10:
                # 결과 타입 제한 없이 다시 시도
                fallback_url = f"https://maps.googleapis.com/maps/api/geocode/json?latlng={lat},{lng}&key={api_key}&language=ko"
                fallback_response = requests.get(fallback_url)
                if fallback_response.status_code == 200:
                    fallback_data = fallback_response.json()
                    if fallback_data.get("results"):
                        # 모든 결과 중에서 가장 자세한 주소 찾기
                        best_address = ""
                        for result in fallback_data["results"]:
                            addr = result["formatted_address"]
                            # 더 긴 주소를 선택 (더 상세할 가능성이 높음)
                            if len(addr) > len(best_address):
                                best_address = addr
                        
                        # 더 자세한 주소가 있으면 사용
                        if len(best_address) > len(formatted_address):
                            return best_address
            
            return formatted_address
    return ""

def summarize_reviews(reviews, max_reviews=5):
    """
    최대 max_reviews개의 리뷰 텍스트를 결합하여 GPT API를 통해 요약문을 생성합니다.
    최신 OpenAI API의 동기 호출 방식을 사용합니다.
    """
    # 리뷰 텍스트 5개(또는 max_reviews 개)를 결합
    reviews_text = "\n".join(rev for rev in reviews[:max_reviews] if rev.strip())
    if not reviews_text:
        return ""

    prompt = f"다음 리뷰들을 300자 안으로 간략하게 요약해줘:\n{reviews_text}\n요약:"

    try:
        response = client.chat.completions.create(# 여기서 직접 openai.ChatCompletion.create를 호출합니다.
            model="gpt-4o-mini",  # 최신 모델 사용 (필요 시 모델명 변경)
            messages=[
                {"role": "system", "content": "You are a helpful assistant that summarizes reviews."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.5,
            max_tokens=300)  # 요약문에 필요한 최대 토큰 수 설정)
        summary = response.choices[0].message.content.strip()
        return summary
    except Exception as e:
        print(f"리뷰 요약 오류 (프롬프트: {prompt[:50]}...): {e}")
        # exit() # 프로그램 종료 대신 오류 메시지 출력 후 빈 문자열 반환
        return ""

def is_excluded_place(place_name):
    """
    가게 이름에 제외 키워드가 포함되어 있는지 확인합니다.
    
    Parameters:
        place_name (str): 가게 이름
        
    Returns:
        bool: 제외 키워드가 포함되어 있으면 True, 아니면 False
    """
    if not place_name:
        return False
    
    for keyword in EXCLUDE_KEYWORDS:
        if keyword in place_name:
            return True
    return False

def clean_text_for_excel(text):
    """
    Excel에서 사용할 수 없는 문자를 제거하거나 대체합니다.
    
    Parameters:
        text: 정제할 텍스트
        
    Returns:
        str: 정제된 텍스트
    """
    if not isinstance(text, str):
        return text
    
    # Excel에서 문제가 될 수 있는 문자 제거/대체
    # 0x00-0x1F 범위의 제어 문자 제거 (탭, 줄바꿈, 캐리지 리턴 제외)
    text = re.sub(r'[\x00-\x08\x0B\x0C\x0E-\x1F]', '', text)
    
    # 줄바꿈 문자를 공백으로 대체
    text = text.replace('\n', ' ').replace('\r', ' ')
    
    return text

def clean_data_for_excel(data_list):
    """
    모든 데이터 필드를 Excel 저장에 적합하게 정제합니다.
    
    Parameters:
        data_list: 정제할 데이터 리스트
        
    Returns:
        list: 정제된 데이터 리스트
    """
    cleaned_data = []
    
    for item in data_list:
        cleaned_item = {}
        for key, value in item.items():
            if isinstance(value, str):
                cleaned_item[key] = clean_text_for_excel(value)
            elif isinstance(value, list) and all(isinstance(i, str) for i in value):
                cleaned_item[key] = [clean_text_for_excel(i) for i in value]
            else:
                cleaned_item[key] = value
        cleaned_data.append(cleaned_item)
    
    return cleaned_data

# 프로그램 시작 시간 기록
start_time = time.time()
print(f"프로그램 시작 시간: {datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')}")

# 여러 검색어와 지역 설정
search_queries = [
    {
        "query": "한식",
        "location": "37.395045, 127.111028",
        "radius": 2000,
        "grid_radius": 50,
        "area_name": "판교" 
    },
    {
        "query": "중식",
        "location": "37.395045, 127.111028",
        "radius": 2000,
        "grid_radius": 50,
        "area_name": "판교" 
    },
    {
        "query": "양식",
        "location": "37.395045, 127.111028",
        "radius": 2000,
        "grid_radius": 50,
        "area_name": "판교" 
    },
    {
        "query": "일식",
        "location": "37.395045, 127.111028",
        "radius": 2000,
        "grid_radius": 50,
        "area_name": "판교" 
    },
    {
        "query": "베이커리",
        "location": "37.395045, 127.111028",
        "radius": 2000,
        "grid_radius": 50,
        "area_name": "판교" 
    },
    {
        "query": "포케",
        "location": "37.395045, 127.111028",
        "radius": 2000,
        "grid_radius": 50,
        "area_name": "판교" 
    }
]

# 새로운 그리드 생성 함수 추가
def generate_grid(center_lat, center_lng, main_radius, cell_radius):#
    """
    중심 좌표를 기준으로 그리드 셀 생성
    
    Parameters:
        center_lat (float): 중심점 위도
        center_lng (float): 중심점 경도
        main_radius (float): 전체 검색 반경(m)
        cell_radius (float): 각 셀 반경(m)
    
    Returns:
        list: 생성된 그리드 좌표 리스트
    """
    grid = []
    earth_radius = 6371000  # 지구 반지름(m)
    
    # 수정 제안: 셀 중심 간격을 셀 반경의 2배로 설정
    step_multiplier = 2.0  # 셀 반경 대비 중심 간격 배율 (2.0이면 지름 간격)
    
    # steps 계산 추가 (NameError 해결)
    if cell_radius <= 0: # 0으로 나누는 것을 방지
        steps = 0
    else:
        # 셀 지름으로 전체 반경을 커버하는데 필요한 스텝 수 계산
        steps = math.ceil(main_radius / (cell_radius * step_multiplier))
        
    lat_step = (cell_radius * step_multiplier / earth_radius) * (180 / math.pi) # math.pi 사용
    
    # 경도 스텝 계산은 수정된 lat_step을 기반으로 동일하게 수행됨
    # lng_step = lat_step / math.cos(math.radians(abs(curr_lat)))
    
    # 각 셀마다 순회
    for i in range(-steps, steps+1):
        curr_lat = center_lat + i * lat_step
        
        # 경도 변화량은 위도에 따라 달라짐 (코사인 법칙)
        # 위도가 높을수록 경도 1도당 거리가 줄어듦
        lng_step = lat_step / math.cos(math.radians(abs(curr_lat)))
        
        for j in range(-steps, steps+1):
            new_lat = center_lat + i * lat_step
            new_lng = center_lng + j * lng_step
            
            # 중심점으로부터의 실제 거리 계산 (Haversine 공식의 근사치)
            dlat = new_lat - center_lat
            dlng = new_lng - center_lng
            a = math.sin(math.radians(dlat)/2)**2 + math.cos(math.radians(center_lat)) * math.cos(math.radians(new_lat)) * math.sin(math.radians(dlng)/2)**2
            distance = 2 * earth_radius * math.asin(math.sqrt(a))
            
            # 전체 반경 내에 있는 셀만 추가
            if distance <= main_radius:
                grid.append(f"{new_lat},{new_lng}")
    
    return grid

def calculate_distance(lat1, lng1, lat2, lng2):
    """
    두 지점 간의 거리를 계산합니다 (Haversine 공식 사용).
    
    Parameters:
        lat1, lng1: 첫 번째 지점의 위도와 경도
        lat2, lng2: 두 번째 지점의 위도와 경도
    
    Returns:
        float: 두 지점 간의 거리(미터)
    """
    earth_radius = 6371000  # 지구 반지름(m)
    
    dlat = math.radians(lat2 - lat1)
    dlng = math.radians(lng2 - lng1)
    
    a = math.sin(dlat/2)**2 + math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlng/2)**2
    distance = 2 * earth_radius * math.asin(math.sqrt(a))
    
    return distance

def adaptive_grid(center_lat, center_lng, main_radius, min_cell_radius=100, max_cell_radius=1000, result_threshold=20, query=None, api_key=None, language="ko"):
    """
    적응형 그리드 생성 함수: 검색 결과 수에 따라 셀 크기를 동적으로 조정합니다.
    결과가 임계값 이상이면 셀 반경이 최소 반경에 도달할 때까지 재귀적으로 분할합니다.

    Returns:
        list: (coordinate_string, cell_radius) 튜플의 리스트
    """
    if not query or not api_key:
        # API 키나 쿼리가 없으면 기본 그리드 생성 (반경 정보 추가) - 이 경우는 거의 없을 것으로 예상
        grid_coords = generate_grid(center_lat, center_lng, main_radius, min_cell_radius)
        return [(coord, min_cell_radius) for coord in grid_coords]

    print(f"적응형 그리드 생성 시작: 중심점=({center_lat}, {center_lng}), 전체반경={main_radius}m")
    print(f"  최대 셀 반경: {max_cell_radius}m, 최소 셀 반경: {min_cell_radius}m, 분할 임계값: {result_threshold}개")

    # 재귀 호출을 위한 내부 헬퍼 함수 정의
    processed_cells_cache = set() # 재귀 중복 처리 방지용 캐시

    def _recursive_grid_search(cell_lat, cell_lng, current_radius):
        """ 재귀적으로 셀을 검색하고 분할하는 내부 함수 """
        # 캐시 키 생성 (위도, 경도, 반경 조합)
        cell_key = f"{cell_lat:.6f},{cell_lng:.6f},{current_radius:.1f}"
        # 이미 처리된 셀이면 빈 리스트 반환 (중복 방지)
        if cell_key in processed_cells_cache:
            # print(f"    └ 캐시됨: ({cell_lat},{cell_lng}), 반경={current_radius}m") # 디버깅용
            return []
        processed_cells_cache.add(cell_key) # 처리 시작 시 캐시에 추가

        print(f"\n  셀 처리 중: 중심=({cell_lat},{cell_lng}), 반경={current_radius}m")

        # 현재 셀 위치에서 샘플 검색 수행
        location_str = f"{cell_lat},{cell_lng}" # API 호출용 좌표 문자열
        sample_result = get_places(query, location_str, current_radius, api_key, language)

        # API 호출 실패 또는 유효하지 않은 결과 처리
        if not sample_result or "results" not in sample_result:
            status = sample_result.get('status', 'UNKNOWN') if sample_result else 'NO_RESPONSE'
            print(f"    └ API 결과 없음 또는 오류 (Status: {status}). 이 셀 탐색 중단.")
            return [] # 오류/결과없음 시 빈 리스트 반환

        result_count = len(sample_result.get("results", []))
        print(f"    └ 검색 결과: {result_count}개")

        # CASE 1: 결과 수가 임계값 미만
        if result_count < result_threshold:
            if result_count > 0: # 결과가 1개 이상 ~ 임계값 미만
                print(f"    └ 결과 임계값 미만({result_count} < {result_threshold}), 현재 셀 유지 (반경: {current_radius}m)")
                # 현재 셀의 좌표와 반경을 리스트에 담아 반환
                return [(location_str, current_radius)]
            else: # 결과가 0개 (ZERO_RESULTS)
                print(f"    └ 결과 0개, 이 셀은 최종 그리드에 포함하지 않음.")
                return [] # 0개 셀은 제외

        # CASE 2: 결과 수가 임계값 이상 (result_count >= result_threshold)
        else:
            # CASE 2-1: 현재 반경이 최소 반경보다 큼 -> 분할 시도
            if current_radius > min_cell_radius + 1e-6: # 최소 반경보다 확실히 클 때만 분할
                print(f"    └ 결과 임계값 이상 ({result_count} >= {result_threshold})이고, 현재 반경({current_radius:.1f}m) > 최소 반경({min_cell_radius}m). 셀 분할 시도.")

                # 다음 단계 셀 반경 계산 (예: 현재 반경의 1/2, 최소 반경 보장)
                # 분할 시 최소 반경 밑으로 내려가지 않도록 max 함수 사용
                next_radius = max(current_radius / 2, min_cell_radius)
                print(f"      └ 다음 분할 반경 계산: max({current_radius / 2:.1f}m, {min_cell_radius}m) => {next_radius:.1f}m")

                # 현재 셀 영역 내에서 더 작은 그리드 생성
                # generate_grid(중심 위도, 중심 경도, 영역 반경, 생성할 셀 반경)
                sub_cell_coords = generate_grid(cell_lat, cell_lng, current_radius, next_radius)
                print(f"      └ {len(sub_cell_coords)}개의 하위 셀 생성됨 (목표 반경: {next_radius:.1f}m). 각 하위 셀에 대해 재귀 호출 시작...")

                final_sub_cells = [] # 이 분기에서 생성된 최종 셀들을 저장할 리스트
                sub_cell_count = 0
                for sub_coord_str in sub_cell_coords:
                    sub_lat, sub_lng = map(float, sub_coord_str.split(','))
                    sub_cell_count += 1
                    # print(f"        └ 하위 셀 {sub_cell_count}/{len(sub_cell_coords)} ({sub_coord_str}) 재귀 처리 시작...") # 로그 상세도 조절 가능

                    # 재귀 호출: 각 하위 셀에 대해 동일한 프로세스 반복
                    recursive_results = _recursive_grid_search(sub_lat, sub_lng, next_radius)
                    # 재귀 호출 결과(리스트)를 최종 리스트에 확장
                    final_sub_cells.extend(recursive_results)

                print(f"      └ 모든 하위 셀({len(sub_cell_coords)}개) 재귀 처리 완료. 이 분기에서 총 {len(final_sub_cells)}개의 최종 셀 반환됨.")
                return final_sub_cells # 재귀적으로 수집된 셀 리스트 반환

            # CASE 2-2: 현재 반경이 이미 최소 반경이거나 작음 -> 분할 중지
            else: # current_radius <= min_cell_radius + 1e-6
                print(f"    └ 결과 임계값 이상 ({result_count} >= {result_threshold})이지만, 이미 최소 반경({min_cell_radius}m) 도달 (현재 반경: {current_radius:.1f}m). 분할 중지.")
                # 현재 셀 (최소 반경)을 최종 결과에 포함
                return [(location_str, current_radius)] # 현재 (최소) 반경의 셀 반환

    # --- adaptive_grid 함수의 메인 로직 시작 ---
    # 1. 초기 대형 그리드 생성 (max_cell_radius 사용)
    # generate_grid 함수는 중심점 기준 main_radius 내에 있으면서 max_cell_radius 크기의 셀 중심 좌표 리스트 반환
    initial_large_grid_coords = generate_grid(center_lat, center_lng, main_radius, max_cell_radius)
    print(f"초기 대형 그리드 생성 완료: {len(initial_large_grid_coords)}개 셀 (초기 셀 반경: {max_cell_radius}m)")

    final_grid_cells = [] # 최종 결과 셀 (좌표, 반경) 튜플을 저장할 리스트
    processed_initial_cells = 0

    # 2. 각 초기 대형 셀에 대해 재귀적 탐색 시작
    for coord_str in initial_large_grid_coords:
        lat, lng = map(float, coord_str.split(','))
        processed_initial_cells += 1
        print(f"\n--- 초기 대형 셀 {processed_initial_cells}/{len(initial_large_grid_coords)} 처리 시작 ---")
        print(f"  └ 셀 중심: {coord_str}, 시작 반경: {max_cell_radius}m")

        # 재귀 함수 호출 (각 초기 셀에 대해 시작 반경은 max_cell_radius)
        results_from_cell = _recursive_grid_search(lat, lng, max_cell_radius)
        # 재귀 호출 결과(리스트)를 최종 리스트에 추가
        final_grid_cells.extend(results_from_cell)
        print(f"--- 초기 대형 셀 {processed_initial_cells} 처리 완료: 이 셀에서 {len(results_from_cell)}개의 최종 검색 셀 추가됨 ---")

    print(f"\n\n적응형 그리드 생성 완료 (모든 초기 셀 재귀 탐색 종료).")
    print(f"총 {len(final_grid_cells)}개의 검색 대상 셀 생성 (중복 제거 전).")

    # 3. 최종 결과에서 중복 제거 (동일 좌표가 다른 반경으로 포함될 수 있음)
    # 좌표 문자열을 키로 사용하여 딕셔너리로 중복 제거. 값이 덮어써지므로 마지막 처리된 반경 유지됨.
    # 만약 더 작은 반경을 우선하려면 로직 수정 필요.
    final_grid_dict = {}
    for coord, radius in final_grid_cells:
        final_grid_dict[coord] = (coord, radius)

    final_unique_grid = list(final_grid_dict.values()) # 딕셔너리의 값들(튜플)만 리스트로 변환
    print(f"중복 제거 후 최종 검색 대상 셀 수: {len(final_unique_grid)}개")

    # 디버깅 및 정보용: 최종 셀 목록 일부 출력
    if final_unique_grid:
        print("\n최종 생성된 검색 셀 목록 (상위 5개):")
        for i, (coord, radius) in enumerate(final_unique_grid[:min(5, len(final_unique_grid))]):
            print(f"  {i+1}: 좌표={coord}, 반경={radius:.1f}m")
        if len(final_unique_grid) > 5:
            print("  ...")
        print("-" * 20) # 구분선

    # 최종적으로 중복 제거된 (좌표, 반경) 튜플 리스트 반환
    return final_unique_grid
# --- adaptive_grid 함수 정의 끝 ---

collected_count = 0

# 가게 정보를 저장할 리스트
places_data = []
# 리뷰 정보를 저장할 리스트
reviews_data = []

# 기존 중복 체크 로직 강화
seen_ids = set()

# 지역별 수집 결과 통계
area_stats = {}

# 쿼리별 데이터를 저장할 딕셔너리
query_data_dict = {}

# 카테고리별 수집된 장소 ID를 저장할 딕셔너리 추가
category_ids = {}
# 카테고리간 중복 제거된 장소를 저장할 딕셔너리
unique_by_category = {}

# ThreadPoolExecutor 생성 (최대 워커 수 지정, 필요시 조절)
# with 문을 사용하여 자동으로 shutdown 되도록 함
with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
    for search_item in search_queries:
        if collected_count >= TOTAL_LIMIT:
            break
            
        current_query = search_item["query"]
        # 카테고리별 ID 셋 초기화
        if current_query not in category_ids:
            category_ids[current_query] = set()
        
        # 지역별 시작 시간
        area_start_time = time.time() 
        print(f"\n===== {search_item['area_name']} 지역 {current_query} 검색 시작: {datetime.fromtimestamp(area_start_time).strftime('%Y-%m-%d %H:%M:%S')} =====")
        # 현재까지 수집된 총 고유 장소 수 (전체 카테고리)
        print(f"현재까지 수집된 고유 장소 수: {len(seen_ids)}, 현재 카테고리({current_query})에서 수집된 장소 수: {len(category_ids[current_query])}")
        
        # 현재 쿼리의 결과를 저장할 리스트 초기화
        current_query_places = []
        # 현재 쿼리의 리뷰 요약 작업을 저장할 리스트
        summary_futures = [] # 추가
        
        # 적응형 그리드 생성 (반환값이 (좌표, 반경) 튜플 리스트임)
        center_lat, center_lng = map(float, search_item["location"].split(','))
        min_cell_radius = search_item.get("grid_radius", 100) # search_queries에서 가져옴
        # max_cell_radius 계산 시 min_cell_radius보다 작아지지 않도록 보장
        # 전체 반경의 1/3 또는 1000m 중 작은 값으로 하되, min_cell_radius*1.1 보다는 크게 설정
        max_cell_radius = max(min(search_item.get("radius", 3000) / 3, 1000), min_cell_radius * 1.1) 

        grid_points_with_radius = adaptive_grid( # 변수명 변경
            center_lat, center_lng,
            search_item["radius"], # 전체 검색 반경
            min_cell_radius=min_cell_radius, # 최소 셀 반경
            max_cell_radius=max_cell_radius, # 최대 셀 반경
            result_threshold=20, # 임계값을 20으로 수정
            query=search_item["query"],
            api_key=API_KEY,
            language=language
        )

        print(f"'{search_item['query']}' - {search_item['area_name']} 지역 적응형 그리드 검색 시작 (총 {len(grid_points_with_radius)}개 셀)")

        excluded_count = 0

        # 수정된 루프: 좌표와 반경을 함께 사용
        for i, (grid_point, cell_search_radius) in enumerate(grid_points_with_radius): # 튜플 언패킹
            print(f"현재 {search_item['area_name']} 지역 셀 검색: {i+1}/{len(grid_points_with_radius)} ({grid_point}, 반경: {cell_search_radius}m)")
            next_page_token = None
            page_count = 1
            stop_fetching = False
            
            # 현재 셀에서 수집된 장소 ID 임시 저장 (페이지네이션 중복 방지)
            current_cell_place_ids = set() 

            while True:
                try:
                    # get_places 호출 시 해당 셀의 반경(cell_search_radius) 사용
                    places_result = get_places(search_item["query"], grid_point, cell_search_radius, API_KEY, language, next_page_token)

                    if not places_result or "results" not in places_result:
                        status = places_result.get('status', 'UNKNOWN') if places_result else 'NO_RESPONSE'
                        if status == "ZERO_RESULTS":
                             print(f"  └ 검색 결과 없음 (Status: {status})")
                        else:
                             print(f"검색 결과가 없거나 API 오류 발생 (Status: {status}). 다음 셀로 이동합니다.")
                        break # 다음 셀로

                    page_places = places_result["results"]
                    print(f"  └ API 응답: 이 페이지에서 {len(page_places)}개 결과 수신")
                    
                    newly_added_count_in_page = 0
                    for place in page_places:
                        place_id = place["place_id"]
                        
                        # 중복 체크 - 전체 카테고리에서 이미 수집된 ID인지 확인
                        if place_id in seen_ids or place_id in current_cell_place_ids: 
                            # 디버깅 로그 추가: 왜 건너뛰는지 출력
                            print(f"    └ 중복 ID 건너뛰기: {place.get('name', '이름 없음')} ({place_id}) (전체 중복: {place_id in seen_ids}, 현재 셀 중복: {place_id in current_cell_place_ids})")
                            continue
                            
                        # 제외 키워드 가게 확인
                        place_name = place["name"]
                        if is_excluded_place(place_name):
                            print(f"  └ 제외된 가게: {place_name} (제외 키워드 포함)")
                            excluded_count += 1
                            continue # 다음 장소로

                        # --- 장소 상세 정보 가져오기 및 처리 ---
                        place_details = get_place_details(place_id, API_KEY, language)
                        if not place_details or "result" not in place_details:
                            print(f"  └ 상세 정보 조회 실패: {place_name} ({place_id})")
                            continue # 다음 장소로

                        result = place_details["result"]

                        # 위경도 정보 추출 (없으면 None)
                        location_data = result.get("geometry", {}).get("location", {})
                        latitude = location_data.get("lat", None)
                        longitude = location_data.get("lng", None)

                        # 역지오코딩을 통해 주소 가져오기 (실패 시 기존 formatted_address 사용)
                        rev_address = result.get("formatted_address", "") # 기본값 설정
                        if latitude and longitude:
                            # 역지오코딩 시도, 실패해도 기본값 유지됨
                            geo_addr = reverse_geocode(latitude, longitude, API_KEY)
                            if geo_addr: rev_address = geo_addr 
                        
                        # 이미지 URL 추가
                        photo_url = ""
                        if "photos" in result and result["photos"]:
                            photo_reference = result["photos"][0]["photo_reference"]
                            photo_url = get_photo_url(photo_reference, API_KEY)

                        # place_info 딕셔너리 생성
                        place_info = {
                            "place_id": place_id,
                            "name": result.get("name", ""),
                            "address": rev_address,
                            "phone": result.get("formatted_phone_number", ""),
                            "website": result.get("website", ""),
                            "rating": result.get("rating", 0),
                            "latitude": latitude,
                            "longitude": longitude,
                            "photo_url": photo_url,
                            "search_query": search_item["query"],
                            "area_name": search_item["area_name"]
                        }
                        if "opening_hours" in result and "weekday_text" in result["opening_hours"]:
                            place_info["opening_hours"] = "\\n".join(result["opening_hours"]["weekday_text"])
                        else:
                            place_info["opening_hours"] = ""

                        # 리뷰 텍스트 저장 및 요약 초기화 준비
                        reviews_texts = []
                        if "reviews" in result:
                            for review in result["reviews"]:
                                reviews_texts.append(review.get("text", ""))
                                # 개별 리뷰는 별도로 저장
                                review_info = {
                                    "place_id": place_id,
                                    "place_name": place_name, # place_info['name'] 사용해도 됨
                                    "author_name": review.get("author_name", ""),
                                    "rating": review.get("rating", 0),
                                    "time": datetime.fromtimestamp(review.get("time", 0)).strftime('%Y-%m-%d %H:%M:%S'),
                                    "text": review.get("text", ""),
                                    "language": review.get("language", "")
                                }
                                reviews_data.append(review_info)
                        
                        place_info["review_summary"] = ""  # 요약은 나중에 일괄 처리 (초기화)

                        # 리뷰 텍스트가 있을 경우 요약 작업 제출
                        if reviews_texts:
                            future = executor.submit(summarize_reviews, reviews_texts)
                            summary_futures.append((future, place_info)) # future와 place_info 참조 저장
                        # --- 상세 정보 처리 끝 ---
                        
                        # 성공적으로 추가된 경우
                        seen_ids.add(place_id)  # 전체 카테고리 중복 방지를 위한 ID 추가
                        category_ids[current_query].add(place_id)  # 현재 카테고리에 ID 추가
                        current_cell_place_ids.add(place_id) # 현재 셀 내 중복 방지
                        current_query_places.append(place_info) # 상세 정보를 current_query_places에 추가
                        collected_count += 1
                        newly_added_count_in_page += 1
                        print(f"    └ [{search_item['area_name']}] {collected_count}/{TOTAL_LIMIT}번째 장소 추가: {place_name}")

                        if collected_count >= TOTAL_LIMIT:
                            print(f"*** 수집 목표 {TOTAL_LIMIT}개 도달. 검색을 중단합니다. ***")
                            stop_fetching = True
                            break # inner for loop 종료

                    print(f"  └ 이 페이지에서 새로 추가된 장소: {newly_added_count_in_page}개")

                    if stop_fetching:
                        break # while loop 종료

                    # 다음 페이지 토큰 처리
                    next_page_token = places_result.get("next_page_token")
                    if not next_page_token:
                        print(f"  └ 이 셀({grid_point})의 마지막 페이지입니다.")
                        break # while loop 종료 (다음 셀로)

                    print(f"  └ 다음 페이지 토큰 존재. 2초 후 다음 페이지 로드...")
                    time.sleep(2) # 페이지 토큰 사용 시 필수 딜레이
                    page_count += 1

                except requests.exceptions.RequestException as req_e:
                     print(f"네트워크 오류 발생: {req_e}. 잠시 후 재시도하거나 다음 셀로 이동합니다.")
                     #time.sleep(5) # 잠시 대기
                     # 필요시 재시도 로직 추가 가능
                     break # 현재 셀 처리 중단하고 다음 셀로
                except Exception as e:
                    print(f"처리 중 예외 발생: {e}")
                    import traceback
                    traceback.print_exc() # 상세 오류 출력
                    break # while 루프 종료 (다음 셀로)

            # stop_fetching 플래그 확인: True이면 그리드 셀 루프도 중단
            if stop_fetching:
                break

            # 각 셀 처리 후 로그 개선 (현재 셀 정보 포함)
            current_area_count = len([p for p in current_query_places if p['area_name'] == search_item['area_name']])
            print(f"  └ 셀 ({grid_point}, 반경 {cell_search_radius}m) 처리 완료. 현재까지 '{search_item['area_name']}' 지역에서 총 {current_area_count}개의 장소 수집.")

        # --- 현재 쿼리/지역의 모든 셀 탐색 완료 후 리뷰 요약 결과 처리 ---
        if summary_futures:
            print(f"\n  └ [{search_item['area_name']} - {search_item['query']}] 리뷰 요약 결과 처리 시작 (총 {len(summary_futures)}개 작업)...")
            processed_summaries = 0
            for future, place_info_ref in summary_futures:
                try:
                    # future.result()는 해당 작업이 완료될 때까지 대기하고 결과를 반환
                    summary = future.result()
                    # 요약 결과를 place_info 딕셔너리에 업데이트하고 엑셀용으로 정제
                    place_info_ref["review_summary"] = clean_text_for_excel(summary)
                    processed_summaries += 1
                    # 진행 상황 로깅 (예: 10개마다)
                    if processed_summaries % 10 == 0 or processed_summaries == len(summary_futures):
                       print(f"    └ 요약 처리 진행: {processed_summaries}/{len(summary_futures)}")
                except Exception as e:
                    # future.result() 실행 중 또는 summarize_reviews 내부에서 처리되지 않은 예외 발생 시
                    place_id = place_info_ref.get("place_id", "알 수 없는 ID")
                    place_name = place_info_ref.get("name", "알 수 없는 이름")
                    print(f"    └ 리뷰 요약 결과 가져오기 중 오류 발생 (Place ID: {place_id}, Name: {place_name}): {e}")
                    place_info_ref["review_summary"] = "요약 중 오류 발생" # 데이터에 오류 발생 표시
            print(f"  └ 리뷰 요약 결과 처리 완료 ({processed_summaries}/{len(summary_futures)}개 완료).")
        else:
            print(f"\n  └ [{search_item['area_name']} - {search_item['query']}] 요약할 리뷰가 없습니다.")

        # 수집된 데이터를 현재 쿼리 리스트에 추가 (이 부분은 제거해도 될 듯, current_query_places를 직접 사용)
        # current_query_places.extend(places_data) # places_data는 이미 비어있으므로 이 줄은 불필요
        
        # 현재 쿼리의 데이터 정제 및 저장
        if current_query_places:
            # 이제 current_query_places에는 review_summary가 포함됨
            cleaned_query_places = clean_data_for_excel(current_query_places) # Excel 정제는 최종 단계에서 수행
            
            # 쿼리별 JSON 파일 저장
            query_json_path = os.path.join(output_dir, f"판교_{search_item['area_name']}_{search_item['query']}.json")
            with open(query_json_path, 'w', encoding='utf-8') as f:
                json.dump(cleaned_query_places, f, ensure_ascii=False, indent=4)
            
            print(f"판교 {search_item['area_name']} 지역 {search_item['query']} 검색 결과 저장 완료 (총 {len(cleaned_query_places)}개)")
            
            # 전체 데이터 사전에 현재 쿼리 결과 추가
            query_key = f"{search_item['area_name']}_{search_item['query']}"
            # query_data_dict[query_key] = cleaned_query_places # 정제된 데이터를 저장
            query_data_dict[query_key] = current_query_places # 원본(요약 포함) 데이터 저장 후 나중에 전체 정제

        # places_data 초기화 (다음 쿼리를 위해) - 이미 current_query_places를 사용하므로 불필요
        # places_data = []

        # stop_fetching 플래그 확인: True이면 전체 검색 중단
        if stop_fetching:
            print("\n*** 전체 수집 목표 도달로 모든 검색 쿼리 처리를 중단합니다. ***")
            break # search_queries 루프 중단

        # 이 카테고리에서 수집 완료 후 통계 출력
        print(f"\n=== {search_item['area_name']} {current_query} 검색 완료 ===")
        print(f"  ├ 이 카테고리에서 수집된 고유 장소 수: {len(category_ids[current_query])}개")
        print(f"  ├ 전체 카테고리에서 수집된 총 고유 장소 수: {len(seen_ids)}개")
        
        # 카테고리별 고유 장소 계산
        for cat_name, cat_ids in category_ids.items():
            if cat_name != current_query:
                overlap_count = len(cat_ids.intersection(category_ids[current_query]))
                if overlap_count > 0:
                    print(f"  ├ {cat_name} 카테고리와 {overlap_count}개 장소 중복")

# 모든 쿼리의 결과를 하나로 합쳐서 저장
if query_data_dict:
    all_places = []
    for places in query_data_dict.values():
        all_places.extend(places)
    
    # 전체 데이터 Excel 정제
    print("\n모든 수집 데이터에 대해 Excel 호환 문자 정제 시작...")
    cleaned_all_places = clean_data_for_excel(all_places)
    print("Excel 호환 문자 정제 완료.")

    # 전체 데이터 JSON 파일 저장 (정제된 데이터로)
    all_places_json_path = os.path.join(output_dir, "판교_전체_가게정보.json")
    with open(all_places_json_path, 'w', encoding='utf-8') as f:
        json.dump(cleaned_all_places, f, ensure_ascii=False, indent=4)
    
    print(f"\n판교 전체 가게정보 JSON 파일 저장 완료 (총 {len(cleaned_all_places)}개)")

# 리뷰 데이터 저장 (별도 파일)
if reviews_data:
    print("\n수집된 리뷰 데이터 정제 시작...")
    cleaned_reviews_data = clean_data_for_excel(reviews_data)
    print("리뷰 데이터 정제 완료.")
 
    reviews_json_path = os.path.join(output_dir, "판교_전체_리뷰정보.json")
    with open(reviews_json_path, 'w', encoding='utf-8') as f:
        json.dump(cleaned_reviews_data, f, ensure_ascii=False, indent=4)
    print(f"판교 전체 리뷰정보 JSON 파일 저장 완료 (총 {len(cleaned_reviews_data)}개)")

# 데이터 저장 완료 후 실행 시간 계산
end_time = time.time()
total_time = end_time - start_time
hours, remainder = divmod(total_time, 3600)
minutes, seconds = divmod(remainder, 60)

print("\n데이터 수집이 완료되었습니다.")
print(f"시작 시간: {datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')}")
print(f"종료 시간: {datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')}")
print(f"총 소요 시간: {int(hours)}시간 {int(minutes)}분 {seconds:.1f}초")

# 카테고리별 고유 장소 통계 출력
print("\n카테고리별 수집 통계:")
for category, ids in category_ids.items():
    print(f"  └ {category}: {len(ids)}개 장소 수집")
print(f"전체 고유 장소(중복 제외): {len(seen_ids)}개")

print(f"모든 파일은 '{output_dir}' 디렉토리에 저장되었습니다.")