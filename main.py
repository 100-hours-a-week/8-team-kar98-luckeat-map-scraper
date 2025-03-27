#!/usr/bin/env python
# coding: utf-8

import requests
import pandas as pd
import json
import time
from datetime import datetime
import os
from openai import OpenAI
import math
import re

client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
from dotenv import load_dotenv

# 수집할 총 데이터 개수 제한
TOTAL_LIMIT = 1000

# .env 파일 로드
load_dotenv()

# OpenAI GPT API 키 설정

# Google API 키 설정
API_KEY = os.getenv('GOOGLE_API_KEY')

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
        "location": location,
        "radius": radius,
        "key": api_key,
        "language": language
    }
    if page_token:
        params["pagetoken"] = page_token
    response = requests.get(endpoint_url, params=params)
    if response.status_code != 200:
        print(f"에러: {response.status_code}, {response.text}")
        return None
    return response.json()

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
    """
    geocode_url = f"https://maps.googleapis.com/maps/api/geocode/json?latlng={lat},{lng}&key={api_key}&language=ko"
    response = requests.get(geocode_url)
    if response.status_code == 200:
        data = response.json()
        if data.get("results"):
            return data["results"][0]["formatted_address"]
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
        print("리뷰 요약 오류:", e)
        exit()
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
        "query": "빵집",
        "location": "33.501616,126.531381",
        "radius": 10000,
        "grid_radius": 3000,
        "area_name": "시청" 
    },
    {
        "query": "빵집",
        "location": "33.488649, 126.488016",
        "radius": 10000,
        "grid_radius": 3000,
        "area_name": "호텔"
    },
    {
        "query": "빵집",
        "location": "33.477327, 126.374886",
        "radius": 10000,
        "grid_radius": 3000,
        "area_name": "애월"
    },
    {
        "query": "빵집",
        "location": "33.396560, 126.246921",
        "radius": 10000,
        "grid_radius": 3000,
        "area_name": "협재"
    },
    {
        "query": "빵집",
        "location": "33.543135, 126.670317",
        "radius": 10000,
        "grid_radius": 3000,
        "area_name": "함덕"
    },
    {
        "query": "빵집",
        "location": "33.556020, 126.796121",
        "radius": 10000,
        "grid_radius": 3000,
        "area_name": "월정리"
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
    
    # 필요한 그리드 셀 수 계산 (반올림하여 정확한 커버리지 확보)
    steps = max(1, round(main_radius / cell_radius))
    
    # 위도 변화량 계산 (라디안 -> 도)
    lat_step = (cell_radius / earth_radius) * (180 / 3.14159)
    
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

collected_count = 0

# 가게 정보를 저장할 리스트
places_data = []
# 리뷰 정보를 저장할 리스트
reviews_data = []

# 기존 중복 체크 로직 강화
seen_ids = set()

# 지역별 수집 결과 통계
area_stats = {}

for search_item in search_queries:
    if collected_count >= TOTAL_LIMIT:
        break
        
    # 지역별 시작 시간
    area_start_time = time.time()
    print(f"\n===== {search_item['area_name']} 지역 검색 시작: {datetime.fromtimestamp(area_start_time).strftime('%Y-%m-%d %H:%M:%S')} =====")
        
    # 그리드 생성
    center_lat, center_lng = map(float, search_item["location"].split(','))
    grid_points = generate_grid(center_lat, center_lng, 
                              search_item["radius"], 
                              search_item["grid_radius"])
    
    print(f"'{search_item['query']}' - {search_item['area_name']} 지역 그리드 검색 시작 (총 {len(grid_points)}개 셀)")
    
    # 제외 키워드 통계를 위한 카운터
    excluded_count = 0
    
    for i, grid_point in enumerate(grid_points):
        print(f"현재 {search_item['area_name']} 지역 셀 검색: {i+1}/{len(grid_points)} ({grid_point})")
        next_page_token = None
        page_count = 1
        stop_fetching = False 

        while True:
            try:
                # if next_page_token:
                #     time.sleep(2)  # 페이지 토큰 사용 시 딜레이
                places_result = get_places(search_item["query"], grid_point, search_item["radius"], API_KEY, language, next_page_token)
                if not places_result or "results" not in places_result:
                    print("검색 결과가 없거나 API 오류가 발생했습니다.")
                    break

                page_places = places_result["results"]

                for place in page_places:
                    if place["place_id"] in seen_ids:
                        continue
                    seen_ids.add(place["place_id"])
                    # 이미 수집한 장소인지 확인 (중복 제거)
                    if any(p["place_id"] == place["place_id"] for p in places_data):
                        continue
                        
                    if collected_count >= TOTAL_LIMIT:
                        stop_fetching = True
                        break

                    place_id = place["place_id"]
                    place_name = place["name"]
                    
                    # 제외 키워드가 포함된 가게 이름인지 확인
                    if is_excluded_place(place_name):
                        print(f"[{search_item['area_name']}] 제외된 가게: {place_name} (제외 키워드 포함)")
                        excluded_count += 1
                        continue

                    place_details = get_place_details(place_id, API_KEY, language)
                    if not place_details or "result" not in place_details:
                        continue

                    result = place_details["result"]

                    # 위경도 정보 추출 (없으면 None)
                    location_data = result.get("geometry", {}).get("location", {})
                    latitude = location_data.get("lat", None)
                    longitude = location_data.get("lng", None)

                    # 역지오코딩을 통해 주소 가져오기 (실패 시 기존 formatted_address 사용)
                    if latitude and longitude:
                        rev_address = reverse_geocode(latitude, longitude, API_KEY)
                    else:
                        rev_address = result.get("formatted_address", "")

                    # 이미지 URL 추가
                    photo_url = ""
                    if "photos" in result and result["photos"]:
                        photo_reference = result["photos"][0]["photo_reference"]
                        photo_url = get_photo_url(photo_reference, API_KEY)

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
                        place_info["opening_hours"] = "\n".join(result["opening_hours"]["weekday_text"])
                    else:
                        place_info["opening_hours"] = ""

                    # 리뷰 요약: 해당 가게의 리뷰 5개를 모아서 요약한 내용을 추가합니다.
                    reviews_texts = []
                    if "reviews" in result:
                        for review in result["reviews"]:
                            reviews_texts.append(review.get("text", ""))
                            # 개별 리뷰는 별도로 저장
                            review_info = {
                                "place_id": place_id,
                                "place_name": place_name,
                                "author_name": review.get("author_name", ""),
                                "rating": review.get("rating", 0),
                                "time": datetime.fromtimestamp(review.get("time", 0)).strftime('%Y-%m-%d %H:%M:%S'),
                                "text": review.get("text", ""),
                                "language": review.get("language", "")
                            }
                            reviews_data.append(review_info)

                    # 리뷰 텍스트 저장 및 요약 초기화
                    place_info["reviews_texts"] = reviews_texts  # 리뷰 텍스트 임시 저장
                    place_info["review_summary"] = ""  # 초기값 설정

                    places_data.append(place_info)
                    collected_count += 1
                    
                    print(f"[{search_item['area_name']}] {collected_count}번째 장소 추가: {place_name}")
                    
                    # 100개를 채우면 중지
                    if collected_count >= TOTAL_LIMIT:
                        stop_fetching = True
                        break
                    
                if stop_fetching or not next_page_token:
                    break
                    
                next_page_token = places_result.get("next_page_token")
                page_count += 1
                
            except Exception as e:
                print(f"오류 발생: {e}")
                break
            
        print(f"'{search_item['query']}' - {search_item['area_name']} 지역에서 총 {len([p for p in places_data if p['area_name'] == search_item['area_name']])}개의 장소 정보를 가져왔습니다.")
    
    # 지역별 종료 시간과 통계
    area_end_time = time.time()
    area_total_time = area_end_time - area_start_time
    area_count = len([p for p in places_data if p['area_name'] == search_item['area_name']])
    area_stats[search_item['area_name']] = {
        "count": area_count,
        "time": area_total_time,
        "excluded": excluded_count  # 제외된 가게 수 통계 추가
    }
    
    print(f"===== {search_item['area_name']} 지역 검색 종료: {datetime.fromtimestamp(area_end_time).strftime('%Y-%m-%d %H:%M:%S')} =====")
    print(f"소요 시간: {area_total_time:.1f}초 ({area_total_time/60:.1f}분)")
    print(f"수집된 가게 수: {area_count}개")
    print(f"제외된 가게 수: {excluded_count}개 (제외 키워드: {', '.join(EXCLUDE_KEYWORDS)})")

# 최종 통계 출력
print("\n===== 지역별 수집 결과 요약 =====")
print(f"{'지역명':<10} {'가게 수':<10} {'제외 수':<10} {'소요 시간(분)':<15}")
print("-" * 45)
total_places = 0
total_excluded = 0
for area, stats in area_stats.items():
    print(f"{area:<10} {stats['count']:<10} {stats.get('excluded', 0):<10} {stats['time']/60:.1f}분")
    total_places += stats['count']
    total_excluded += stats.get('excluded', 0)
print("=" * 45)
print(f"{'합계':<10} {total_places:<10} {total_excluded:<10}")

print(f"\n최종적으로 총 {len(places_data)}개의 장소 정보를 가져왔습니다. (제외된 장소: {total_excluded}개)")

# 모든 데이터 수집이 끝난 후 리뷰 요약 처리
print("\n리뷰 요약 작업 시작...")
# 지역별 요약 통계 추적
area_summary_counts = {area: 0 for area in set(place["area_name"] for place in places_data)}

for idx, place in enumerate(places_data, 1):
    area_name = place["area_name"]
    if place["reviews_texts"]:
        place["review_summary"] = summarize_reviews(place["reviews_texts"], max_reviews=5)
        area_summary_counts[area_name] += 1
    # 진행률 표시
    print(f"진행 현황: {idx}/{len(places_data)}개 장소 처리 - 현재: [{area_name}] {place['name']}", end='\r')

print("\n리뷰 요약 완료!")
for area, count in area_summary_counts.items():
    print(f"{area} 지역: {count}개 장소 리뷰 요약 완료")

if places_data:
    # 데이터 정제 - Excel 저장을 위해
    cleaned_places_data = clean_data_for_excel(places_data)
    places_df = pd.DataFrame(cleaned_places_data)
    
    # 지역별로 데이터 분리하여 저장
    for area in set(place["area_name"] for place in cleaned_places_data):
        area_places = [place for place in cleaned_places_data if place["area_name"] == area]
        if area_places:
            area_df = pd.DataFrame(area_places)
            area_csv_path = os.path.join(output_dir, f"제주_{area}_빵집_가게정보.csv")
            area_excel_path = os.path.join(output_dir, f"제주_{area}_빵집_가게정보.xlsx")
            area_json_path = os.path.join(output_dir, f"제주_{area}_빵집_가게정보.json")
            
            area_df.to_csv(area_csv_path, index=False, encoding="utf-8-sig")
            
            # Excel 저장 시 오류 처리
            try:
                area_df.to_excel(area_excel_path, index=False)
            except Exception as e:
                print(f"Excel 저장 중 오류 발생: {e}")
                print(f"CSV 파일로만 저장됩니다: {area_csv_path}")
            
            with open(area_json_path, 'w', encoding='utf-8') as f:
                json.dump(area_places, f, ensure_ascii=False, indent=4)
            
            print(f"제주 {area} 지역 가게정보 파일 저장 완료 (총 {len(area_places)}개)")
    
    # 전체 데이터도 저장
    places_csv_path = os.path.join(output_dir, "제주_전체_빵집_가게정보.csv")
    places_excel_path = os.path.join(output_dir, "제주_전체_빵집_가게정보.xlsx")
    places_json_path = os.path.join(output_dir, "제주_전체_빵집_가게정보.json")

    places_df.to_csv(places_csv_path, index=False, encoding="utf-8-sig")
    
    # Excel 저장 시 오류 처리
    try:
        places_df.to_excel(places_excel_path, index=False)
    except Exception as e:
        print(f"Excel 저장 중 오류 발생: {e}")
        print(f"CSV 파일로만 저장됩니다: {places_csv_path}")
    
    with open(places_json_path, 'w', encoding='utf-8') as f:
        json.dump(cleaned_places_data, f, ensure_ascii=False, indent=4)
    
    print(f"제주 전체 가게정보 파일 저장 완료 (총 {len(cleaned_places_data)}개)")

if reviews_data:
    # 데이터 정제 - Excel 저장을 위해
    cleaned_reviews_data = clean_data_for_excel(reviews_data)
    reviews_df = pd.DataFrame(cleaned_reviews_data)
    
    # 가게별로 리뷰 매핑하고 지역 정보 추가
    for review in cleaned_reviews_data:
        for place in cleaned_places_data:
            if review["place_id"] == place["place_id"]:
                review["area_name"] = place["area_name"]
                break
    
    # 지역별로 리뷰 데이터 분리하여 저장
    for area in set(review.get("area_name", "") for review in cleaned_reviews_data if "area_name" in review):
        area_reviews = [review for review in cleaned_reviews_data if review.get("area_name") == area]
        if area_reviews:
            area_reviews_df = pd.DataFrame(area_reviews)
            area_reviews_csv_path = os.path.join(output_dir, f"제주_{area}_빵집_리뷰.csv")
            area_reviews_excel_path = os.path.join(output_dir, f"제주_{area}_빵집_리뷰.xlsx")
            area_reviews_json_path = os.path.join(output_dir, f"제주_{area}_빵집_리뷰.json")
            
            area_reviews_df.to_csv(area_reviews_csv_path, index=False, encoding="utf-8-sig")
            
            # Excel 저장 시 오류 처리
            try:
                area_reviews_df.to_excel(area_reviews_excel_path, index=False)
            except Exception as e:
                print(f"Excel 저장 중 오류 발생: {e}")
                print(f"CSV 파일로만 저장됩니다: {area_reviews_csv_path}")
            
            with open(area_reviews_json_path, 'w', encoding='utf-8') as f:
                json.dump(area_reviews, f, ensure_ascii=False, indent=4)
            
            print(f"제주 {area} 지역 리뷰 파일 저장 완료 (총 {len(area_reviews)}개)")
    
    # 전체 리뷰 데이터도 저장
    reviews_csv_path = os.path.join(output_dir, "제주_전체_빵집_리뷰.csv")
    reviews_excel_path = os.path.join(output_dir, "제주_전체_빵집_리뷰.xlsx")
    reviews_json_path = os.path.join(output_dir, "제주_전체_빵집_리뷰.json")

    reviews_df.to_csv(reviews_csv_path, index=False, encoding="utf-8-sig")
    
    # Excel 저장 시 오류 처리
    try:
        reviews_df.to_excel(reviews_excel_path, index=False)
    except Exception as e:
        print(f"Excel 저장 중 오류 발생: {e}")
        print(f"CSV 파일로만 저장됩니다: {reviews_csv_path}")
    
    with open(reviews_json_path, 'w', encoding='utf-8') as f:
        json.dump(cleaned_reviews_data, f, ensure_ascii=False, indent=4)
    
    print(f"제주 전체 리뷰 파일 저장 완료 (총 {len(cleaned_reviews_data)}개)")

# 데이터 저장 완료 후 실행 시간 계산
end_time = time.time()
total_time = end_time - start_time
hours, remainder = divmod(total_time, 3600)
minutes, seconds = divmod(remainder, 60)

print("\n데이터 수집이 완료되었습니다.")
print(f"시작 시간: {datetime.fromtimestamp(start_time).strftime('%Y-%m-%d %H:%M:%S')}")
print(f"종료 시간: {datetime.fromtimestamp(end_time).strftime('%Y-%m-%d %H:%M:%S')}")
print(f"총 소요 시간: {int(hours)}시간 {int(minutes)}분 {seconds:.1f}초")
print(f"모든 파일은 '{output_dir}' 디렉토리에 저장되었습니다.")