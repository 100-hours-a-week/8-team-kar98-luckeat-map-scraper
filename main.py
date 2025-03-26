#!/usr/bin/env python
# coding: utf-8

import requests
import pandas as pd
import json
import time
from datetime import datetime
import os
from openai import OpenAI

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

# 프로그램 시작 시간 기록
start_time = time.time()

# 여러 검색어와 지역 설정
search_queries = [
    {"query": "제주 빵집", "location": "33.387282,126.542940", "radius": 500000}
    # {"query": "애월 빵집", "location": "33.461624,126.310551", "radius": 50000},
    # {"query": "제주시 빵집", "location": "33.504243,126.519844", "radius": 50000},
    # {"query": "서귀포 빵집", "location": "33.253926,126.559577", "radius": 50000},
    # {"query": "제주 베이커리", "location": "33.499621,126.531188", "radius": 50000},
    # {"query": "제주 과자점", "location": "33.499621,126.531188", "radius": 50000},
    # {"query": "제주 케이크", "location": "33.499621,126.531188", "radius": 100000},
]


collected_count = 0

# 가게 정보를 저장할 리스트
places_data = []
# 리뷰 정보를 저장할 리스트
reviews_data = []

for search_item in search_queries:
    if collected_count >= TOTAL_LIMIT:
        break
        
    search_query = search_item["query"]
    location = search_item["location"]
    radius = search_item["radius"]
    
    print(f"'{search_query}' 검색 시작...")
    next_page_token = None
    page_count = 1
    stop_fetching = False 

    while True:
        try:
            # if next_page_token:
            #     time.sleep(2)  # 페이지 토큰 사용 시 딜레이
            places_result = get_places(search_query, location, radius, API_KEY, language, next_page_token)
            if not places_result or "results" not in places_result:
                print("검색 결과가 없거나 API 오류가 발생했습니다.")
                break

            page_places = places_result["results"]

            for place in page_places:
                # 이미 수집한 장소인지 확인 (중복 제거)
                if any(p["place_id"] == place["place_id"] for p in places_data):
                    continue
                    
                if collected_count >= TOTAL_LIMIT:
                    stop_fetching = True
                    break

                place_id = place["place_id"]
                place_name = place["name"]

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
                    "search_query": search_query
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

                if reviews_texts:
                    # summary = summarize_reviews(reviews_texts, max_reviews=5)
                    # place_info["review_summary"] = summary
                    place_info["review_summary"] = ""
                else:
                    place_info["review_summary"] = ""

                places_data.append(place_info)
                collected_count += 1
                
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
            
    print(f"'{search_query}'에서 총 {len(places_data)}개의 장소 정보를 가져왔습니다.")

print(f"최종적으로 총 {len(places_data)}개의 장소 정보를 가져왔습니다.")

if places_data:
    places_df = pd.DataFrame(places_data)
    places_csv_path = os.path.join(output_dir, "제주_빵집_가게정보.csv")
    places_excel_path = os.path.join(output_dir, "제주_빵집_가게정보.xlsx")
    places_json_path = os.path.join(output_dir, "제주_빵집_가게정보.json")

    places_df.to_csv(places_csv_path, index=False, encoding="utf-8-sig")
    places_df.to_excel(places_excel_path, index=False)
    with open(places_json_path, 'w', encoding='utf-8') as f:
        json.dump(places_data, f, ensure_ascii=False, indent=4)

if reviews_data:
    reviews_df = pd.DataFrame(reviews_data)
    reviews_csv_path = os.path.join(output_dir, "제주_빵집_리뷰.csv")
    reviews_excel_path = os.path.join(output_dir, "제주_빵집_리뷰.xlsx")
    reviews_json_path = os.path.join(output_dir, "제주_빵집_리뷰.json")

    reviews_df.to_csv(reviews_csv_path, index=False, encoding="utf-8-sig")
    reviews_df.to_excel(reviews_excel_path, index=False)
    with open(reviews_json_path, 'w', encoding='utf-8') as f:
        json.dump(reviews_data, f, ensure_ascii=False, indent=4)

# 데이터 저장 완료 후 실행 시간 계산
end_time = time.time()
total_time = end_time - start_time

print("데이터 수집이 완료되었습니다.")
print(f"총 소요 시간: {total_time:.2f}초 ({total_time/60:.1f}분)")
print(f"모든 파일은 '{output_dir}' 디렉토리에 저장되었습니다.")