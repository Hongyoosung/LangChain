import os
from langchain.prompts import PromptTemplate
from langchain.llms import Ollama
from langchain.chains import LLMChain

# Ollama 모델 초기화
llama3 = Ollama(model="llama3")

# 질문 템플릿 형식 정의
template = "{area1}와 {area2}의 시차는 몇시간이야?"

# 템플릿 완성
prompt = PromptTemplate.from_template(template)

# 연결된 체인(Chain) 객체 생성
chain = LLMChain(llm=llama3, prompt=prompt)

# 입력 데이터 리스트
input_list = [
    {"area1": "파리", "area2": "뉴욕"},
    {"area1": "서울", "area2": "하와이"},
    {"area1": "캔버라", "area2": "베이징"},
]

# 반복문으로 결과 출력
for input_data in input_list:
    response = chain.invoke(input_data)
    print(f'[답변]: {response.strip()}')
