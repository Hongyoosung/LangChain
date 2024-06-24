from langchain.prompts import PromptTemplate
from langchain.llms import Ollama
from langchain.chains import LLMChain

# Ollama 모델 초기화
llama3 = Ollama(model="llama3")

# 질문 템플릿 형식 정의
template = "{topic} 에 대해 쉽게 설명해주세요."

# 템플릿 완성
prompt_template = PromptTemplate.from_template(template)

# LLMChain 객체 생성 (스트리밍 콜백 포함)
chain = LLMChain(llm=llama3, prompt=prompt_template)

# 입력 데이터를 처리하는 함수 정의
def process_input(character_chain, topic):
    input_data = {"topic": topic}
    response = character_chain(input_data)
    print(response)

# 메인 함수 실행
if __name__ == "__main__":
    process_input(chain, "인공지능 모델의 학습 원리")
