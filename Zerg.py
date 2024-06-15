import asyncio
from langchain.prompts import PromptTemplate
from langchain.llms import Ollama
from langchain.chains import LLMChain
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler

# Ollama 모델 초기화
llama3 = Ollama(model="llama3")

# 질문 템플릿 형식 정의
overmind_template = "오버마인드: {question}\n오버마인드의 대답:"
daggoth_template = "다고스: {question}\n다고스의 대답:"
zasz_template = "자츠: {question}\n자츠의 대답:"

# 템플릿 완성
overmind_prompt = PromptTemplate.from_template(overmind_template)
daggoth_prompt = PromptTemplate.from_template(daggoth_template)
zasz_prompt = PromptTemplate.from_template(zasz_template)

# 연결된 체인(Chain) 객체 생성
overmind_chain = LLMChain(llm=llama3, prompt=overmind_prompt, callbacks=[StreamingStdOutCallbackHandler()])
daggoth_chain = LLMChain(llm=llama3, prompt=daggoth_prompt, callbacks=[StreamingStdOutCallbackHandler()])
zasz_chain = LLMChain(llm=llama3, prompt=zasz_prompt, callbacks=[StreamingStdOutCallbackHandler()])

# 비동기 함수 정의
async def process_input(character_chain, character_name, question):
    input_data = {"question": question}
    response = await character_chain.invoke(input_data)
    print(f'{character_name}: {response.strip()}')

# 입력 질문 리스트
questions = [
    "프로토스와의 전투 전략은 무엇인가?",
    "테란의 방어선을 뚫는 방법은 무엇인가?",
    "저그 군단의 다음 진화 방향은 무엇인가?",
]

# 비동기 이벤트 루프 실행
async def main():
    tasks = []
    for question in questions:
        tasks.append(process_input(overmind_chain, "오버마인드", question))
        tasks.append(process_input(daggoth_chain, "다고스", question))
        tasks.append(process_input(zasz_chain, "자츠", question))
    await asyncio.gather(*tasks)

# 메인 함수 실행
if __name__ == "__main__":
    asyncio.run(main())
