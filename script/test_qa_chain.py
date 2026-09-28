from app.rag.chains import build_qa_chain

qa_chain = build_qa_chain()

question_1 = "什么是南瓜书？"
question_2 = "Prompt Engineering for Developer是谁写的？"

result = qa_chain.invoke(question_1)
print("大模型+知识库后回答 question_1 的结果：")
print(result)