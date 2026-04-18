from openai import OpenAI
from elasticsearch import Elasticsearch
from elasticsearch.helpers import scan
import json
import re
import Case_Retrieval_and_Analysis
import Governance_Framework_Generation
import knowledge_query
import Task_classification
import resource_search
import Technical_Solution_Consulting
import rag_es


# 初始化Elasticsearch客户端
es = Elasticsearch("http://47.101.62.20:9200")
INDEX_NAME = "qa"  # 替换为你的索引名

# 检查Elasticsearch连接
if not es.ping():
    print("无法连接到Elasticsearch")
    exit(1)

# 设置本地模型API密钥和基础URL
# 设置本地模型API密钥和基础URL
local_api_key = "sk-qwen-8b"
local_base_url = "http://47.101.62.20:1024/v1"
local_model = "Qwen3-8B"



# 设置外部模型API密钥和基础URL
external_api_key = "sk-9965b6e2a17547a6affd769eb64306cd"
external_base_url = "https://api.deepseek.com"
external_model = "deepseek-chat"

# 初始化OpenAI客户端
Client_local = OpenAI(api_key=local_api_key, base_url=local_base_url)
Client_external = OpenAI(api_key=external_api_key, base_url=external_base_url)

# 用户输入
input_question = "根据最新发布的《人工智能生成合成内容标识办法》，从技术层面分析，如何在AI生成内容的传播过程中有效实现显式标识和隐式标识的双重监管机制？请从技术实现的角度详细说明解决方案。"

# 获取用户输入词嵌入
embedding = rag_es.get_embeddings(input_question)
# 任务分类
task_class = Task_classification.classify_task(input_question, model=external_model, client=Client_external)
# 根据任务标签进行检索
if task_class["label"] == "基本概念问答" or task_class["label"] == "法律法规问答":
    rag_task_label = "知识问答"
elif task_class["label"] == "技术方案咨询":
    rag_task_label = "知识问答&技术方案咨询"
else:
    rag_task_label = task_class["label"]
rag_content = rag_es.hybrid_search(rag_task_label, input_question, embedding,top_k=10)
# 只保留rag_content中的content, title, release_time, release_team字段
json_rag_content = [
    {   
        "title": item.get("title"),
        "content": item.get("content"),
        "releaseDate": item.get("releaseDate"),
        "releaseTeam": item.get("releaseTeam"),
        "link": item.get("link")
    }
    for item in rag_content
]
for idx, item in enumerate(json_rag_content, 1):
    item["index"] = idx

# 根据任务分类调用相应的处理函数
if task_class["label"] == "基本概念问答":
    answer = knowledge_query.get_answer_concept(input_question, json_rag_content, model=local_model, client=Client_local)
elif task_class["label"] == "法律法规问答":
    answer = knowledge_query.get_answer_law(input_question, json_rag_content, model=local_model, client=Client_local)
elif task_class["label"] == "案例查询与分析":
    answer = Case_Retrieval_and_Analysis.get_answer(input_question, json_rag_content, model=local_model, client=Client_local)
elif task_class["label"] == "技术方案咨询":
    answer = Technical_Solution_Consulting.get_answer(input_question, json_rag_content, model=local_model, client=Client_local)
elif task_class["label"] == "资源查找":
    answer = resource_search.get_answer(input_question, json_rag_content, model=local_model, client=Client_local)
elif task_class["label"] == "治理方案生成":
    answer = Governance_Framework_Generation.get_answer(input_question, json_rag_content, model=local_model, client=Client_local)
elif task_class["label"] == "其他":
    answer = {"message": "无法识别的任务类型，请检查输入内容。"}
else:
    answer = {"message": "无法识别的任务类型，请检查输入内容。"}
        # 输出答案
        # print("任务分类结果:", task_class)
# 输出答案
print("任务分类结果:", task_class)

# print("检索内容:", json.dumps(json_rag_content, ensure_ascii=False, indent=2))

print("生成的答案:", answer)



