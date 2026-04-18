from openai import OpenAI
import json
import re


def classify_task(question,model="deepseek-chat",client= OpenAI(api_key="sk-9965b6e2a17547a6affd769eb64306cd", base_url="https://api.deepseek.com")):
    # 对用户输入问题进行任务分类包括基本概念问答、法律法规问答、案例查询与分析、技术方案咨询、资源查找、治理方案生成和其他。
    prompt = f"""
                <用户输入内容>
                {question}
                <任务描述>
                你作为专业的 AI 治理任务分类助手，需依据用户输入内容，精准判断其所属任务类别，类别基于“AI 治理大模型各功能介绍”表，包含基本概念问答、法律法规问答、案例查询与分析、技术方案咨询、资源查找、治理方案生成，若均不匹配则输出“其他” 。在进行仔细分析后，最终分类结果以json格式输出，json格式如下{{"label":"任务类别"}}。最终任务类别只能属于一类。

                <分类判断依据>
                1. **基本概念问答**：围绕人工智能安全治理领域的概念性、理论性问题，如对基本概念、法律法规、伦理规范等理论内容的询问。
                2. **法律法规问答**：围绕人工智能安全治理领域的法律法规、部门规章、标准规范等内容的询问。
                3. **案例查询与分析**：检索人工智能安全治理案例，且需解析案例中治理方式、技术应用、治理路径等实践内容。 
                4. **技术方案咨询**：针对人工智能安全治理具体问题，寻求技术方法、适用场景、原理等技术解决方案相关内容。 
                5. **资源查找**：查找人工智能安全治理相关的数据资源、治理工具资源，包含资源名称、简介等基本信息需求。 
                6. **治理方案生成**：为特定对象（政府、企业等 ）与特定治理场景生成全套人工智能安全治理措施方案。 
                7. **其他**：不属于以上任务类别的任务。

                <示例 1：用户输入>
                “人工智能的基本概念？” 
                ## 分类流程
                判断为围绕概念性问题，属于**基本概念问答** 。
                ## 输出结果
                {{"label":"基本概念问答"}}

                <示例 2：用户输入>
                “找一个金融行业人工智能安全治理案例，分析其治理方式和技术应用” 
                ## 分类流程
                是检索案例并解析实践内容，属于**案例查询与分析** 。
                ## 输出结果
                {{"label":"案例查询与分析"}}

                <示例 3：用户输入>
                “解决人工智能在医疗影像诊断场景的偏见风险，有哪些技术方案，各自原理和优缺点是什么” 
                ## 分类流程
                针对具体问题求技术方案，属于**技术方案咨询** 。
                ## 输出结果
                {{"label":"技术方案咨询"}} 

                <示例 4：用户输入>
                “哪里能找到人工智能安全治理的开源数据集，求推荐资源名称和简介” 
                ## 分类流程
                为查找治理相关数据资源，属于**资源查找** 。
                ## 输出结果
                {{"label":"资源查找"}} 

                <示例 5：用户输入>
                “为零售企业人工智能客服应用，生成包含数据安全、风险防控的安全治理方案” 
                ## 分类流程
                为特定企业场景生成治理措施，属于**治理方案生成** 。
                ## 输出结果
                {{"label":"治理方案生成"}} 

                <示例 6：用户输入>
                “如何上好一门大学AI课程？” 
                ## 分类流程
                与 AI 治理五大类别无关，属于**其他** 。
                ## 输出结果
                {{"label":"其他"}} 
"""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=2048,
        temperature=0.7
    )
    classification = response.choices[0].message.content
    # 使用正则表达式提取JSON格式的标签
    match = re.search(r'\{.*\}', classification)
    if match:
        classification = match.group(0)
    else:
        classification = '{"label":"其他"}'
    # 确保输出格式正确
    try:
        classification_json = json.loads(classification)
        if 'label' not in classification_json:
            classification_json = '{"label":"其他"}'
    except json.JSONDecodeError:
        classification_json = '{"label":"其他"}'
    return classification_json


if __name__ == "__main__":
    question = "如何上好一门大学AI课程？"
    result = classify_task(question)
    print(result)  # Expected output: {"label":"其他"}
    
    question = "人工智能的基本概念？"
    result = classify_task(question)
    print(result)  # Expected output: {"label":"基本概念问答"}
