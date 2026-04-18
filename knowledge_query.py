from openai import OpenAI


def get_answer_concept(question,retrieved_content,model="deepseek-chat",client= OpenAI(api_key="sk-9965b6e2a17547a6affd769eb64306cd", base_url="https://api.deepseek.com")):
    # 回答用户提出的基本概念。
    prompt = f"""
            <检索内容>
            {retrieved_content}
            <用户询问的基本概念>
            {question}
            <任务描述>
            你是一个人工智能治理咨询智能助手，你的任务是根据检索内容向用户解释其询问的基本概念。
            你生成的内容引用了哪些检索内容，必须要在生成的内容后面通过[^index^]标记出来，index表示文档在检索结果中的位置，index从1开始，并且将最终引用了的检索内容的基本信息通过json列表的形式<REF>[{{"index":"","title":"","releaseDate": "","releaseTeam":"","link": ""}},{{"index":"","title":"","releaseDate": "","releaseTeam":"","link": ""}}]</REF>放在最后，如果没有引用则不需要将其信息放在最后。

            示例：

            大模型(Large Model)是近年来人工智能领域最具革命性的技术突破之一，它正在重塑各行各业并深刻影响着人类社会的发展进程。根据最新研究资料，大模型通常是指由人工神经网络构建的、具有​​超大规模参数​​(通常达到百亿、千亿甚至万亿级别)和​​复杂计算结构​​的机器学习模型。[^2^]这些模型通过在海量数据上进行预训练，展现出强大的通用任务解决能力、人类指令遵循能力和复杂推理能力，成为推动新一代人工智能发展的新型基础设施。[^1^]

            <REF>[{{   "index":"1",
                "title": "大模型技术与应用",
                "releaseDate": "2019.09",
                "releaseTeam": "华中科技大学出版社",
                "link": ""
            }},{{   "index":"2",
                "title": "人工智能 服务能力成熟度评估",
                "releaseDate": "2025.06.30",
                "releaseTeam": "国家市场监督管理总局&国家标准化管理委员会",
                "link": ""
            }}]</REF>
"""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=8192,
        temperature=0.7
    )
    answer = response.choices[0].message.content

    return answer

def get_answer_law(question,retrieved_content,model="deepseek-chat",client= OpenAI(api_key="sk-9965b6e2a17547a6affd769eb64306cd", base_url="https://api.deepseek.com")):
    # 回答用户提出的法律法规。
    prompt = f"""
            <检索内容>
            {retrieved_content}
            <用户询问的法律法规>
            {question}
            <任务描述>
            <任务描述>
            你是一个人工智能治理咨询智能助手，你的任务是根据检索内容向用户提供其询问的法律法规、行政法规、地方性法规、部门规章、标准规范或伦理规范。
            如果是法律法规、行政法规、地方性法规、部门规章，回答的结构如下(注意原文链接处严格使用Markdown格式，使得链接能超链接到File_name中)：


            ## 1. 法律法规名称
            - **发文机关**：
            - **施行时间/发布时间**：
            - **原文链接**：[File_name](链接link)
            ### 主要条款
            - **第几条**  

            ## 2. 法律法规名称
            - **发文机关**：
            - **施行时间/发布时间**：
            - **原文链接**：[File_name](链接link)
            ### 主要条款
            - **第几条**  

            如果是标准规范或伦理规范，回答的结构如下：
            ## 1. 规范名称
            - **发文机关**：
            - **发布时间**：
            - **原文链接**：[File_name](链接link)
            ### 主要条款
            - **第几条**  

            ## 2. 法律法规名称
            - **发文机关**：
            - **发布时间**：
            - **原文链接**：[File_name](链接link)
            ### 主要内容
            
            <其它要求>
            将最终引用了的检索内容的基本信息通过json列表的形式<REF>[{{"index":"","title":"","releaseDate": "","releaseTeam":"","link": ""}},{{"index":"","title":"","releaseDate": "","releaseTeam":"","link": ""}}]</REF>
            放在最后，如果没有引用则不需要将其信息放在最后。
            
            <示例>
            中国已出台的包含个人隐私保护法律法规如下：
            ## 1. 《中华人民共和国数据安全法》

            - **发文机关**：全国人民代表大会常务委员会  
            - **施行日期**：2021年9月1日  
            - **原文链接**：[中华人民共和国数据安全法_中国人大网]( http://www.npc.gov.cn/npc/c2/c30834/202106/t20210610_311888.html)

            ### 主要条款

            - **第八条**  开展数据处理活动，应当遵守法律、法规，尊重社会公德和伦理，遵守商业道德和职业道德，诚实守信，履行数据安全保护义务，承担社会责任，不得危害国家安全、公共利益，不得损害个人、组织的合法权益。

            - **第二十一条**  国家建立数据分类分级保护制度，根据数据在经济社会发展中的重要程度，以及一旦遭到篡改、破坏、泄露或者非法获取、非法利用，对国家安全、公共利益或者个人、组织合法权益造成的危害程度，对数据实行分类分级保护。国家数据安全工作协调机制统筹协调有关部门制定重要数据目录，加强对重要数据的保护。

            ## 2. 《上海市数据条例》
            - **发文机关**：上海市人民代表大会常务委员会  
            - **施行时间**：2022年1月1日  
            - **原文链接**：[上海市数据条例](https://www.shanghai.gov.cn/nw12344/20211129/a1a38c3dfe8b4f8f8fcba5e79fbe9251.html)
            ### 主要条款
            - **第十六条**  市、区人民政府及其有关部门可以依法要求相关自然人、法人和非法人组织提供突发事件处置工作所必需的数据。
            要求自然人、法人和非法人组织提供数据的，应当在其履行法定职责的范围内依照法定的条件和程序进行，并明确数据使用的目的、范围、方式、期限。收集的数据不得用于与突发事件处置工作无关的事项。对在履行职责中知悉的个人隐私、个人信息、商业秘密、保密商务信息等应当依法予以保密，不得泄露或者非法向他人提供。

            - **第十八条** 除法律、行政法规另有规定外，处理个人信息的，应当取得个人同意。个人信息的处理目的、处理方式和处理的个人信息种类发生变更的，应当重新取得个人同意。
            处理个人自行公开或者其他已经合法公开的个人信息，应当依法在合理的范围内进行；个人明确拒绝的除外。处理已公开的个人信息，对个人权益有重大影响的，应当依法取得个人同意。

            <REF>[{{"index":"1","title":"中华人民共和国数据安全法","releaseDate":"2021.09.01","releaseTeam":"全国人民代表大会常务委员会","link":"https://flk.npc.gov.cn/detail2.html?ZmY4MDgxODE3OWY1ZTA4MDAxNzlmODg1YzdlNzAzOTI%3D"}},{{"index":"2","title":"上海市数据条例","releaseDate":"2022.01.01","releaseTeam":"上海市人民代表大会常务委员会","link":"https://flk.npc.gov.cn/detail2.html?ZmY4MDgxODE3ZjQyMGFjODAxN2Y0Zjc5OTY0ZDA2N2Q%3D"}}]</REF>
"""
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "user", "content": prompt}
        ],
        max_tokens=8192,
        temperature=0.7
    )
    answer = response.choices[0].message.content
    return answer


if __name__ == "__main__":
    question = "人工智能的基本概念？"
    result = get_answer_concept(question)
    print(result)

    question = "人工智能相关标准规范有哪些？"
    result = get_answer_law(question)
    print(result)  