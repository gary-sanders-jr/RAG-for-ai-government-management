from openai import OpenAI



def get_answer(question,retrieved_content,model="deepseek-chat",client= OpenAI(api_key="sk-9965b6e2a17547a6affd769eb64306cd", base_url="https://api.deepseek.com")):
    # 查询用户所需资源。
    prompt = f"""
            <检索内容>
            {retrieved_content}
            <用户需要查询的资源>
            {question}
            <任务描述>
            你是一个人工智能治理咨询智能助手，你的任务是根据检索内容向用户提供其询问的AI治理相关的资源信息，回答结构参考如下。
            <回答框架>

            ## 1.资源名称
            - **发文机关**：
            - **施行时间/发布时间**：
            - **原文链接**：[File_name](链接link)
            - **主要内容**

            ## 2.资源名称
            - **发文机关**：
            - **施行时间/发布时间**：
            - **原文链接**：[File_name](链接link)
            - **主要内容**

            你需要将最终引用了的检索内容的基本信息通过json列表的形式<REF>[{{"index":"","title":"","releaseDate": "","releaseTeam":"","link": ""}},{{"index":"","title":"","releaseDate": "","releaseTeam":"","link": ""}}]</REF>放在最后，如果没有引用则不需要将其信息放在最后。


            <示例>
            
            1. **资源名称**：nvidia_Aegis-AI-Content-Safety-Dataset-2.0
            - **发布团队**：nvidia
            - **发布时间**：2025年6月
            - **资源介绍**：NVIDIA 发布的Aegis-AI-Content-Safety-Dataset-2.0是一款聚焦内容安全与大型语言模型对齐的开源数据集，旨在为 AI 内容审查、毒性检测等研究提供高质量标注数据。该数据集含约 33,416 条交互记录，原始以 JSON 格式存储，后转为 Parquet 格式以提升处理效率，适用于文本分类任务，尤其专注于大语言模型的安全审查、毒性检测及模型对齐。其采用 NVIDIA 精细化分类体系，涵盖 “非不安全类别”“核心不安全类别” 及 “细粒度不安全类别” 三个层级，精准区分安全内容、高风险内容（如仇恨言论、暴力教唆等）及更细分的危险场景（如非法活动、虚假信息等）。数据来源混合了 Mistral-7B-v0.1 生成的响应、Anthropic 人类偏好数据及 GPT-4 生成的拒绝数据，标注由专业人员通过 Label Studio 完成，辅以严格的复核与培训机制，同时注重标注人员的心理健康保护。该数据集以 CC-BY-4.0 许可证发布，支持内容安全、AI 审查等领域的研究与开发，为模型安全性优化提供了可靠基准。

            “Nemotron Content Safety Dataset V2” 聚焦大模型内容安全领域，可用于追溯模型错误决策的责任主体（开发者、训练数据或部署场景）。该数据集采用全面灵活的安全风险分类体系，涵盖 12 个顶级危害类别及 9 个细分子类别。顶级危害类别既包括 “Safe”（即安全）、“Needs Caution”（即需谨慎）等非不安全类别，也包含 “仇恨 / 身份仇恨”“性相关内容”“自杀与自我伤害” 等核心不安全类别；细分子类别则有 “非法活动”“不道德 / 不伦理行为” 等。这种细致分类能精准识别不同安全风险，为追溯责任主体提供有力依据。该数据集由 33,416 条人类与大语言模型的标注交互组成，分为 30,007 条训练样本、1,445 条验证样本和 1,964 条测试样本。其构建采用混合数据生成管道，结合全对话级别的人工标注与多 LLM “评审团” 系统评估回复安全性。训练数据来源广泛，包括从 Anthropic RLHF、Do-Anything-Now DAN 和 AI-assisted Red-Teaming 等数据集收集的人类编写提示，由 Mistral-7B-v0.1 生成的 LLM 回复，以及通过 Mixtral-8x22B-v0.1、Mistral-NeMo-12B-Instruct 和 Gemma-2-27B-it 这 3 个 LLM 集成生成的回复安全标签。此外，还混入使用 Gemma-2-27B 通过自定义提示工程偏转策略生成的拒绝数据，以及通过 Mixtral-8x7B-v0.1 生成的主题跟随数据进行增强。
            
            使用该数据集时，需严格遵守 CC-BY-4.0 许可协议。由于数据包含可能令人反感的内容 —— 如歧视性语言、虐待、暴力、自我伤害等讨论，使用者应依自身风险承受能力谨慎处理，仅用于旨在提升模型安全性的研究。尽管构建过程经过严格质量把控，包括定期随机抽样重评估和标注人员培训，但面对复杂实际场景仍可能存在局限。例如，处理分类体系未明确涵盖的新型安全风险时，可能无法准确分类，进而影响责任主体追溯。使用时可结合人工审核与持续模型优化，提升适用性与准确性。该数据集为大模型算法问责治理提供重要参考，助力深入分析错误决策来源、明确责任主体，推动大模型安全发展。
                
            - **资源链接**：[Aegis-AI-Content-Safety-Dataset-2.0] (https://huggingface.co/datasets/nvidia/Aegis-AI-Content-Safety-Dataset-2.0)

            2. **资源名称**：galileo-ai_ragbench
            - **发布团队**：galileo-ai
            - **发布时间**：2024年6月
            - **资源介绍**:Galileo-ai_ragbench 是一款专注于检索增强生成（RAG）任务评估的开源数据集，旨在为 RAG 系统的性能评估、优化及相关研究提供全面且高质量的标注数据。该数据集规模庞大，涵盖约 100k 条 RAG 示例，适用于多种 RAG 任务类型，尤其为工业应用场景下的 RAG 系统开发提供了极具针对性的参考。Galileo-ai_ragbench 由 12 个子组件数据集构成，每个子数据集都包含训练集、验证集和测试集，以满足不同阶段的模型训练与评估需求。这些子数据集覆盖了五个独特的行业特定领域，其示例来源于行业语料库，如用户手册等，使得数据集在工业应用中具有高度的相关性和实用性。数据集的每个示例包含丰富的特征信息，如问题、相关文档、生成的响应等，同时还提供了多种评估指标，如 Trulens 的 groundedness 和 context relevance、Ragas 的 faithfulness 和 context relevance 等，这些指标有助于全面评估 RAG 系统在不同方面的性能。此外，数据集还记录了句子支持信息、不支持的响应句子键、相关性解释等详细信息，为深入分析模型的表现提供了有力支持。Galileo-ai_ragbench 以 CC-BY-4.0 许可证发布，方便研究人员和开发者在遵循许可证条款的前提下自由使用和分享。它支持多种数据加载方式，可通过 Hugging Face 的datasets库轻松加载，无论是加载单个子数据集的特定划分，还是加载整个数据集，都十分便捷。这为 RAG 领域的研究与开发提供了可靠的基准，有助于推动 RAG 技术在各个行业的应用和发展。
                
            “Galileo-ai_ragbench” 聚焦检索增强生成（RAG）系统的性能评估与责任追溯，可用于定位大模型在检索 - 生成流程中错误决策的责任主体（开发者、训练数据或部署场景）。该数据集采用多维度的任务与评估体系，涵盖 12 个子数据集（如 covidqa、cuad、finqa 等），覆盖医疗、法律、金融等五个行业领域，每个子数据集均包含训练、验证、测试三个分裂，且每条样本包含丰富特征：既包括 “question”（用户问题）、“documents”（检索文档）、“response”（模型生成内容）等核心交互信息，也包含 “generation_model_name”（生成模型名称）、“annotating_model_name”（标注模型名称）等溯源字段，以及 “trulens_groundedness”（事实一致性）、“ragas_context_relevance”（上下文相关性）等 10 余种评估指标，这种细致的结构设计能精准拆解 RAG 系统在 “检索不足”“生成失实” 等环节的问题，为责任追溯提供清晰依据。
            
            该数据集包含约 100k 条 RAG 示例，12 个子数据集各有明确的领域定位（如 covidqa 聚焦医疗问答、cuad 专注法律文档理解），数据来源以行业语料库（如用户手册、专业文献）为核心，确保与实际应用场景高度贴合。构建过程中，通过标准化的特征设计记录完整链路信息：从 “documents_sentences”（文档句子拆分）到 “sentence_support_information”（句子级支持关系），再到 “unsupported_response_sentence_keys”（无依据的生成内容标记），完整保留检索质量、生成逻辑与标注过程的关联数据，可直接用于分析错误是源于 “训练数据中领域知识缺失”（如金融领域文档覆盖不足）、“开发者对检索结果利用率低”（utilization_score 偏低），还是 “部署场景与训练领域不匹配”（如用通用模型处理专业法律问题）。
            
            使用该数据集时，需严格遵守 CC-BY-4.0 许可协议，由于覆盖领域广泛（从医疗到技术文档），部分子数据集可能存在领域特异性知识偏差（如法律术语标注差异），使用者应结合具体应用场景进行适配性校验。尽管数据集通过多指标评估（如 gpt35_utilization、completeness_score）强化了责任追溯能力，但面对跨领域复杂任务时，仍可能存在评估指标泛化不足的问题，建议搭配人工审核补充判断。该数据集为 RAG 系统的算法问责提供了标准化基准，助力精准定位错误环节、明确责任主体，推动检索增强生成技术在行业场景中的可靠应用。

            - **资源链接**：[galileo-ai_ragbench] (https://huggingface.co/datasets/galileo-ai/ragbench)
            <REF>[{{
                "index": "1",
                "title": "nvidia_Aegis-AI-Content-Safety-Dataset-2.0",
                "releaseDate": "2025.06.10",
                "releaseTeam": "nvidia",
                "link": "https://huggingface.co/datasets/nvidia/Aegis-AI-Content-Safety-Dataset-2.0"
            }},
            {{
                "index": "2",
                "title": "galileo-ai_ragbench",
                "releaseDate": "2024.06.14",
                "releaseTeam": "galileo-ai",
                "link": "https://huggingface.co/datasets/galileo-ai/ragbench"
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
    answer = response.choices[0].message['content']

    return answer

if __name__ == "__main__":
    question = "我目前正在研究大模型的数据安全治理，请帮我推荐一些适用于数据安全研究的数据集资源。"
    result = get_answer(question)
    print(result)  