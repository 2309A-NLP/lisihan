"""
RAG系统性能评测模块
====================
这个模块提供RAG系统的质量评估功能，包括：
- 使用RAGAS框架进行自动化评测
- 计算多个质量指标（上下文精度、召回率、忠实度等）
- 从知识库生成测试用例

评测指标说明：
- ContextPrecision: 上下文精度 - 检索到的内容是否相关
- ContextRecall: 上下文召回率 - 是否检索到所有必要信息
- ContextRelevance: 上下文相关性 - 内容与问题的相关程度
- Faithfulness: 忠实度 - 回答是否基于检索到的内容
- AnswerRelevancy: 答案相关性 - 回答是否针对问题

RAGAS (RAG Assessment) 是一个专门用于评估RAG系统的框架
"""

import logging

# ========== 导入 RAGAS 评测框架（带降级处理）==========
try:
    # 尝试导入 ragas 评测库
    # ragas 是一个专门评估 RAG 系统的 Python 库
    from ragas import evaluate  # 评估函数，执行评测流程
    from ragas.metrics import (
        ContextPrecision,  # 上下文精度：检索内容中有多少是真正需要的
        ContextRecall,  # 上下文召回率：需要的信息有多少被检索到
        _ContextRelevance,  # 上下文相关性：检索内容与问题的相关程度
        Faithfulness,  # 忠实度：回答是否基于检索到的内容（不胡编）
        AnswerRelevancy  # 答案相关性：回答是否针对问题（不跑题）
    )
except Exception:
    # 如果 ragas 未安装，将这些变量都设为 None
    # 这样代码可以继续运行，只是评测功能不可用
    evaluate = None
    ContextPrecision = ContextRecall = _ContextRelevance = Faithfulness = AnswerRelevancy = None

# 导入 RAG 系统实例
from app.core.rag import rag_system  # 核心RAG系统，提供检索和生成功能

# 导入类型注解
from typing import List, Dict  # List: 列表类型, Dict: 字典类型

logger = logging.getLogger(__name__)

class RAGEvaluation:
    """
    RAG 系统性能评测类

    功能：
    - 使用标准测试用例评估 RAG 系统
    - 计算多个质量指标
    - 自动生成测试用例

    使用场景：
    - 测试新知识库的质量
    - 对比不同配置的 RAG 系统
    - 持续集成中的自动测试
    - 系统优化前后的效果对比

    评测流程：
    1. 准备测试用例（问题 + 标准答案）
    2. 对每个问题执行 RAG 检索和生成
    3. 使用 RAGAS 计算各种指标
    4. 返回量化评分
    """

    def __init__(self):
        """
        初始化评测模块

        功能：
        - 创建评测指标实例
        - 如果 ragas 未安装，指标列表为空

        注意：
        - ragas 需要额外安装：pip install ragas
        - 如果不安装，评测功能会降级为空操作
        """
        # 初始化评测指标
        if evaluate:
            # 如果 ragas 可用，创建5个指标实例
            # 这些指标会在 evaluate() 时使用
            self.metrics = [
                ContextPrecision(),  # 上下文精度指标
                ContextRecall(),  # 上下文召回率指标
                _ContextRelevance(),  # 上下文相关性指标
                Faithfulness(),  # 忠实度指标
                AnswerRelevancy()  # 答案相关性指标
            ]
        else:
            # ragas 不可用，指标列表为空
            self.metrics = []

    def evaluate(self, test_cases: List[Dict]) -> Dict:
        """
        评测 RAG 系统性能

        这是核心评测方法，对 RAG 系统进行全面评估

        评测流程详解：
        1. 遍历每个测试用例
        2. 使用问题检索知识库，获取相关上下文
        3. 基于上下文生成回答
        4. 收集所有数据（问题、答案、上下文、标准答案）
        5. 调用 RAGAS 计算各项指标
        6. 返回评估结果

        Args:
            test_cases: 测试用例列表，每个测试用例包含：
                - question: 问题（必填）
                - ground_truth: 标准答案（必填）
                - context: 上下文（可选，如果不提供会自动检索）

        Returns:
            dict: 评测结果，包含各项指标分数
                  例如：{
                      "context_precision": 0.85,
                      "context_recall": 0.92,
                      "faithfulness": 0.88,
                      ...
                  }
        """
        try:
            # ========== 检查依赖 ==========
            # 如果 ragas 未安装，无法进行评测
            if evaluate is None:
                return {"error": "未安装 ragas，无法执行 RAGAS 评测"}

            print(f"开始评测 RAG 系统，测试用例数量: {len(test_cases)}")

            # ========== 准备评测数据 ==========
            # 创建四个列表，分别存储不同类型的数据
            questions = []  # 用户问题列表
            ground_truths = []  # 标准答案列表
            contexts = []  # 检索到的上下文列表
            answers = []  # RAG生成的答案列表

            # ========== 处理每个测试用例 ==========
            for test_case in test_cases:
                # 提取测试用例中的问题和标准答案
                question = test_case.get('question')
                ground_truth = test_case.get('ground_truth')

                # 验证必要字段
                if not question or not ground_truth:
                    # 跳过缺少问题或标准答案的测试用例
                    continue

                # ========== 步骤1: 检索知识库 ==========
                # 使用 RAG 系统搜索相关知识
                # top_k=3 表示返回最相关的3个文档块
                context_results = rag_system.search_knowledge(question, top_k=3)

                # 提取检索结果的文本内容
                # 例如: context_results = [
                #     {"content": "民法典第1条内容...", "score": 0.95},
                #     {"content": "民法典第2条内容...", "score": 0.87},
                # ]
                # 转换为: ["民法典第1条内容...", "民法典第2条内容..."]
                context = [result['content'] for result in context_results]

                # ========== 步骤2: 生成回答 ==========
                # 基于问题和检索到的上下文生成答案
                # 参数说明：
                # - question: 用户问题
                # - context: 检索到的相关文本
                # - "" : 角色模板（空字符串表示使用默认）
                answer = rag_system.generate_response(question, context, "")

                # ========== 步骤3: 收集数据 ==========
                questions.append(question)
                # RAGAS 要求 ground_truth 是列表格式
                # 转换为 ["标准答案文本"]
                ground_truths.append([ground_truth])
                contexts.append(context)
                answers.append(answer)

                # 打印进度信息（用于调试）
                print(f"处理测试用例: {question[:50]}...")  # 只显示前50字符
                print(f"生成回答: {answer[:100]}...")  # 只显示前100字符

            # ========== 验证数据有效性 ==========
            if not questions:
                return {"error": "没有有效的测试用例"}

            # ========== 步骤4: 构建评测数据集 ==========
            # 按照 RAGAS 要求的格式组织数据
            eval_data = {
                "question": questions,  # 问题列表
                "answer": answers,  # RAG生成的答案列表
                "contexts": contexts,  # 检索到的上下文列表
                "ground_truth": ground_truths  # 标准答案列表
            }

            # ========== 步骤5: 执行 RAGAS 评测 ==========
            print("开始执行 RAGAS 评测...")

            # evaluate 函数会：
            # 1. 对每个测试用例计算各项指标
            # 2. 汇总所有指标的平均值
            # 3. 返回包含评分的数据集对象
            result = evaluate(
                eval_data,  # 评测数据
                metrics=self.metrics  # 使用的指标列表
            )

            # ========== 步骤6: 转换结果格式 ==========
            # 将 RAGAS 的结果对象转换为普通字典
            # 这样更容易序列化为 JSON 返回给 API
            result_dict = result.to_dict()
            print(f"评测完成，结果: {result_dict}")

            return result_dict

        except Exception as e:
            # ========== 错误处理 ==========
            print(f"评测失败: {e}")
            logger.exception("RAG 评测失败: error=%s", e)
            return {"error": str(e)}

    def generate_test_cases(self, knowledge_base: List[str], count: int = 10) -> List[Dict]:
        """
        从知识库自动生成测试用例

        功能：
        - 基于现有知识库内容自动生成测试用例
        - 用于快速创建评测数据集

        生成策略：
        - 目前使用简单规则（提取内容作为答案，构造通用问题）
        - 实际应用中可以使用 LLM 生成更真实的问题

        改进方向：
        - 使用 LLM 根据内容生成自然问题
        - 生成不同类型的问题（事实性、推理性、比较性）
        - 自动评估生成质量

        Args:
            knowledge_base: 知识库内容列表
                每个元素是一段知识文本
            count: 要生成的测试用例数量（默认10个）

        Returns:
            list: 测试用例列表，每个包含：
                - question: 生成的问题
                - ground_truth: 标准答案（即原知识内容）

        示例：
            knowledge_base = [
                "民法典第1043条规定夫妻应当互相忠实",
                "公司注册需要提供营业执照",
                ...
            ]

            test_cases = generator.generate_test_cases(knowledge_base, count=5)
            # 返回: [
            #     {
            #         "question": "关于这部分内容：民法典第1043条规定...，请解释一下",
            #         "ground_truth": "民法典第1043条规定夫妻应当互相忠实"
            #     },
            #     ...
            # ]
        """
        try:
            print(f"从知识库生成 {count} 个测试用例...")

            test_cases = []

            # ========== 简单的测试用例生成逻辑 ==========
            # 遍历知识库内容，直到生成足够数量的测试用例
            # 注意：目前只取前 count 条，实际应该随机采样或按重要性选择
            for i, content in enumerate(knowledge_base[:count]):
                # 生成问题：使用内容前缀 + 通用问题模板
                # 这是一个非常简单的策略，实际效果可能不理想
                # 更好的做法：使用 LLM 生成相关问题
                question = f"关于这部分内容：{content[:100]}...，请解释一下"

                # 标准答案：直接使用原内容
                ground_truth = content

                # 添加到测试用例列表
                test_cases.append({
                    "question": question,
                    "ground_truth": ground_truth
                })

            print(f"生成测试用例完成，数量: {len(test_cases)}")
            return test_cases

        except Exception as e:
            print(f"生成测试用例失败: {e}")
            return []
