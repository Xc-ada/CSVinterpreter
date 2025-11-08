import re
import time
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from colorama import Fore, Style

from .llm_client import LLMClient
from .csv_loader import CSVDataLoader
from .code_executor import CodeExecutor, ExecutionResult

@dataclass
class SubQuery:
    """子问题"""
    id: str
    text: str
    priority: int 
    order: int = 0
    dependencies: List[str] = field(default_factory=list)


@dataclass
class StepResult:
    """单步分析结果"""
    subquery: SubQuery
    code: str
    data: Any
    insights: str
    success: bool
    execution_time: float = 0.0
    output: Optional[str] = None


@dataclass
class AnalysisResult:
    """完整分析结果"""
    history: List[StepResult]
    results: Dict[str, Any]
    reasoning: List[str]
    total_time: float = 0.0


class QueryDecomposer:
    """将复杂问题拆解为多个可执行的子问题"""
    
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
    
    def decompose(self, question: str, data_context: Dict[str, Any]) -> List[SubQuery]:
        """
        拆解问题为多个子问题
        
        Args:
            question: 用户原始问题
            data_context: 数据上下文
        
        Returns:
            List[SubQuery]: 子问题列表
        """
        print(f"正在分析问题结构...")
        
        prompt = self._build_decompose_prompt(question, data_context)
        
        try:
            response = self.llm._call_api(
                messages=[
                    {"role": "system", "content": "你是一个数据分析专家，擅长将复杂问题拆解为简单步骤。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
            
            # 解析子问题
            sub_queries = self._parse_subqueries(response)
            
            return sub_queries
            
        except Exception as e:
            print(f"{Fore.RED}⚠️ 问题分解失败: {e}")
            # 返回单个问题作为fallback
            return [SubQuery(id="sq_1", text=question, priority=1, order=1)]
    
    def _build_decompose_prompt(self, question: str, data_context: Dict) -> str:
        """构建问题分解的Prompt"""
        
        prompt = f"""
你需要将一个复杂的数据分析问题拆解为多个简单的、可以依次执行的子问题。

原始问题：
{question}

数据信息：
- 列名: {data_context.get('columns', [])}
- 数据维度: {data_context.get('shape', 'unknown')}
- 数据样例: {str(data_context.get('sample_data', []))[:300]}...

拆解要求：
1. 子问题应该有逻辑顺序（后面的可以依赖前面的结果）
2. 每个子问题应该清晰、具体、可用Python代码实现
3. 数量控制在3-6个
4. 按重要性分为：必需(priority=1)、重要(priority=2)、补充(priority=3)

输出格式（严格遵守）：
<subquery id="sq_1" priority="1">具体的子问题描述</subquery>
<subquery id="sq_2" priority="1">具体的子问题描述</subquery>
<subquery id="sq_3" priority="2">具体的子问题描述</subquery>

示例（针对"分析2017年销售情况并对比其他年份"）：
<subquery id="sq_1" priority="1">计算2017年的总销售额和基本统计</subquery>
<subquery id="sq_2" priority="1">计算2015年和2016年的总销售额</subquery>
<subquery id="sq_3" priority="2">对比三年的销售额增长趋势</subquery>
<subquery id="sq_4" priority="2">分析2017年各类别的销售构成</subquery>

现在请拆解上述问题：
"""
        return prompt
    
    def _parse_subqueries(self, response: str) -> List[SubQuery]:
        """解析LLM返回的子问题"""
        
        pattern = r'<subquery\s+id="([^"]+)"\s+priority="(\d+)">([^<]+)</subquery>'
        matches = re.findall(pattern, response)
        
        sub_queries = []
        for order, (sq_id, priority, text) in enumerate(matches, 1):
            sub_queries.append(SubQuery(
                id=sq_id,
                text=text.strip(),
                priority=int(priority),
                order=order
            ))
        
        if not sub_queries:
            lines = [line.strip() for line in response.split('\n') 
                    if line.strip() and not line.strip().startswith('<')]
            for i, line in enumerate(lines[:6], 1):
                if line:
                    sub_queries.append(SubQuery(
                        id=f"sq_{i}",
                        text=line,
                        priority=1 if i <= 3 else 2,
                        order=i
                    ))
        
        return sub_queries


class IterativeAnalysisEngine:
    """管理迭代分析过程"""
    
    def __init__(self, data_loader: CSVDataLoader, 
                 code_executor: CodeExecutor,
                 llm_client: LLMClient,
                 conversation_history: List[Dict] = None):
        self.data_loader = data_loader
        self.executor = code_executor
        self.llm = llm_client
        self.conversation_history = conversation_history or []
        
        self.analysis_history: List[StepResult] = []
        self.intermediate_results: Dict[str, Any] = {}
        self.reasoning_chain: List[str] = []
    
    def analyze(self, sub_queries: List[SubQuery], 
                max_retries: int = 5) -> AnalysisResult:
        """
        迭代执行子问题分析
        
        Args:
            sub_queries: 子问题列表
            max_retries: 每步最大重试次数
        
        Returns:
            AnalysisResult: 完整分析结果
        """
        start_time = time.time()
        
        sorted_queries = sorted(sub_queries, key=lambda x: (x.priority, x.order))
        
        for i, sub_query in enumerate(sorted_queries, 1):
            print(f"第 {i}/{len(sorted_queries)} 步分析")
            
            print(f" {sub_query.text}\n")
            
            step_result = self._execute_step_with_retry(
                sub_query, i, max_retries
            )
 
            self.analysis_history.append(step_result)
            if step_result.success:
                self.intermediate_results[sub_query.id] = step_result.data

            self._update_reasoning(sub_query, step_result)
        
        total_time = time.time() - start_time
        
        return AnalysisResult(
            history=self.analysis_history,
            results=self.intermediate_results,
            reasoning=self.reasoning_chain,
            total_time=total_time
        )
    
    def _execute_step_with_retry(self, sub_query: SubQuery, 
                                 step_num: int,
                                 max_retries: int) -> StepResult:
        """执行单步分析"""
        
        error_history = []
        
        for attempt in range(max_retries + 1):
            try:
                if attempt > 0:
                    print(f"重试 {attempt}/{max_retries}...")
                
                step_result = self._execute_step(
                    sub_query, step_num, error_history
                )
                
                if step_result.success:
                    return step_result
                else:
                    error_history.append(step_result.data)  # 错误信息
                    
            except Exception as e:
                print(f"执行异常: {e}")
                error_history.append(str(e))
        
        return StepResult(
            subquery=sub_query,
            code="# 执行失败",
            data=None,
            insights=f"分析步骤失败: {error_history[-1] if error_history else 'Unknown'}",
            success=False
        )
    
    def _execute_step(self, sub_query: SubQuery, step_num: int,
                     error_history: List = None) -> StepResult:
        """执行单个分析步骤"""
        
        step_start = time.time()
        
        context = self._build_context(step_num)
        
        print(f"正在生成代码...")
        code = self._generate_code_for_step(sub_query, context, error_history)
        
        # 显示代码
        print(f"生成的代码:")
        for i, line in enumerate(code.split('\n'), 1):
            print(f"{Style.DIM}{i:3d} |{Style.RESET_ALL} {line}")
        
        print(f"\n正在执行...")
        exec_result = self.executor.execute(code)
        
        execution_time = time.time() - step_start
        
        if exec_result.success:
            print(f"执行成功 (耗时: {execution_time:.2f}秒)")
            if exec_result.output:
                print(f"{exec_result.output[:200]}")

            print(f"\n 提取发现...")
            insights = self._extract_insights(sub_query, code, exec_result)
            print(f"{insights}")
            
            return StepResult(
                subquery=sub_query,
                code=code,
                data=exec_result.result,
                insights=insights,
                success=True,
                execution_time=execution_time,
                output=exec_result.output
            )
        else:
            print(f"{exec_result.error['type']}")
            print(f"{exec_result.error['message']}")
            
            return StepResult(
                subquery=sub_query,
                code=code,
                data=exec_result.error,
                insights="执行失败",
                success=False,
                execution_time=execution_time
            )
    
    def _build_context(self, step_num: int) -> Dict[str, Any]:
        """构建当前步骤的上下文"""
        
        context = {
            'data_info': self.data_loader.get_metadata(),
            'step_number': step_num,
            'previous_steps': []
        }
        
        for i, step in enumerate(self.analysis_history, 1):
            if step.success:
                context['previous_steps'].append({
                    'step': i,
                    'question': step.subquery.text,
                    'code': step.code,
                    'result': str(step.data)[:200], 
                    'insights': step.insights
                })
        
        return context
    
    def _generate_code_for_step(self, sub_query: SubQuery, 
                                context: Dict,
                                error_history: List = None) -> str:
        """为特定步骤生成代码"""
        
        prompt = f"""
你是一个Python数据分析专家。当前正在进行多步骤数据分析。

当前步骤: 第 {context['step_number']} 步
当前问题: {sub_query.text}

数据信息:
- 列名: {context['data_info']['columns']}
- 数据类型: {context['data_info']['dtypes']}
- 数据维度: {context['data_info']['shape']}

环境说明:
- 数据已加载到变量 df (pandas DataFrame)
- 可用库: pandas(pd), numpy(np), matplotlib.pyplot(plt), seaborn(sns)
- 将最终结果保存到变量 result 中

"""
        
        if hasattr(self, 'conversation_history') and self.conversation_history:
            prompt += "\n# 之前的对话历史（供参考理解上下文）:\n"
            recent_history = self.conversation_history[-3:] if len(self.conversation_history) > 3 else self.conversation_history
            for i, entry in enumerate(recent_history, 1):
                prompt += f"\n对话{i}:\n"
                prompt += f"  问题: {entry.get('question', 'N/A')}\n"
                answer = entry.get('answer', '')
                if answer:
                    answer_preview = answer[:100] + '...' if len(answer) > 100 else answer
                    prompt += f"  结果: {answer_preview}\n"
            prompt += "\n"
        
        if context['previous_steps']:
            prompt += "\n# 当前深度分析中之前步骤的结果:\n"
            for prev in context['previous_steps']:
                prompt += f"步骤{prev['step']}: {prev['question']}\n"
                prompt += f"  结果: {prev['result']}\n"
                prompt += f"  发现: {prev['insights']}\n"
        
        # 添加错误历史
        if error_history:
            prompt += f"\n之前的尝试失败了，错误信息:\n{error_history[-1]}\n"
            prompt += "请修正错误后重新生成代码。\n"
        
        prompt += f"""
要求:
1. 只输出Python代码，不要任何解释
2. 代码要完整且可直接运行
3. 将最终结果保存到变量 result 中
4. 如果需要可视化，保存图表到 outputs/ 目录
5. 使用 ```python 代码块包裹

生成代码:
"""
        
        response = self.llm._call_api(
            messages=[
                {"role": "system", "content": self.llm._get_system_prompt()},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )
        
        code = self.llm._extract_code(response)
        return code
    
    def _extract_insights(self, sub_query: SubQuery, 
                         code: str, 
                         exec_result: ExecutionResult) -> str:
        """从执行结果中提取关键洞察"""
        
        # 格式化结果
        result_str = str(exec_result.result)[:500]
        
        prompt = f"""
分析以下数据分析结果，提取1-2句关键发现。

问题: {sub_query.text}
执行结果: {result_str}
输出: {exec_result.output[:200] if exec_result.output else 'None'}

要求:
1. 用一句话总结最重要的发现
2. 包含具体的数字
3. 简洁明了
4. 不要说"根据分析"之类的废话

示例:
"2017年总销售额为$379,200，较2016年增长34.8%"

你的洞察:
"""
        
        try:
            insights = self.llm._call_api(
                messages=[
                    {"role": "system", "content": "你是数据分析专家，擅长提炼关键洞察。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3
            )
            return insights.strip()
        except:
            return f"完成分析: {sub_query.text}"
    
    def _update_reasoning(self, sub_query: SubQuery, step_result: StepResult):
        """更新推理链"""
        
        reasoning = f"第{step_result.subquery.order}步: {sub_query.text}\n"
        if step_result.success:
            reasoning += f"  发现: {step_result.insights}\n"
        else:
            reasoning += f"  状态: 执行失败\n"
        
        self.reasoning_chain.append(reasoning)


class ResultSynthesizer:
    """综合所有子分析结果，生成最终答案"""
    
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
    
    def synthesize(self, original_question: str,
                  analysis_result: AnalysisResult) -> str:
        """
        综合分析结果生成最终答案
        
        Args:
            original_question: 原始问题
            analysis_result: 完整的分析结果
        
        Returns:
            str: 最终综合答案
        """
        
        prompt = self._build_synthesis_prompt(
            original_question,
            analysis_result
        )
        
        try:
            final_answer = self.llm._call_api(
                messages=[
                    {"role": "system", "content": "你是一个专业的数据分析师，擅长综合多个分析结果生成全面的报告。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.7
            )
            
            return final_answer
            
        except Exception as e:
            print(f"答案综合失败: {e}")
            return self._generate_fallback_answer(analysis_result)
    
    def _build_synthesis_prompt(self, question: str, 
                                analysis_result: AnalysisResult) -> str:
        """构建综合Prompt"""
        
        # 整理所有成功的步骤
        successful_steps = [s for s in analysis_result.history if s.success]
        
        steps_summary = []
        for i, step in enumerate(successful_steps, 1):
            steps_summary.append(f"""
### 步骤{i}: {step.subquery.text}
**执行代码:**
```python
{step.code}
```

**结果:** {str(step.data)[:300]}...
**关键发现:** {step.insights}
""")
        
        prompt = f"""
用户提出了一个复杂的数据分析问题，我们通过{len(successful_steps)}步深度分析得到了结果。
现在请综合所有结果，生成一个全面、专业、有洞察力的答案。

**原始问题:**
{question}

**分析过程:**
{''.join(steps_summary)}

**推理链:**
{''.join(analysis_result.reasoning)}

**要求:**
1. 用 <think>...</think> 包裹完整的思考过程，展示每一步的推理逻辑
2. 在思考之后，生成最终答案，要求:
   - 直接回答原问题
   - 结构清晰（使用Markdown格式：标题、列表、加粗）
   - 包含具体数据支撑
   - 提供深度洞察和建议
   - 总结关键发现
3. 不要说"根据上述分析"之类的废话，直接陈述结论

生成答案:
"""
        
        return prompt
    
    def _generate_fallback_answer(self, analysis_result: AnalysisResult) -> str:
        
        answer_parts = ["## 深度分析结果\n"]
        
        for i, step in enumerate(analysis_result.history, 1):
            if step.success:
                answer_parts.append(f"\n### 步骤{i}: {step.subquery.text}")
                answer_parts.append(f"{step.insights}\n")
        
        return "\n".join(answer_parts)

class DeepAnalysisManager:
    
    def __init__(self, conversation_manager):
        
        self.conv_manager = conversation_manager
        
        # 初始化各个组件
        self.decomposer = QueryDecomposer(
            llm_client=conversation_manager.llm
        )
        
        self.engine = IterativeAnalysisEngine(
            data_loader=conversation_manager.data_loader,
            code_executor=conversation_manager.executor,
            llm_client=conversation_manager.llm,
            conversation_history=conversation_manager.history
        )
        
        self.synthesizer = ResultSynthesizer(
            llm_client=conversation_manager.llm
        )
    
    def analyze_deeply(self, user_question: str) -> str:
        """
        执行深度分析的完整流程
        
        Args:
            user_question: 用户问题
        
        Returns:
            str: 最终综合答案
        """
        
        print(f"深度分析模式")
        
        total_start = time.time()
        
        #阶段1: 问题分解
        print(f"阶段 1/3: 问题分解")
        
        data_context = self.conv_manager.data_loader.get_metadata()
        sub_queries = self.decomposer.decompose(user_question, data_context)
        
        # 显示子问题
        print(f"\n已将问题拆解为 {len(sub_queries)} 个子问题:\n")
        for i, sq in enumerate(sub_queries, 1):
            print(f"   {i}. {sq.text}")
        
        #迭代分析
        print(f"阶段 2/3: 迭代分析")
        
        analysis_result = self.engine.analyze(sub_queries)
        
        # 统计成功率
        successful = sum(1 for s in analysis_result.history if s.success)
        print(f"\n分析完成: {successful}/{len(sub_queries)} 步成功")
        
        #结果综合 
        print(f"阶段 3/3: 结果综合")
        print(f"正在生成综合答案...")
        
        final_answer = self.synthesizer.synthesize(
            user_question,
            analysis_result
        )
        
        total_time = time.time() - total_start
        
        # 保存到历史
        self.conv_manager._save_to_history(
            question=user_question,
            code=None,  # 深度分析没有单一代码
            result=analysis_result,
            answer=final_answer,
            success=True
        )
        
        print(f"分析统计")
        print(f"总耗时: {total_time:.2f}秒")
        print(f"分析步骤: {len(sub_queries)}")
        print(f"成功率: {successful}/{len(sub_queries)} ({successful/len(sub_queries)*100:.0f}%)")
        
        return final_answer

