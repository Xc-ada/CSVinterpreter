CSV数据分析系统 - 实现说明

基于通义千问大模型的智能数据分析系统，模仿OpenAI Code Interpreter实现。

---

功能实现

1. 读取CSV数据

- 支持指定任意路径的CSV文件
- 自动解析CSV文件，提取列名、数据类型、样例数据
- 智能数据清洗（去除货币符号、百分号，转换为数值）
- 生成数据元信息供LLM理解数据结构

核心代码:

class CSVDataLoader:
    def load_data(self) -> pd.DataFrame:
        """加载并清洗CSV数据"""
        df = pd.read_csv(self.file_path)
        df = self.clean_data(df)
        return df
    
    def get_metadata(self) -> Dict:
        """提取数据元信息供LLM使用"""
        return {
            'columns': list(self.df.columns),
            'dtypes': self.df.dtypes.to_dict(),
            'shape': self.df.shape,
            'sample_data': self.df.head(3).to_dict('records')
        }



2. 读取数据分析问题

- 通过命令行接受自然语言数据分析请求
- 支持交互式多轮对话
- 维护对话历史,每轮对话的问题、代码、结果都被记录

核心代码:

class ConversationManager:
    def __init__(self):
        self.history = []  # 存储对话历史
    
    def chat(self, user_question: str) -> str:
        """处理用户问题，支持基于历史的连续对话"""
        # 1. 生成代码时传递对话历史
        code = self.llm.generate_code(
            question=user_question,
            conversation_history=self.history  # 传递历史
        )
        
        # 2. 执行代码并获取结果
        result = self.executor.execute(code)
        
        # 3. 保存到历史
        self.history.append({
            'question': user_question,
            'code': code,
            'result': result,
            'answer': answer
        })


对话历史格式:
self.history = [
    {
        'question': '2023年的总销售额是多少？',
        'code': 'result = df[df["Year"]==2023]["Sales"].sum()',
        'result': 234567.89,
        'answer': '2023年总销售额为$234,567.89'
    },
    {
        'question': '和2022年相比呢？',  # 引用上一轮
        'code': '...',
        'result': '...',
        'answer': '...'
    }
]
3. 基于大模型的Python代码生成

- 使用通义千问(Qwen)大模型生成Python代码
- Prompt包含：数据信息、对话历史、用户问题
- 自动提取LLM响应中的Python代码块

核心代码:

class LLMClient:
    def generate_code(self, question: str, context: Dict, 
                     conversation_history: List[Dict] = None) -> str:
        """调用LLM生成Python代码"""
        
        # 构建Prompt
        prompt = self._build_code_generation_prompt(
            question, context, conversation_history
        )
        
        # 调用通义千问API
        response = self._call_api(messages=[
            {"role": "system", "content": "你是Python数据分析专家..."},
            {"role": "user", "content": prompt}
        ])
        
        # 提取代码
        code = self._extract_code(response)
        return code


Prompt结构:

'''
# 数据信息
列名: ['Year', 'Category', 'Product', 'Sales', 'Rating']
数据类型: {...}
数据维度: 76行 × 5列

# 对话历史:
第1轮对话:
  用户问题: 2023年的总销售额是多少？
  生成的代码: result = df[df['Year'] == 2023]['Sales'].sum()
  分析结果: 2023年总销售额为 $234,567.89

# 当前用户问题
和2022年相比增长了多少？

# 请生成代码
'''

4. 代码纠错

- 捕获代码执行错误
- 将错误类型、错误信息、修复建议反馈给LLM
- LLM根据错误信息重新生成代码
- 支持多次重试

核心代码:

class ConversationManager:
    def chat(self, user_question: str, max_retries: int = 3) -> str:
        error_history = []
        
        for attempt in range(max_retries):
            # 生成代码（如果有错误历史，一并传递）
            code = self.llm.generate_code(
                question=user_question,
                error_history=error_history if attempt > 0 else None
            )
            
            # 执行代码
            exec_result = self.executor.execute(code)
            
            if exec_result.success:
                return self._generate_answer(...)
            else:
                # 记录错误信息
                error_history.append({
                    'type': exec_result.error['type'],
                    'message': exec_result.error['message'],
                    'suggestions': exec_result.error['suggestions']
                })
                # 继续重试...

class CodeExecutor:
    def _get_error_suggestions(self, error_type: str) -> List[str]:
        """根据错误类型提供修复建议"""
        if error_type == 'KeyError':
            return [
                "列名不存在，请检查拼写",
                f"可用列名: {list(self.df.columns)}"
            ]
        elif error_type == 'TypeError':
            return [
                "数据类型错误，请检查操作是否适用",
                "尝试使用 .astype() 转换类型"
            ]
        # ...更多错误类型处理

5. 代码运行

- 在沙箱环境中安全执行Python代码
- 使用受限的命名空间，防止危险操作
- 捕获代码执行的所有异常，确保系统不崩溃
- 返回执行结果、输出和错误信息

核心代码:

class CodeExecutor:
    def execute(self, code: str) -> ExecutionResult:
        """安全执行Python代码"""
        try:
            # 捕获标准输出
            output_buffer = io.StringIO()
            
            with redirect_stdout(output_buffer):
                # 在受限命名空间中执行
                exec(code, self.namespace)
            
            # 获取结果
            result = self.namespace.get('result', None)
            output = output_buffer.getvalue()
            
            return ExecutionResult(
                success=True,
                result=result,
                output=output
            )
            
        except Exception as e:
            # 捕获所有异常，系统不崩溃
            error_info = {
                'type': type(e).__name__,
                'message': str(e),
                'suggestions': self._get_error_suggestions(type(e).__name__)
            }
            
            return ExecutionResult(
                success=False,
                error=error_info
            )

沙箱环境:

def _init_namespace(self) -> Dict[str, Any]:
    """初始化执行环境"""
    return {
        'df': self.df,           
        'pd': pd,                
        'np': np,                
        'plt': plt,                
        'sns': sns,               
        'result': None,            
        '__builtins__': {           
            'print': print,
            'len': len,
            'sum': sum,
            '__import__': __import__,  
            # ...
        }
    }


6. 基于运行结果的解释与应答

- 将代码执行结果传递给LLM
- LLM生成自然语言解释
- 返回用户友好的分析报告

核心代码:

class LLMClient:
    def explain_result(self, question: str, code: str, 
                      result: Any, output: str) -> str:
        """生成自然语言解释"""
        
        prompt = f"""
# 用户问题
{question}

# 执行的代码
{code}

# 执行结果
{result}

# 程序输出
{output}

请用自然语言解释分析结果，回答用户问题。
"""
        
        response = self._call_api(messages=[...])
        return response

7. 额外功能：深度分析模式

- 将复杂问题拆解为多个子问题
- 逐步执行每个子问题的分析
- 积累中间结果和推理链
- 最终综合生成完整答案

核心代码:

class QueryDecomposer:
    def decompose(self, question: str, data_context: Dict) -> List[SubQuery]:
        """将复杂问题拆解为子问题"""
        # 调用LLM分析问题结构
        sub_queries = self.llm.decompose_query(question)
        return sub_queries

class IterativeAnalysisEngine:
    def analyze(self, sub_queries: List[SubQuery]) -> AnalysisResult:
        """迭代执行子问题分析"""
        for sub_query in sub_queries:
            # 1. 生成代码
            code = self._generate_code_for_step(sub_query)
            
            # 2. 执行代码
            result = self.executor.execute(code)
            
            # 3. 提取洞察
            insights = self._extract_insights(sub_query, result)
            
            # 4. 保存中间结果
            self.analysis_history.append(StepResult(...))

class ResultSynthesizer:
    def synthesize(self, question: str, 
                   analysis_result: AnalysisResult) -> str:
        """综合所有分析结果生成最终答案"""
        # 汇总所有步骤的发现
        all_insights = [step.insights for step in analysis_result.history]
        
        # 让LLM生成综合答案
        final_answer = self.llm.synthesize_answer(question, all_insights)
        return final_answer

完整流程:

用户问题: "详细分析2023年销售情况并找出驱动因素"
    ↓
阶段1: 问题分解
    拆解为: [
        "2023年各类别销售额统计",
        "2023年与往年对比",
        "销售增长的主要类别",
        "驱动因素分析"
    ]
    ↓
阶段2: 迭代分析
    步骤1: 执行"2023年各类别销售额统计"
           → 生成代码 → 执行 → 提取洞察
    步骤2: 执行"2023年与往年对比"
           → 生成代码 → 执行 → 提取洞察
    步骤3: ...
    步骤4: ...
    ↓
阶段3: 结果综合
    综合所有洞察，生成完整答案