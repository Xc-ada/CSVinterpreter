import re
import json
from typing import Dict, List, Optional, Any


class LLMClient:
    """通义千问API客户端"""
    
    SUPPORTED_MODELS = ['qwen-max', 'qwen-plus', 'qwen-turbo', 'qwen-vl-max']
    
    def __init__(self, api_key: str, model: str = "qwen-max"):
        """
        初始化通义千问客户端
        
        Args:
            api_key: 通义千问 API Key (DASHSCOPE_API_KEY)
            model: 模型名称，默认 qwen-max
        """
        self.api_key = api_key
        self.model = model
        self.client = self._init_client()
        
        print(f"初始化通义千问客户端: {self.model}")
    
    def _init_client(self):
        """初始化 DashScope 客户端"""
        import dashscope
        dashscope.api_key = self.api_key
        return dashscope

    def generate_code(self, 
                     question: str, 
                     context: Dict[str, Any],
                     error_history: Optional[List[Dict]] = None,
                     conversation_history: Optional[List[Dict]] = None) -> str:
        """
        生成Python数据分析代码
        
        Args:
            question: 用户问题
            context: 数据上下文（元信息）
            error_history: 之前的错误历史（用于纠错）
            conversation_history: 对话历史（用于理解上下文）
        
        Returns:
            str: Python代码
        """
        prompt = self._build_code_generation_prompt(question, context, error_history, conversation_history)
        
        response = self._call_api(
            messages=[
                {"role": "system", "content": self._get_system_prompt()},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1
        )
        
        # 提取代码
        code = self._extract_code(response)
        
        return code
    
    def explain_result(self, 
                      question: str, 
                      code: str, 
                      result: Any,
                      output: Optional[str] = None) -> str:
        """
        将代码执行结果解释为自然语言
        
        Args:
            question: 原始问题
            code: 执行的代码
            result: 执行结果
            output: 代码输出
        
        Returns:
            str: 自然语言解释
        """
        prompt = self._build_explanation_prompt(question, code, result, output)
        
        response = self._call_api(
            messages=[
                {"role": "system", "content": "你是一个专业的数据分析师，善于用简洁易懂的语言解释分析结果。"},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7  # 较高温度使回答更自然
        )
        
        return response
    
    def _get_system_prompt(self) -> str:
        """获取系统提示词（代码生成）"""
        return """你是一个Python数据分析专家，精通pandas、numpy和数据可视化。

你的任务是根据用户的数据分析需求，生成可执行的Python代码。

关键要求：
1. 只输出Python代码，不要任何解释文字
2. 代码必须完整且可直接运行
3. 将最终结果保存到变量 `result` 中
4. 如果需要可视化，使用matplotlib/seaborn并保存为PNG文件
5. 使用 ```python 代码块包裹代码
6. 代码要简洁高效，避免冗余

可用资源：
- df: pandas DataFrame（已加载的数据）
- pd: pandas库
- np: numpy库
- plt: matplotlib.pyplot
- sns: seaborn库
"""
    
    def _build_code_generation_prompt(self, 
                                     question: str, 
                                     context: Dict[str, Any],
                                     error_history: Optional[List[Dict]] = None,
                                     conversation_history: Optional[List[Dict]] = None) -> str:
        """构建代码生成提示词"""
        
        # 基础信息
        prompt_parts = [
            "# 数据信息",
            f"列名: {context['columns']}",
            f"数据类型: {context['dtypes']}",
            f"数据维度: {context['shape'][0]}行 × {context['shape'][1]}列",
            "",
            "# 数据样例（前3行）:"
        ]
        
        # 添加样例数据
        if 'sample_data' in context and context['sample_data']:
            for i, row in enumerate(context['sample_data'][:3], 1):
                prompt_parts.append(f"行{i}: {row}")
        
        prompt_parts.append("")
        
        if conversation_history and len(conversation_history) > 0:
            prompt_parts.append("# 对话历史（供参考理解上下文）:")
            # 只显示最近3轮对话
            recent_history = conversation_history[-3:] if len(conversation_history) > 3 else conversation_history
            for i, entry in enumerate(recent_history, 1):
                prompt_parts.append(f"\n第{i}轮对话:")
                prompt_parts.append(f"  用户问题: {entry.get('question', 'N/A')}")
                code = entry.get('code', '')
                if code:
                    code_preview = code[:150] + '...' if len(code) > 150 else code
                    prompt_parts.append(f"  生成的代码: {code_preview}")
                answer = entry.get('answer', '')
                if answer:
                    answer_preview = answer[:100] + '...' if len(answer) > 100 else answer
                    prompt_parts.append(f"  分析结果: {answer_preview}")
            prompt_parts.append("")
        
        prompt_parts.append(f"# 当前用户问题\n{question}")
        
        if error_history:
            prompt_parts.append("\n# 之前的尝试失败了，请修正错误")
            for i, error in enumerate(error_history[-2:], 1):  # 只显示最近2个错误
                prompt_parts.append(f"\n错误 {i}:")
                prompt_parts.append(f"  类型: {error['type']}")
                prompt_parts.append(f"  信息: {error['message']}")
                if error.get('suggestions'):
                    prompt_parts.append(f"  提示: {'; '.join(error['suggestions'])}")
        
        prompt_parts.append("\n# 请生成代码")
        
        return "\n".join(prompt_parts)
    
    def _build_explanation_prompt(self, 
                                 question: str, 
                                 code: str, 
                                 result: Any,
                                 output: Optional[str] = None) -> str:
        """构建结果解释提示词"""
        
        # 格式化结果
        if isinstance(result, (int, float, str)):
            result_str = str(result)
        elif hasattr(result, 'to_string'):
            result_str = result.to_string()
        elif hasattr(result, '__str__'):
            result_str = str(result)
        else:
            result_str = repr(result)
        
        # 限制结果长度
        if len(result_str) > 1000:
            result_str = result_str[:1000] + "\n... (结果过长，已截断)"
        
        prompt = f"""用户问了一个数据分析问题，我们已经执行了代码并得到结果。请用专业且易懂的语言解释这个结果。

用户问题：
{question}

执行的代码：
```python
{code}
```

执行结果：
{result_str}

{f"代码输出：{output}" if output else ""}

请提供：
1. 直接回答用户的问题（用具体数据支撑）
2. 关键发现和洞察
3. 如果生成了图表，提示用户查看
4. 回答要简洁专业，突出重点

你的回答："""
        
        return prompt
    
    def _call_api(self, messages: List[Dict], temperature: float = 0.7) -> str:
        """
        调用通义千问API
        
        Args:
            messages: 消息列表
            temperature: 温度参数
        
        Returns:
            str: 模型响应
        """
        try:
            from dashscope import Generation
            
            response = Generation.call(
                model=self.model,
                messages=messages,
                temperature=temperature,
                result_format='message'
            )
            
            if response.status_code == 200:
                return response.output.choices[0].message.content
            else:
                raise Exception(f"API错误: {response.message}")
        
        except Exception as e:
            raise Exception(f"API调用失败: {str(e)}")
    
    def _extract_code(self, response: str) -> str:
        """
        从LLM响应中提取代码
        
        Args:
            response: LLM的完整响应
        
        Returns:
            str: 提取的Python代码
        """
        # 尝试匹配 ```python ... ```
        pattern = r'```python\s*(.*?)\s*```'
        matches = re.findall(pattern, response, re.DOTALL)
        
        if matches:
            return matches[0].strip()
        
        # 尝试匹配 ``` ... ```（无语言标记）
        pattern = r'```\s*(.*?)\s*```'
        matches = re.findall(pattern, response, re.DOTALL)
        
        if matches:
            # 过滤掉非Python代码块
            for match in matches:
                if any(keyword in match for keyword in ['import', 'def', 'class', '=', 'df']):
                    return match.strip()
        
        # 如果没有代码块标记，检查是否整个响应就是代码
        if any(keyword in response for keyword in ['import ', 'df[', 'df.', 'result =', 'plt.']):
            return response.strip()
        
        # 都没有，返回原始响应
        return response.strip()
    
    def __repr__(self):
        return f"LLMClient(provider={self.provider}, model={self.model})"



