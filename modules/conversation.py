import os
import time
from typing import List, Dict, Any, Optional
from colorama import Fore, Style, init

from .csv_loader import CSVDataLoader
from .code_executor import CodeExecutor, ExecutionResult
from .llm_client import LLMClient
from .deep_analysis import DeepAnalysisManager

# 初始化colorama（Windows兼容）
init(autoreset=True)


class ConversationManager:
    """对话会话管理器"""
    
    def __init__(self, 
                 csv_path: str, 
                 llm_client: LLMClient,
                 max_retries: int = 10,
                 save_plots: bool = True):
        """
        初始化对话管理器
        
        Args:
            csv_path: CSV文件路径
            llm_client: LLM客户端
            max_retries: 代码错误最大重试次数
            save_plots: 是否保存生成的图表
        """
        print(f"初始化CSV数据分析系统")
        
        print(f"正在加载数据...")
        self.data_loader = CSVDataLoader(csv_path)
        self.df = self.data_loader.load()
        
        self.executor = CodeExecutor(self.df)
        
        self.llm = llm_client

        self.max_retries = max_retries
        self.save_plots = save_plots
        self.output_dir = "outputs"
        
        if self.save_plots and not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.history: List[Dict[str, Any]] = []
        self.conversation_count = 0
        
        self.deep_analysis = None
        
    
    def chat(self, user_question: str) -> str:
        """
        处理用户问题的主流程
        
        Args:
            user_question: 用户的数据分析问题
        
        Returns:
            str: 自然语言回答
        """
        self.conversation_count += 1
        
        choice = input(f"\n{Fore.YELLOW}是否启用深度分析模式? [y/n] (默认: n): ").strip().lower()
            
        if choice == 'y':
            if self.deep_analysis is None:
                self.deep_analysis = DeepAnalysisManager(self)
            
            return self.deep_analysis.analyze_deeply(user_question)
        
        print(f"对话 #{self.conversation_count}")
        
        print(f"正在思考...")
        
        context = self.data_loader.get_metadata()
        
        error_history = []
        code = None
        exec_result = None
        
        for attempt in range(self.max_retries):
            try:
                # 生成代码
                print(f"\n生成代码 (尝试 {attempt + 1}/{self.max_retries})...")
                
                code = self.llm.generate_code(
                    question=user_question,
                    context=context,
                    error_history=error_history if attempt > 0 else None,
                    conversation_history=self.history 
                )
                
                self._display_code(code, attempt + 1)

                print(f"\n执行代码中...")
                exec_result = self.executor.execute(code)
                
                if exec_result.success:
                    print(f"执行成功！ (耗时: {exec_result.execution_time:.2f}秒)")
                    
                    if exec_result.output:
                        print(f"\n代码输出:")
                        print(f"{exec_result.output}")
                    
                    break
                
                else:
                    # 执行失败
                    print(f"执行失败: {exec_result.error['type']}")
                    print(f"{exec_result.error['message']}")
                    
                    # 添加到错误历史
                    error_history.append(exec_result.error)
                    
                    # 如果是最后一次尝试，返回错误信息
                    if attempt == self.max_retries - 1:
                        error_summary = self._format_error_summary(error_history)
                        answer = f"尝试了 {self.max_retries} 次仍然无法成功执行。\n\n{error_summary}"
                        
                        self._save_to_history(user_question, code, None, answer, success=False)
                        
                        return answer
                    
                    print(f"正在重新生成代码...")
            
            except Exception as e:
                print(f"发生异常: {str(e)}")
                if attempt == self.max_retries - 1:
                    return f"系统错误: {str(e)}"
        
        # 生成自然语言解释
        print(f"\n生成回答...")
        
        try:
            answer = self.llm.explain_result(
                question=user_question,
                code=code,
                result=exec_result.result,
                output=exec_result.output
            )
        except Exception as e:
            print(f"解释生成失败: {e}")
            answer = f"分析完成。结果: {exec_result.result}"
        
        self._save_to_history(user_question, code, exec_result.result, answer, success=True)
        
        return answer
    
    def _display_code(self, code: str, attempt: int):
        print(f"\n生成的代码 (尝试 {attempt}):")
        lines = code.split('\n')
        for i, line in enumerate(lines, 1):
            print(f"{Style.DIM}{i:3d} |{Style.RESET_ALL} {line}")
    
    
    def _format_error_summary(self, error_history: List[Dict]) -> str:
        """格式化错误摘要"""
        summary = "错误历史:\n"
        for i, error in enumerate(error_history, 1):
            summary += f"\n{i}. {error['type']}: {error['message']}"
            if error.get('suggestions'):
                summary += f"\n   建议: {'; '.join(error['suggestions'])}"
        
        return summary
    
    def _save_to_history(self, 
                        question: str, 
                        code: Optional[str], 
                        result: Any,
                        answer: str,
                        success: bool = True):
        """保存对话到历史"""
        history_entry = {
            'id': self.conversation_count,
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'question': question,
            'code': code,
            'result': str(result)[:500] if result is not None else None,  # 限制长度
            'answer': answer,
            'success': success
        }
        
        self.history.append(history_entry)
    
    def get_history(self) -> List[Dict[str, Any]]:
        """获取对话历史"""
        return self.history
    
    
    def __repr__(self):
        return f"ConversationManager(conversations={len(self.history)}, df_shape={self.df.shape})"




