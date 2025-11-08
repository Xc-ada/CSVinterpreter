import io
import os
import sys
import traceback
import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import seaborn as sns
from contextlib import redirect_stdout, redirect_stderr
from typing import Any, Dict, Optional
from dataclasses import dataclass

matplotlib.use('Agg')

@dataclass
class ExecutionResult:
    success: bool                   
    result: Any                  
    output: Optional[str] = None    
    error: Optional[Dict] = None     
    execution_time: float = 0.0    


class CodeExecutor:
    
    def __init__(self, df: pd.DataFrame):

        self.df = df
        self.global_namespace = self._init_namespace()
        self.execution_count = 0
    
    def _init_namespace(self) -> Dict[str, Any]:
        """
        初始化执行环境的命名空间
        """
        namespace = {
            'df': self.df,
            
            'pd': pd,
            'np': np,
            'plt': plt,
            'sns': sns,
            
            'result': None,
            
            # 内置函数
            '__builtins__': {
                'print': print,
                'len': len,
                'range': range,
                'enumerate': enumerate,
                'zip': zip,
                'map': map,
                'filter': filter,
                'sum': sum,
                'min': min,
                'max': max,
                'abs': abs,
                'round': round,
                'sorted': sorted,
                'list': list,
                'dict': dict,
                'set': set,
                'tuple': tuple,
                'str': str,
                'int': int,
                'float': float,
                'bool': bool,
                'type': type,
                'isinstance': isinstance,
                '__import__': __import__, 
                'Exception': Exception,
            }
        }
        
        return namespace
    
    def execute(self, code: str) -> ExecutionResult:
        import time
        
        self.execution_count += 1
        start_time = time.time()
        
        output_buffer = io.StringIO()
        error_buffer = io.StringIO()
        
        try:
            # 重定向输出
            with redirect_stdout(output_buffer), redirect_stderr(error_buffer):
                # 执行代码
                exec(code, self.global_namespace)
            
            execution_time = time.time() - start_time
            
            result = self.global_namespace.get('result')
            output = output_buffer.getvalue()
            
            return ExecutionResult(
                success=True,
                result=result,
                output=output if output else None,
                error=None,
                execution_time=execution_time
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            
            # 格式化错误信息
            error_info = self._format_error(e, code)
            
            return ExecutionResult(
                success=False,
                result=None,
                output=output_buffer.getvalue() if output_buffer.getvalue() else None,
                error=error_info,
                execution_time=execution_time
            )
    
    def _format_error(self, exception: Exception, code: str) -> Dict[str, Any]:
        """
        格式化错误信息
        
        Args:
            exception: 异常对象
            code: 执行的代码
        
        Returns:
            Dict: 格式化的错误信息
        """
        tb_str = traceback.format_exc()
        
        error_type = type(exception).__name__
        error_message = str(exception)
        
        tb_lines = tb_str.split('\n')
        line_info = None
        for line in tb_lines:
            if 'line' in line.lower() and '<string>' in line:
                line_info = line.strip()
                break
        
        error_dict = {
            'type': error_type,
            'message': error_message,
            'line_info': line_info,
            'traceback': tb_str,
            'suggestions': self._get_error_suggestions(error_type, error_message)
        }
        
        return error_dict
    
    def _get_error_suggestions(self, error_type: str, error_message: str) -> list:
        """
        根据错误类型提供修复建议

        Args:
            error_type: 错误类型
            error_message: 错误消息
        
        Returns:
            list: 修复建议列表
        """
        suggestions = []
        
        if error_type == 'KeyError':
            suggestions.append(f"列名错误。可用列名: {list(self.df.columns)}")
            suggestions.append("请检查列名的大小写和拼写")
        
        elif error_type == 'NameError':
            suggestions.append("变量未定义。请检查变量名是否正确")
            suggestions.append(f"可用的主要变量: df (DataFrame), pd, np, plt, sns")
        
        elif error_type == 'AttributeError':
            suggestions.append("属性或方法不存在。请检查API调用")
        
        elif error_type == 'TypeError':
            suggestions.append("类型错误。请检查数据类型是否匹配")
            if 'Sales' in error_message or 'Rating' in error_message:
                suggestions.append(f"Sales和Rating已转换为数值类型")
        
        elif error_type == 'ValueError':
            suggestions.append("值错误。请检查输入参数")
        
        elif error_type == 'IndexError':
            suggestions.append(f"索引超出范围。数据shape: {self.df.shape}")
        
        elif 'ModuleNotFoundError' in error_type or 'ImportError' in error_type:
            suggestions.append("模块导入错误。可用库: pandas(pd), numpy(np), matplotlib.pyplot(plt), seaborn(sns)")
        
        else:
            # 其他未明确处理的错误类型
            suggestions.append(f"遇到 {error_type} 错误")
            suggestions.append("请检查代码逻辑和数据")
            suggestions.append(f"可用数据列: {list(self.df.columns)}")
            suggestions.append(f"数据维度: {self.df.shape}")
        
        return suggestions
    
    def reset_namespace(self):
        self.global_namespace = self._init_namespace()
        self.execution_count = 0
        print("执行环境已重置")
    
    def __repr__(self):
        return f"CodeExecutor(df_shape={self.df.shape}, executions={self.execution_count})"


# 测试代码
if __name__ == "__main__":
    test_df = pd.DataFrame({
        'Year': [2017, 2016, 2015],
        'Category': ['A', 'B', 'C'],
        'Sales': [1000, 2000, 3000]
    })
    
    executor = CodeExecutor(test_df)
    
    # 测试1: 成功执行
    print("\n测试1: 正常执行")
    code1 = """
result = df['Sales'].sum()
print(f"总销售额: {result}")
"""
    result1 = executor.execute(code1)
    print(f"能否运行: {result1.success}")
    print(f"结果: {result1.result}")
    print(f"输出: {result1.output}")
    
    # 测试2: 错误处理
    print("\n测试2: 错误处理")
    code2 = """
result = df['sles'].sum()  # 故意写错列名
"""
    result2 = executor.execute(code2)
    print(f"能否运行: {result2.success}")
    print(f"错误类型: {result2.error['type']}")
    print(f"错误信息: {result2.error['message']}")
    print(f"建议: {result2.error['suggestions']}")

