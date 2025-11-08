import pandas as pd
import os
from typing import Dict, Any


class CSVDataLoader:
    
    def __init__(self, file_path: str):
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"CSV文件不存在: {file_path}")
        
        self.file_path = file_path
        self.df = None
        self.metadata = {}
        self.original_df = None  
    
    def load(self) -> pd.DataFrame:

        try:
            self.original_df = pd.read_csv(self.file_path)
            print(f"成功读取CSV文件: {self.file_path}")
            print(f"   原始数据: {self.original_df.shape[0]} 行, {self.original_df.shape[1]} 列")
            
            # 复制数据用于处理
            self.df = self.original_df.copy()
            
            # 数据清洗
            self.df = self._clean_data(self.df)
            
            # 提取元信息
            self._extract_metadata()
            
            print(f"数据预处理完成")
            
            return self.df
            
        except Exception as e:
            raise ValueError(f"加载CSV文件失败: {str(e)}")
    
    def _clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        清洗数据
        
        处理：
        - 货币符号和逗号
        - 百分比
        - 空白字符
        
        Args:
            df: 原始数据框
        
        Returns:
            pd.DataFrame: 清洗后的数据框
        """
        df_cleaned = df.copy()
        
        # 处理Sales列（货币格式）
        if 'Sales' in df_cleaned.columns:
            try:
                df_cleaned['Sales'] = (
                    df_cleaned['Sales']
                    .astype(str)
                    .str.replace('$', '', regex=False)
                    .str.replace(',', '', regex=False)
                    .str.strip()
                )
                # 转换为数值，无效值变为NaN
                df_cleaned['Sales'] = pd.to_numeric(df_cleaned['Sales'], errors='coerce')
                print(f"清洗Sales列 (货币格式 -> 数值)")
            except Exception as e:
                print(f"Sales列清洗失败: {e}")
        
        # 处理Rating列（百分比格式）
        if 'Rating' in df_cleaned.columns:
            try:
                df_cleaned['Rating'] = (
                    df_cleaned['Rating']
                    .astype(str)
                    .str.replace('%', '', regex=False)
                    .str.strip()
                )
                # 转换为数值并除以100
                df_cleaned['Rating'] = pd.to_numeric(df_cleaned['Rating'], errors='coerce') / 100
                print(f"清洗Rating列")
            except Exception as e:
                print(f"Rating列清洗失败: {e}")
        
        for col in df_cleaned.select_dtypes(include=['object']).columns:
            df_cleaned[col] = df_cleaned[col].str.strip() if df_cleaned[col].dtype == 'object' else df_cleaned[col]
        
        df_cleaned = df_cleaned.dropna(how='all')
        
        return df_cleaned
    
    def _extract_metadata(self):
        """提取数据元信息（供LLM理解数据结构）"""
        if self.df is None:
            return
        
        self.metadata = {
            'file_path': self.file_path,
            'shape': self.df.shape,
            'columns': list(self.df.columns),
            'dtypes': {col: str(dtype) for col, dtype in self.df.dtypes.items()},
            'sample_data': self.df.head(5).to_dict('records'),
            'statistics': self._get_statistics(),
            'missing_values': self.df.isnull().sum().to_dict(),
            'unique_values': {
                col: self.df[col].nunique() 
                for col in self.df.columns
            }
        }
    
    def _get_statistics(self) -> Dict[str, Any]:
        """获取数据统计信息"""
        stats = {}
        
        # 数值列统计
        numeric_cols = self.df.select_dtypes(include=['number']).columns
        if len(numeric_cols) > 0:
            stats['numeric'] = self.df[numeric_cols].describe().to_dict()
        
        # 分类列统计
        categorical_cols = self.df.select_dtypes(include=['object']).columns
        if len(categorical_cols) > 0:
            stats['categorical'] = {}
            for col in categorical_cols:
                top_values = self.df[col].value_counts().head(5).to_dict()
                stats['categorical'][col] = {
                    'unique_count': self.df[col].nunique(),
                    'top_values': top_values
                }
        
        return stats
    
    def get_metadata(self) -> Dict[str, Any]:

        return self.metadata
    

# 测试代码
if __name__ == "__main__":
    # 测试加载器
    try:
        loader = CSVDataLoader("D:\Project\data.csv")
        df = loader.load()
        print(f"列类型: {loader.metadata['dtypes']}")
        print(f"缺失值: {loader.metadata['missing_values']}")
    except Exception as e:
        print(f"测试失败: {e}")

