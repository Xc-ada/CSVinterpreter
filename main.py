import os
import sys
import argparse
from colorama import Fore, Style, init
from modules import CSVDataLoader, LLMClient, ConversationManager

# 初始化colorama
init(autoreset=True)


def print_banner():
    banner = f"""
CSV数据分析助手

"""
    print(banner)


def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description='CSV数据分析系统'
    )
    
    parser.add_argument(
        '--csv',
        type=str,
    )
    
    parser.add_argument(
        '--api-key',
        type=str,
    )
    
    parser.add_argument(
        '--model',
        type=str,
        default='qwen-max',
    )
    
    parser.add_argument(
        '--max-retries',
        type=int,
        default=5,
    )
    
    return parser.parse_args()


def initialize_system(args):
    """初始化系统"""
    # 获取CSV路径
    if args.csv:
        csv_path = args.csv
    else:
        csv_path = input("请输入CSV文件路径: ").strip()
        if not csv_path:
            csv_path = "data.csv"
    
    # 检查文件是否存在
    if not os.path.exists(csv_path):
        print(f"错误: 文件不存在: {csv_path}")
        sys.exit(1)
    
    # 获取通义千问 API Key
    if args.api_key:
        api_key = args.api_key
    else:
        print("\n请输入通义千问API密钥:")
        api_key = input("API Key: ").strip()
    
    if not api_key:
        print("错误: 必须提供API Key")
        sys.exit(1)
    
    model = args.model
    print(f"使用模型: {model}")
    
    try:
        llm_client = LLMClient(
            api_key=api_key,
            model=model
        )
        
        # 初始化对话管理器
        manager = ConversationManager(
            csv_path=csv_path,
            llm_client=llm_client,
            max_retries=args.max_retries
        )
        
        return manager
    
    except Exception as e:
        print(f"初始化失败: {e}")
        sys.exit(1)


def main_loop(manager: ConversationManager):
    """主交互循环"""
    print(f"\n开始分析\n")
    
    while True:
        try:
            # 获取用户输入
            user_input = input(f"你: ").strip()
            
            if not user_input:
                continue
            
            # 处理数据分析问题
            answer = manager.chat(user_input)
            
            print(f"{Style.RESET_ALL}{answer}\n")
        
        except KeyboardInterrupt:
            print(f"\n\n检测到中断，正在退出...")
            break
        
        except Exception as e:
            print(f"\n发生错误: {e}")
            import traceback
            traceback.print_exc()
            
            # 询问是否继续
            cont = input(f"继续使用? (y/n): ").strip().lower()
            if cont != 'y':
                break


def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 打印横幅
    print_banner()
    
    # 初始化系统
    manager = initialize_system(args)
    
    # 进入主循环
    main_loop(manager)


if __name__ == "__main__":
    main()

