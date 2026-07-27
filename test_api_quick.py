#!/usr/bin/env python3
"""快速测试 DeepSeek API 连通性，结果写入 test_api_result.txt"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

result_file = os.path.join(os.path.dirname(__file__), "test_api_result.txt")

lines = []
try:
    import openai
    lines.append("✓ openai 库已安装")
    
    client = openai.OpenAI(
        api_key="sk-ae3a40051e534edfb2c892e7bfa87284",
        base_url="https://api.deepseek.com"
    )
    lines.append("✓ OpenAI 客户端初始化成功")
    
    resp = client.chat.completions.create(
        model="deepseek-coder",
        messages=[{"role": "user", "content": "Reply with exactly: OK"}],
        max_tokens=10,
        temperature=0.0,
    )
    answer = resp.choices[0].message.content.strip()
    lines.append(f"✓ API 调用成功，回复: {answer}")
    lines.append("✅ DeepSeek API 可用，可以开始正式运行！")
    
except ImportError as e:
    lines.append(f"❌ 缺少依赖: {e}")
    lines.append("  运行: pip install openai")
except Exception as e:
    lines.append(f"❌ API 错误: {e}")

with open(result_file, "w", encoding="utf-8") as f:
    f.write("\n".join(lines) + "\n")

print("\n".join(lines))
print(f"\n结果已写入: {result_file}")