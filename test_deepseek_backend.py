#!/usr/bin/env python3
"""测试 DeepSeek API 后端配置"""

import sys
from pathlib import Path

# 添加项目路径
sys.path.insert(0, str(Path(__file__).parent))

try:
    # 测试导入
    from bench4hls.backends import DeepSeekAPIBackend
    print("✓ DeepSeekAPIBackend 导入成功")
    
    # 测试配置加载
    from bench4hls.settings import load_config
    cfg = load_config(Path(__file__).parent)
    print(f"✓ 配置文件加载成功")
    print(f"  - backend: deepseek_api")
    print(f"  - model: {cfg.deepseek_api_model}")
    print(f"  - api_key: {cfg.deepseek_api_key[:10]}...")
    
    # 测试后端初始化（不实际调用 API）
    backend = DeepSeekAPIBackend(
        api_key=cfg.deepseek_api_key,
        model=cfg.deepseek_api_model,
        max_new_tokens=cfg.max_new_tokens,
        temperature=cfg.temperature,
    )
    print(f"✓ DeepSeekAPIBackend 初始化成功")
    print(f"  - model: {backend.model}")
    print(f"  - max_tokens: {backend.max_new_tokens}")
    print(f"  - temperature: {backend.temperature}")
    
    print("\n✅ 所有测试通过！DeepSeek API 后端配置正确")
    print("\n使用方法:")
    print("  python bench4hls_runner.py --backend deepseek_api [其他选项]")
    
except Exception as e:
    print(f"\n❌ 测试失败: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)