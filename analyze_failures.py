#!/usr/bin/env python3
"""Phase 1: 诊断失败模式"""
import json

r = json.load(open('E:/FPGA/project/FPT/evaluation_output/deepseek_api_full/agent_report.json'))
skipped = [t for t in r['per_task'] if t['final_action'] == 'skip']
print(f'Skipped tasks: {len(skipped)}')
print()

# 按失败类型分组
fail_types = {}
for t in skipped:
    s = t['best_status']
    key = f"comp={s['compilation']} sim={s['simulation']} synth={s['synthesis']}"
    fail_types.setdefault(key, []).append(t['task_id'])

print('=== 失败模式分布 ===')
for k, v in sorted(fail_types.items()):
    print(f'  {k}: {len(v)} tasks')
    for tid in v[:5]:
        print(f'    - {tid}')
    if len(v) > 5:
        print(f'    ... 还有 {len(v)-5} 个')

print()
print('=== 前15个跳过任务详情 ===')
for t in skipped[:15]:
    print(f"  {t['task_id']} attempts={t['attempts']} status={t['best_status']}")