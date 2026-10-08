"""Explicit benchmark aggregation; never substitute paper targets for measurements."""
from collections import defaultdict
from memo.dataset import parse_choice
from memo.benchmarks.data import STREAMING_TASKS, OVO_TASKS

def score_answer(q, response, protocol):
    choice = parse_choice(response, q['options']) if q['options'] else None
    if not q['options'] or q['answer'] is None:
        return choice, None
    if protocol == 'strict':
        return choice, choice == q['answer']
    if q['benchmark'] == 'ovobench':
        # Match the provided original local OVO scorer, including its permissive behavior.
        return choice, q['answer'] in response
    if q['benchmark'] == 'streamingbench':
        text = response.strip().upper()
        return choice, text == q['answer'] or text.startswith(q['answer'])
    return choice, choice == q['answer']

def summarize(results, benchmark):
    buckets = defaultdict(list)
    for r in results:
        if r.get('correct') is not None:
            buckets[r.get('task', 'custom')].append(bool(r['correct']))
    scored = [v for values in buckets.values() for v in values]
    tasks = STREAMING_TASKS if benchmark == 'streamingbench' else OVO_TASKS if benchmark == 'ovobench' else tuple(sorted(buckets))
    per_task = {task: {'count': len(buckets[task]), 'correct': sum(buckets[task]),
                       'accuracy_pct': 100 * sum(buckets[task]) / len(buckets[task]) if buckets[task] else None}
                for task in tasks}
    micro = 100 * sum(scored) / len(scored) if scored else None
    available = [v['accuracy_pct'] for v in per_task.values() if v['count']]
    macro = sum(available) / len(available) if available else None
    complete = bool(tasks) and all(per_task[t]['count'] for t in tasks)
    return dict(answered=len(results), scored=len(scored), per_task=per_task,
                micro_accuracy_pct=micro, macro_available_tasks_pct=macro,
                all_tasks_present=complete,
                paper_average_pct=(macro if benchmark == 'ovobench' else micro) if complete else None)
