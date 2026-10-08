"""Read native benchmark annotations without importing inference dependencies."""
import ast
import csv
import json
import math
import re
from pathlib import Path

STREAMING_TASKS = ('OP', 'CR', 'CS', 'ATP', 'EU', 'TR', 'PR', 'SU', 'ACP', 'CT')
OVO_TASKS = ('OCR', 'ACR', 'ATR', 'STU', 'FPD', 'OJR')
TASK_NAMES = dict(zip(STREAMING_TASKS, (
    'Object Perception', 'Causal Reasoning', 'Clips Summarize',
    'Attribute Perception', 'Event Understanding', 'Text-Rich Understanding',
    'Prospective Reasoning', 'Spatial Understanding', 'Action Perception', 'Counting')))

def task_code(value):
    normalize = lambda s: re.sub(r'[^A-Z0-9]', '', s.upper())
    token = normalize(value)
    for code, name in TASK_NAMES.items():
        if token in (code, normalize(name)):
            return code
    if token in ('CLIPSUMMARIZATION', 'CLIPSSUMMARIZATION', 'CLIPSUMMARIZE'):
        return 'CS'
    raise ValueError(f'Unknown StreamingBench task: {value!r}')

def timestamp(value):
    if isinstance(value, bool):
        raise ValueError('Boolean timestamps are invalid')
    parts = str(value).strip().split(':')
    if not 1 <= len(parts) <= 3:
        raise ValueError(f'Invalid timestamp: {value!r}')
    numbers = [float(x) for x in parts]
    if any(not math.isfinite(x) or x < 0 for x in numbers):
        raise ValueError(f'Invalid timestamp: {value!r}')
    if len(numbers) > 1 and any(x >= 60 for x in numbers[1:]):
        raise ValueError(f'Invalid timestamp components: {value!r}')
    return sum(x * 60 ** i for i, x in enumerate(reversed(numbers)))

def read_records(path, benchmark):
    if benchmark == 'streamingbench':
        with Path(path).open(encoding='utf-8-sig', newline='') as f:
            rows = list(csv.DictReader(f))
        records = []
        for row in rows:
            qid = row['question_id'].strip()
            options = ast.literal_eval(row['options'])
            if not isinstance(options, list):
                raise ValueError(f'{qid}: options must be a list')
            # Preserve the original option strings in the benchmark prompt.
            match = re.search(r'(?:^|_)sample_([^_]+)(?:_|$)', qid)
            video = row.get('video', '').strip() or (
                f'sample_{match[1]}_real.mp4' if match else f'{qid}.mp4')
            records.append(dict(id=qid, video=video, timestamp=timestamp(row['time_stamp']),
                question=row['question'].strip(), options=options,
                answer=row['answer'].strip().upper(), task=task_code(row['task_type']),
                benchmark=benchmark))
        return records
    if benchmark == 'ovobench':
        rows = json.loads(Path(path).read_text(encoding='utf-8'))
        if not isinstance(rows, list):
            raise ValueError('OVO annotations must be a JSON array')
        records = []
        for row in rows:
            if row['task'] not in OVO_TASKS:
                continue  # The manuscript table contains only the six real-time tasks.
            records.append(dict(id=str(row['id']), video=row['video'],
                timestamp=timestamp(row['realtime']), question=row['question'].strip(),
                options=row['options'], answer=row['gt'], task=row['task'], benchmark=benchmark))
        return records
    if benchmark == 'custom':
        rows = json.loads(Path(path).read_text(encoding='utf-8'))
        if not isinstance(rows, list):
            raise ValueError('Custom annotations must be a JSON array')
        return [dict(r, benchmark='custom') for r in rows]
    raise ValueError(f'Unknown benchmark: {benchmark}')

def load_questions(path, video_root, benchmark, tasks=None, max_videos=0, max_questions=0):
    from memo.dataset import validate_records
    records = read_records(path, benchmark)
    if tasks:
        allowed = set(STREAMING_TASKS if benchmark == 'streamingbench' else OVO_TASKS)
        if not set(tasks) <= allowed:
            raise ValueError(f'Invalid task filter: {tasks}')
        records = [r for r in records if r.get('task') in tasks]
    groups = validate_records(records, video_root)
    if max_videos:
        groups = dict(list(groups.items())[:max_videos])
    if max_questions:
        groups = {v: qs[:max_questions] for v, qs in groups.items()}
    return groups

def prompt(q):
    if q['benchmark'] == 'streamingbench':
        return (q['question'] + '\n\nOptions:\n' + '\n'.join(q['options']) + '\n\n'
                'Please select the most appropriate answer and output the option letter only (e.g., A).')
    if q['benchmark'] == 'ovobench':
        options = '; '.join(f'{chr(65+i)}. {o}' for i, o in enumerate(q['options'])) + ';'
        return ('\nQuestion: ' + q['question'] + '\nOptions:\n' + options + '\n\n'
                'Respond only with the letter corresponding to your chosen option (e.g., A, B, C). \n'
                'Do not include any additional text or explanation in your response.\n')
    from memo.dataset import build_prompt
    return build_prompt(q)
