import csv
import json
from pathlib import Path
import tempfile
import unittest

from memo.benchmarks.data import load_questions, prompt, timestamp
from memo.benchmarks.scoring import score_answer, summarize
from memo.configuration import DEFAULTS, load_config
from memo.dataset import validate_records
from memo.eval.frame_selection import dual_budget
from memo.scripts.summarize import validate_result

class ReleaseTests(unittest.TestCase):
    def setUp(self):
        self.temp=tempfile.TemporaryDirectory()
        self.root=Path(self.temp.name)
        (self.root/'sample_0_real.mp4').touch()
    def tearDown(self): self.temp.cleanup()
    def question(self,**kwargs):
        return dict(dict(id='q',video='sample_0_real.mp4',question='Color?',options=['Red','Blue'],
                         answer='A',timestamp=0.5,benchmark='custom'),**kwargs)
    def config(self,**kwargs):
        path=self.root/'config.json';path.write_text(json.dumps(kwargs));return load_config(path)
    def test_independent_budgets(self):
        result=dual_budget(list(range(100)),list(range(100,200)),8,8)
        self.assertEqual(len(result),16)
        self.assertEqual(sum(x<100 for x in result),8)
        self.assertEqual(sum(x>=100 for x in result),8)
        self.assertEqual(len(dual_budget([],list(range(100)),8,8)),8)
    def test_duplicate_ids_rejected(self):
        with self.assertRaises(ValueError): validate_records([self.question(),self.question()],self.root)
    def test_future_sort_and_end_question(self):
        qs=[self.question(id='end',timestamp=None),self.question(id='later',timestamp=2),self.question(id='first')]
        self.assertEqual([q['id'] for q in next(iter(validate_records(qs,self.root).values()))],['first','later','end'])
    def test_timestamp_validation(self):
        self.assertEqual(timestamp('01:02:03.5'),3723.5)
        for t in ('nan','-1','00:70',True):
            with self.assertRaises(ValueError): timestamp(t)
    def test_strict_and_legacy_scoring(self):
        q=self.question(benchmark='ovobench')
        self.assertEqual(score_answer(q,'A or B','strict'),(None,False))
        self.assertTrue(score_answer(q,'A or B','legacy')[1])
        self.assertEqual(score_answer(q,'Answer: A','strict'),('A',True))
    def test_macro_not_micro(self):
        records=[dict(task='OCR',correct=True)]*9+[dict(task='ACR',correct=False)]
        summary=summarize(records,'ovobench')
        self.assertEqual(summary['micro_accuracy_pct'],90)
        self.assertEqual(summary['macro_available_tasks_pct'],50)
        self.assertIsNone(summary['paper_average_pct'])
    def test_unknown_and_invalid_config(self):
        for config in ({'typo':1},{'top_k':0},{'history_frames':True},{'target_fps':float('nan')},{'retrieval_weights':[1,1]}):
            with self.assertRaises(ValueError): self.config(**config)
    def test_ovo_realtime_subset(self):
        records=[dict(id=1,task='ATR',video='sample_0_real.mp4',realtime=.5,question='Color?',options=['Red','Blue'],gt=0),
                 dict(id=2,task='EPM')]
        p=self.root/'ovo.json';p.write_text(json.dumps(records))
        groups=load_questions(p,self.root,'ovobench')
        q=next(iter(groups.values()))[0]
        self.assertEqual(q['answer'],'A');self.assertIn('A. Red; B. Blue;',prompt(q))
    def test_csv_safe_literal_and_option_labels(self):
        p=self.root/'input.csv'
        def write(options):
            with p.open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=['question_id','time_stamp','question','options','answer','task_type'])
                w.writeheader();w.writerow(dict(question_id='qa_sample_0_1',time_stamp='00:01',question='Color?',options=options,answer='A',task_type='Clips Summarize'))
        write("['A. Red', 'B. Blue']")
        q=next(iter(load_questions(p,self.root,'streamingbench').values()))[0]
        self.assertEqual(q['task'],'CS');self.assertNotIn('A. A.',prompt(q))
        write("__import__('os').getcwd()")
        with self.assertRaises(ValueError): load_questions(p,self.root,'streamingbench')
    def test_incomplete_output_rejected(self):
        for data in ({'status':'failed'},dict(status='completed',results=[],expected_questions=1)):
            with self.assertRaises(ValueError): validate_result(data)
    def test_all_shipped_configs_validate(self):
        root=Path(__file__).resolve().parents[2]/'configs'
        for p in root.rglob('*.json'):
            if p.name!='paper_targets.json': load_config(p)

if __name__=='__main__': unittest.main()
