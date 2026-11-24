"""Current Stage 5 entry point and bundle dependency regression checks."""
import ast
import contextlib
import importlib
import io
from pathlib import Path
import unittest
from scripts.stage5 import train_stage5 as pipeline

class RetiredStage4Tests(unittest.TestCase):
    def test_help_does_not_run_pipeline(self):
        with contextlib.redirect_stdout(io.StringIO()):
            with self.assertRaises(SystemExit) as caught:
                pipeline.main(['--help'])
        self.assertEqual(caught.exception.code, 0)

    def test_current_configs(self):
        source = Path('checkpoints/stage4/stage4_policy.pt')
        output = Path('fixture-output')
        b16 = pipeline.b16_config(source, output, 'cpu')
        paper = pipeline.paper_config(source, output, 'cpu')
        self.assertEqual(b16.stage4_checkpoint, source)
        self.assertEqual(paper.stage4_checkpoint, source)
        self.assertEqual(paper.policy_init_checkpoint, output / 'b16/last.pt')
        self.assertEqual(paper.value_checkpoint, output / 'value/best.pt')
        self.assertEqual((b16.baseline, b16.value_mode), ('zero', 'disabled'))
        self.assertEqual((paper.baseline, paper.value_mode), ('value', 'frozen'))
        self.assertIsNone(b16.resume_checkpoint)
        self.assertIsNone(paper.resume_checkpoint)

    def test_bundle_import_dependencies_and_imports(self):
        root = pipeline.REPO_ROOT
        paths = {line.strip() for line in (root / 'scripts/stage5/stage5_value_bundle_files.txt').read_text().splitlines()
                 if line.strip() and not line.startswith('#')}
        self.assertIn('scripts/stage5/train_stage5.py', paths)
        for relative in paths:
            self.assertTrue((root / relative).is_file(), relative)
            if not relative.endswith('.py'):
                continue
            tree = ast.parse((root / relative).read_text(encoding='utf-8'))
            for node in ast.walk(tree):
                if not isinstance(node, ast.ImportFrom) or node.level or not node.module:
                    continue
                if node.module.split('.')[0] not in {'games', 'models', 'training', 'scripts'}:
                    continue
                prefix = node.module.replace('.', '/')
                self.assertTrue(any(p in paths for p in (prefix + '.py', prefix + '/__init__.py')),
                                f'{relative} imports missing {node.module}')
            module = relative[:-3].replace('/', '.')
            if module.endswith('.__init__'):
                module = module[:-9]
            importlib.import_module(module)
