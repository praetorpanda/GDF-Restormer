"""Standard-library-only static checks; does not import training dependencies or run training."""
import ast
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def assigned_name(node):
    if isinstance(node, ast.Assign) and isinstance(node.targets[0], ast.Name):
        return node.targets[0].id


def main():
    files = list(ROOT.rglob('*.py'))
    for path in files:
        compile(path.read_text(encoding='utf-8-sig'), str(path), 'exec')
    entries = json.loads((ROOT/'docs/training_entries.json').read_text(encoding='utf-8'))
    assert {e['model'] for e in entries} == {
        'Restormer_StrictGlobalDegField',
        'Restormer_PSFLikeDegField',
        'Restormer_GlobalPSFScaleRetinexWaveletMoE',
    }
    assert not list(ROOT.glob('run_*.py'))
    assert not (ROOT/'legacy').exists()
    assert {p.name for p in ROOT.glob('train_*.py')} == {e['train'] for e in entries}
    assert {p.name for p in (ROOT/'engines').glob('engine_*.py')} == {Path(e['engine']).name for e in entries}
    for entry in entries:
        runner = ROOT/entry['train']
        engine = ROOT/entry['engine']
        tree = ast.parse(runner.read_text(encoding='utf-8'))
        engine_tree = ast.parse(engine.read_text(encoding='utf-8'))
        imports = [n.module for n in ast.walk(tree) if isinstance(n, ast.ImportFrom)]
        assert 'engines.'+engine.stem in imports, runner
        assert runner.stem.removeprefix('train_') == engine.stem.removeprefix('engine_')
        functions = {n.name:n for n in engine_tree.body if isinstance(n, ast.FunctionDef)}
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in functions:
                fn = functions[node.func.id]
                if not fn.args.kwarg:
                    allowed = {a.arg for a in fn.args.args+fn.args.kwonlyargs}
                    assert {k.arg for k in node.keywords if k.arg} <= allowed, runner
            if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith('models.'):
                module = ROOT.joinpath(*node.module.split('.')).with_suffix('.py')
                model_tree = ast.parse(module.read_text(encoding='utf-8-sig'))
                names = {getattr(n, 'name', None) for n in model_tree.body}
                names.update(assigned_name(n) for n in model_tree.body)
                assert all(a.name in names for a in node.names), (runner, node.module)
        experiments = [n for n in tree.body if assigned_name(n) == 'EXPERIMENTS']
        if experiments:
            assert len(experiments[0].value.elts) == 1, runner
            exp = experiments[0].value.elts[0]
            fields = {k.value:v for k,v in zip(exp.keys, exp.values) if isinstance(k, ast.Constant)}
            assert fields['model_class'].id == entry['model']
            assert ast.literal_eval(fields['enabled']) is True
        variants = [n for n in engine_tree.body if assigned_name(n) == 'model_variants']
        if variants:
            assert len(variants[0].value.keys) == 1, engine
        result = subprocess.run([sys.executable, str(runner), '--help'], capture_output=True, text=True)
        assert result.returncode == 0, (runner, result.stderr)
    print(f'PASS: {len(files)} Python files compile; {len(entries)} model/engine pairs and CLI help verified.')
    print('Runtime dependencies, GPU execution and numerical reproducibility were not tested.')


if __name__ == '__main__':
    main()
