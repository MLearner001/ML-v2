import re

with open('phase2_experimental/inference_pipeline.py', 'r') as f:
    code = f.read()

code = code.replace('(Updated: V2 dengan Physics-Informed Hard-Clipping)',
                    '(Updated: V3 tanpa Hard-Clipping dan dengan fix bug inverse transformation)')

with open('phase2_experimental/inference_pipeline.py', 'w') as f:
    f.write(code)
