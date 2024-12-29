import os

root = '/data/wjdu/aware/output'

text = ""
models = os.listdir(root)
for model in models:
    if model in ['UniTS_HEAD', 'UniTS_HEAD_1']:
        continue
    configs = os.listdir(os.path.join(root, model))
    for config in configs:
        model_path = os.path.join(root, model, config)
        # if 'th' in config:
        #     data_path = '/home/wjdu/aware/data/eval/th.yaml'
        # elif 'wr' in config:
        #     data_path = '/home/wjdu/aware/data/eval/wr.yaml'
        # elif 'all' in config:
        #     data_path = '/home/wjdu/aware/data/eval/all.yaml'
        # else:
        #     assert False
        if 's_th' not in config and 's_wr' not in config:
            continue
        data_path = '/home/wjdu/aware/data/eval/all.yaml'
        output_path = os.path.join('/data/wjdu/aware/', 'result1', model, config)
        os.makedirs(output_path, exist_ok=True)
        text += f"python infer.py -l {model_path} -d {data_path} -o {output_path} > {output_path}/output.log \n"
        text += f"python eval.py {output_path} > {output_path}/output_still.log \n"

open('batch_infer.sh', 'w').write(text)