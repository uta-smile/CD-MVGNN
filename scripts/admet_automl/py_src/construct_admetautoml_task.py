#!/usr/bin/env python
# coding=utf-8
import yaml
import csv
import os
import re

sf = "scripts/admet_automl/tmp/"
os.makedirs(sf, exist_ok=True)
sy = yaml.load(open('scripts/admet_automl/admet_template.yml'))
source_csv = csv.DictReader(open('scripts/admet_automl/taskconfig.csv'))
sh_out = open("scripts/admet_automl/scripts/all_finetune_tasks.sh", 'w')
sh_out.write('#!/bin/bash\n')
sh_out.write('rm main.sh\n')
sh_out.write('rm seven.yml\n')
sh_out.write('cp scripts/admet_automl/scripts/main_finetune.sh main.sh\n')
sh_out.write('cp scripts/admet_automl/seven_fintune.yml seven.yml\n')

task_tag = 'dualmpnn_plus_dhid'
id = 0
p_set = set(['dataset', 'dataset_type', 'metric', 'split_type'])
n_dict = {"random": "RD",
          "scaffold_balanced": "SB"}
for r in source_csv:
    print(r)
    dataset = r["dataset"]
    metric = r["metric"]
    split_type = r['split_type']
    optimizationType = r['optimizationType'].strip()
    objectValueName = "overall_%s_test_%s" % (split_type, metric)
    cnt = 0

    sy['common']['studyName'] = "%s_%s_%s" % (task_tag, dataset, n_dict[split_type])
    sy['common']['optimizationType'] = optimizationType
    sy['common']['objectValueName'] = objectValueName
    sy['schedulerConfig']['trialNum'] = 200
    sy['schedulerConfig']['parallelNum'] = 20
    for param in sy['parameterConfigs']:
        k = param['name']
        if k in p_set:
            param['categoricalValues'] = [r[k]]
    print(sy['parameterConfigs'])
    # print(p_dict)
    # print(r)
    foutname = "%s_%s.yml" % (dataset, n_dict[split_type])
    fpath = os.path.join(sf, foutname)
    fout = open(fpath, 'w')
    print(yaml.dump(sy, fout))
    sh_out.write("automl run --conf %s\n" % (fpath))
    id += 1
sh_out.write("rm main.sh\n")
sh_out.write("rm seven.yml\n")
sh_out.close()
