# `DGLT` 设计文档
## 目录结构
```
dglt
└───model
│   │   layers.py
│   │   modules.py
│   └───zoo
│       │   model1.py
│       │   model2.py
│       │   ...
│   
└───data
│   └───transformer
│   └───featurization
|   └───dataset
│    
└───train
│   └───prediction
│   └───generation
│   │   ...
│
└───contrib
└───scrpits
└───docs

```
其中：
 - `model`: 基础模型目录。所有模型相关代码放在这个文件夹。
   - `layers.py`: 基础模型，例如`Dense`, `GCN`等。`forward`函数输入为基本数据类型。需要对公共模块做出比较好的抽象。
   - `modules.py`: 基于基础模型构造的中间件模型，例如`MPNN encoder`。`forward`函数为复杂数据类型。任何一个`module`都可以同
   、`Dataloader`的输入配合得到`forward`输出。不允许在`modules`添加任何`loss function`以及其他非标准公开函数，可以添加私有函数。
   此外，`modules.py`的`forward`输入必须要明确，并且`forward`函数内根据不同的输入对应不同的分支，这种情况应该写成两个中间件。
   - `zoo`: 具体模型，按照每个模型一个`py`文件组织。通过组合`layers`和`modules`组合成模型，其包含`loss`函数。可以通过实现
   `get_loss_func`回调函数来实现获取模型loss的功能，在这个文件夹的模型可以添加任意自定义函数，比如上例举例的`get_loss_func`。
 - `data`: 数据模块
   - `dataset`：数据集模块。（继承pytorch Dataset）
        - `MolecularDataset.py`:属性预测的小分子小分子数据集
        - `LigandProteinDataset.py`:虚拟筛选用的配体·蛋白质数据集
        - `SmilesSequence.py`:分子生成的数据集
   - `transformer`: 数据变换模块，提供数据变换、分割、batch相关功能。
        - `MolCollator.py`：提供batch功能(作为pytorch Dataloader的collate_fn的参数，将pytorch Dataset __getitem__()的返回值组成Batch)
   - `featurization`: 对于不能直接用于模型的数据，提供特征提取功能。 例如，SMILES。（待定）
        - `mol2graph.py`: 将smiles转成图
        - `mol2RdkitFeature.py`: 将smiles转成Rdkit feature
        - `mol2PadelFeature.py`:将smiles转成Padel feature
        - `seqEmebeding.py`:Embedding蛋白质序列
 - `train`: 模型训练（以及可能的测试）相关代码。其主要是用于满足不同任务的训练测试需求，可以有比较大的自由度。例如AMDET的预测为一个Task，风格迁移为另外一个Task。
 - `contrib`: 测试模块，未完成单元测试的模块。文件夹组织结构同`dglt`
 - `scripts`: 一些脚本，例如automl脚本，seven运行脚本，机智脚本等。
 - `docs`: API文档


 