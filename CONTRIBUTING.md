# 开发规范
## 代码风格
 - Python: 采用 `PEP8`,请提交前使用pylint检查，如果使用Pycharm请使用相关插件。合并到主分支的代码必须符合代码规范。
 - [公司的python代码规范](https://git.code.oa.com/standards/code/tree/master/Python)
 
## 单元测试
  - 使用JUnit作为单元测试模块。 如果使用Pycharm可以在对应的类别上右键->`GOTO`->`Test` 创建新的单元测试，在没有特殊情况下，提交代码单元测试不超过`90%`不能合并到主分支。所有没完成单元测试的代码放在`contrib`文件夹。 
  - 单元测试文件以`test_Classname.py`命名，若是单个的函数，不属于任何方法，则统一以`test_Filename.py`命名，所有测试文件放在对应目录`test`文件夹下。
  - 所有新加入没经过单元测试的代码放在`contrib`内。
  - 旧的模块如果要进行修改和重构，必须先在当前状态下补充好单元测试，确保和原来一致才进行重构。
  - 重构的时候需要移动对应的单元测试文件到新的位置的`test`文件夹下。

## 注释
  使用`reStructuredText` 作为函数和类说明文档标准注释格式（Pycharm自带）。格式参考：[QuickRef](http://docutils.sourceforge.net/docs/user/rst/quickref.html)

## 模型添加Best Practice
  - 将现有应用迁移到`contrib`文件夹下（新建一个文件夹来放置代码），并确保可以顺利跑通。
  - 将应用中模型部分（nn.Module）迁移到`contrib/models`下。
  - 根据实际模型抽象出不同层次的基本操作层(layers)和中间件层(modules)。（请先阅读DESGIN.md了解这两个层的抽象原则，如果遇到困难请
  在群内提出进行讨论。）
  - 查看`models`下对应的`layers.py`和`modules.py`，看其中组件是否可以在当前模型中复用，如果可以则尝试重构替换为对应模型。
  - 在完成上一步重构后，将`layers.py`和`modules.py`中不包含的基本操作和中间件迁移到`layers.py`和`modules.py`, **并且完成相应文档和单元测试，提交合并主分支。**
  
 
## 附录
### 如何使用Pycharm连接Devnet机器进行远程调试
http://km.oa.com/group/37005/articles/show/397181?kmref=search&from_page=1&no=1
### 如何使用Pycharm的pylint插件进行代码风格检查。
https://www.cnblogs.com/gaowengang/p/7892661.html
### 如何使用Pycharm创建单元测试。
https://blog.csdn.net/luopotaotao/article/details/89641049
 