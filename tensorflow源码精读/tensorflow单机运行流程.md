

总结一下单机环境下，在session run之前的代码流程


# 1. 通过Python client创建图
在import tensorflow的时候，Python就会创建一个默认图，创建图的代码在tensorflow/python/framework/c_api_util.py

代码为
```py
class ScopedTFGraph(object):
  """Wrapper around TF_Graph that handles deletion."""

  __slots__ = ["graph", "deleter"]

  def __init__(self):
    self.graph = c_api.TF_NewGraph()
    # Note: when we're destructing the global context (i.e when the process is
    # terminating) we may have already deleted other modules. By capturing the
    # DeleteGraph function here, we retain the ability to cleanly destroy the
    # graph at shutdown, which satisfies leak checkers.
    self.deleter = c_api.TF_DeleteGraph

  def __del__(self):
    self.deleter(self.graph)
```

其中ScopedTFGraph类的调用在tensorflow/python/framework/ops.py中，对象

```py
class Graph(object):
    def __init__(self):
        ...
        self._scoped_c_graph = c_api_util.ScopedTFGraph()
```

而c_api.TF_NewGraph()我们在graph中有过介绍。应该注意的是c_api_util.ScopedTFGraph() 和 class Graph(object)生成不是一类对象，c_api_util.ScopedTFGraph()生成对象是tensorflow.python._pywrap_tf_session.TF_Graph，而class Graph(object)生成的是tensorflow.python.framework.ops.Graph


# 2. Python和c++代码的连接

从2020年开始，tensorflow的Python和c++的代码连接是通过pybind实现的，而不是swig了(https://github.com/tensorflow/community/blob/master/rfcs/20190208-pybind11.md#replace-swig-with-pybind11)。网上很多博客说python和c++的代码连接靠的是swig，是基于比较旧版本的tensorflow。swig和pybind本人都没有用过，具体优劣并不熟悉，有兴趣可以看这个项目(https://github.com/UlovHer/PythonCallCpp)，而且这也不是我们学习tensorflow的重点。

python和c++的代码连接比较简单，c++的代码编译成一系列的.so文件，这些.so文件都放在{tensorflow安装路径}/python中(在不同的tensorflow版本中这个路径可能略有不同，但是一般都在/python或者其子目录下。

在Python代码中会利用pybind11导入.so文件，例如在上面的例子中。首先在tensorflow\tensorflow\python\client\pywrap_tf_session.py中通过如下语句导入了_pywrap_tf_session.so。
```py
from tensorflow.python.client._pywrap_tf_session import *
from tensorflow.python.client._pywrap_tf_session import _TF_SetTarget
from tensorflow.python.client._pywrap_tf_session import _TF_SetConfig
from tensorflow.python.client._pywrap_tf_session import _TF_NewSessionOptions
```

然后在tensorflow/python/framework/c_api_util.py利用如下语句，再次导入了

```py
from tensorflow.core.framework import api_def_pb2
from tensorflow.core.framework import op_def_pb2
from tensorflow.python.client import pywrap_tf_session as c_api
from tensorflow.python.util import compat
from tensorflow.python.util import tf_contextlib
```


_pywrap_tf_session.so的定义在tensorflow\python\client\tf_session_wrapper.cc，相关的源码如下：

```cpp
#include "tensorflow/c/c_api.h"
...
PYBIND11_MODULE(_pywrap_tf_session, m) {
  ...
    m.def("TF_NewGraph", TF_NewGraph, py::return_value_policy::reference,
        py::call_guard<py::gil_scoped_release>());
  ...
}

```
上面的代码中，PYBIND11_MODULE(_pywrap_tf_session, m)规定了.so的包名，m.def定义了这个包里有哪些python的函数。早python中只需要导入这个包，然后调用这个函数就可以了。

op_def_library.py 中有op_def_library.apply_op

apply_op调用_apply_op_helper

_apply_op_helper调用_GetOpDef 获取 opdef， g, producer

然后调用g._create_op_internal
  

# 3. 创建node
