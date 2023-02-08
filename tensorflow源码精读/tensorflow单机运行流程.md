

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

下面以matmul为例展示tensorflow如何从Python端的代码在后端图上创建一个节点。

在python上运行matmul函数时，调用的是/Users/shuangzhu.kang/opt/anaconda3/lib/python3.8/site-packages/tensorflow/python/ops/gen_math_ops.py 中的 mat_mul 函数,核心的代码如下

```py
from tensorflow.python.framework import op_def_library as _op_def_library
_op_def_library._apply_op_helper( "MatMul", a=a, b=b, transpose_a=transpose_a, transpose_b=transpose_b,
                  name=name)
```
## _apply_op_helper
_apply_op_helper的源码如下

```py
def _apply_op_helper(op_type_name, name=None, **keywords):  # pylint: disable=invalid-name
  """Implementation of apply_op that returns output_structure, op."""

  op_def, g, producer = _GetOpDef(op_type_name, keywords)
  name = name if name else op_type_name

  attrs, attr_protos = {}, {}
  default_type_attr_map, allowed_list_attr_map = {}, {}
  inputs, input_types, output_structure = [], [], []
  fallback = True

  if (_CanExtractAttrsFastPath(op_def, keywords) and
      flags.config().graph_building_optimization.value()):
    fallback = False
    attr_protos, inputs, input_types, output_structure = (
        op_def_library_pybind.process_inputs(op_type_name, producer, keywords))

  if fallback:
    _CheckOpDeprecation(op_type_name, op_def, producer)
    _ExtractDefaultTypesAndAllowedTypes(op_def, default_type_attr_map,
                                        allowed_list_attr_map)

  # Requires that op_def has passed validation (using the C++
  # ValidateOpDef() from ../framework/op_def_util.h).
  with g.as_default(), ops.name_scope(name) as scope:
    if fallback:
      _ExtractInputsAndAttrs(op_type_name, op_def, allowed_list_attr_map,
                             keywords, default_type_attr_map, attrs, inputs,
                             input_types)
      _ExtractRemainingAttrs(op_type_name, op_def, keywords,
                             default_type_attr_map, attrs)
      _ExtractAttrProto(op_type_name, op_def, attrs, attr_protos)
      del attrs  # attrs is no longer authoritative, use attr_protos instead
      _ExtractOutputStructure(op_type_name, op_def, attr_protos,
                              output_structure)
      _CheckAllInputsUsed(op_type_name, keywords)

    # NOTE(mrry): We add an explicit colocation constraint between
    # the newly created op and any of its reference-typed inputs.
    must_colocate_inputs = [val for arg, val in zip(op_def.input_arg, inputs)
                            if arg.is_ref]
    with _MaybeColocateWith(must_colocate_inputs):
      # Add Op to graph
      # pylint: disable=protected-access
      op = g._create_op_internal(op_type_name, inputs, dtypes=None,
                                 name=scope, input_types=input_types,
                                 attrs=attr_protos, op_def=op_def)

    # `outputs` is returned as a separate return value so that the output
    # tensors can the `op` per se can be decoupled so that the
    # `op_callbacks` can function properly. See framework/op_callbacks.py
    # for more details.
    outputs = op.outputs
    # Conditionally invoke tfdbg v2's op callback(s).
    if op_callbacks.should_invoke_op_callbacks():
      callback_outputs = op_callbacks.invoke_op_callbacks(
          op.node_def.op, tuple(op.inputs), attr_protos, tuple(outputs),
          op_name=op.name, graph=g)
      if callback_outputs is not None:
        outputs = callback_outputs

    return output_structure, op_def.is_stateful, op, outputs

```
我们抛开各种复杂的逻辑判断，核心代码就两行
```cpp
op_def, g, producer = _GetOpDef(op_type_name, keywords)
op = g._create_op_internal(op_type_name, inputs, dtypes=None,
                                 name=scope, input_types=input_types,
                                 attrs=attr_protos, op_def=op_def)
```
### _GetOpDef
其中_GetOpDef的代码如下

```py
from tensorflow.python.framework import op_def_registry
def _GetOpDef(op_type_name, keywords):
  """Returns the OpDef, Graph and Producer. For use in _apply_op_helper."""
  op_def = op_def_registry.get(op_type_name)
  if op_def is None:
    raise RuntimeError(f"Unrecognized Op name {op_type_name}")

  # Determine the graph context.
  try:
    # Need to flatten all the arguments into a list.
    # pylint: disable=protected-access
    g = ops._get_graph_from_inputs(_Flatten(keywords.values()))
    producer = g.graph_def_versions.producer
    # pylint: enable=protected-access
  except AssertionError as e:
    raise RuntimeError(
        f"Cannot determine graph for Op '{op_type_name}' due to: {e.message}")

  return op_def, g, producer
```

_GetOpDef函数做了两件事：检测op是否被注册，获取当前的图。 获取当前的图不介绍了


#### _op_def_registry
检测op是否被注册用的是op_def_registry.get(op_type_name)，即调用了op_def_registry文件中的get函数，这个函数代码是：

```py
from tensorflow.python.framework import _op_def_registry
serialized_op_def = _op_def_registry.get(name)
```
这两行代码也是一个典型的python调用c++的过程，首先导入_op_def_registry包，然后调用_op_def_registry中的get函数。而get函数的定义在tensorflow/python/framework/op_def_registry.cc。就是从op的注册信息OpRegistry中取出相应的op。

```cpp
PYBIND11_MODULE(_op_def_registry, m) {
  m.def("get", [](const std::string& name) {
    const tensorflow::OpDef* op_def = nullptr;
    auto status = tensorflow::OpRegistry::Global()->LookUpOpDef(name, &op_def);
    if (!status.ok()) return py::reinterpret_borrow<py::object>(py::none());

    tensorflow::OpDef stripped_op_def = *op_def;
    tensorflow::RemoveNonDeprecationDescriptionsFromOpDef(&stripped_op_def);

    tensorflow::MaybeRaiseFromStatus(status);
    std::string serialized_op_def;
    if (!stripped_op_def.SerializeToString(&serialized_op_def)) {
      throw std::runtime_error("Failed to serialize OpDef to string");
    }

    // Explicitly convert to py::bytes because std::string is implicitly
    // convertable to py::str by default.
    return py::reinterpret_borrow<py::object>(py::bytes(serialized_op_def));
  });
}
```


## g._create_op_internal

_create_op_internal在tensorflow/python/framework/ops.py，源码如下
```py
  def _create_op_internal(
      self,
      op_type,
      inputs,
      dtypes=None,  # pylint: disable=redefined-outer-name
      input_types=None,
      name=None,
      attrs=None,
      op_def=None,
      compute_device=True):
    """Creates an `Operation` in this graph.

    Implements `Graph.create_op()` without the overhead of the deprecation
    wrapper.

    Args:
      op_type: The `Operation` type to create. This corresponds to the
        `OpDef.name` field for the proto that defines the operation.
      inputs: A list of `Tensor` objects that will be inputs to the `Operation`.
      dtypes: (Optional) A list of `DType` objects that will be the types of the
        tensors that the operation produces.
      input_types: (Optional.) A list of `DType`s that will be the types of the
        tensors that the operation consumes. By default, uses the base `DType`
        of each input in `inputs`. Operations that expect reference-typed inputs
        must specify `input_types` explicitly.
      name: (Optional.) A string name for the operation. If not specified, a
        name is generated based on `op_type`.
      attrs: (Optional.) A dictionary where the key is the attribute name (a
        string) and the value is the respective `attr` attribute of the
        `NodeDef` proto that will represent the operation (an `AttrValue`
        proto).
      op_def: (Optional.) The `OpDef` proto that describes the `op_type` that
        the operation will have.
      compute_device: (Optional.) If True, device functions will be executed to
        compute the device property of the Operation.

    Raises:
      ValueError: if colocation conflicts with existing device assignment.

    Returns:
      An `Operation` object.
    """
    self._check_not_finalized()
    if name is None:
      name = op_type
    # If a names ends with a '/' it is a "name scope" and we use it as-is,
    # after removing the trailing '/'.
    if name and name[-1] == "/":
      name = name_from_scope_name(name)
    else:
      name = self.unique_name(name)

    node_def = _NodeDef(op_type, name, attrs)

    input_ops = set(t.op for t in inputs)
    control_inputs = self._control_dependencies_for_inputs(input_ops)
    # _create_op_helper mutates the new Operation. `_mutation_lock` ensures a
    # Session.run call cannot occur between creating and mutating the op.
    with self._mutation_lock():
      ret = Operation(
          node_def,
          self,
          inputs=inputs,
          output_types=dtypes,
          control_inputs=control_inputs,
          input_types=input_types,
          original_op=self._default_original_op,
          op_def=op_def)
      self._create_op_helper(ret, compute_device=compute_device)
    return ret
```
上面的代码，核心的的也只有两行

```py
ret = Operation(
    node_def,
    self,
    inputs=inputs,
    output_types=dtypes,
    control_inputs=control_inputs,
    input_types=input_types,
    original_op=self._default_original_op,
    op_def=op_def)
self._create_op_helper(ret, compute_device=compute_device)
```
我们一一拆解

### Operation

Operation的代码在tensorflow/python/framework/ops.py, 源码如下
```py
class Operation(object):
  """Represents a graph node that performs computation on tensors.

  An `Operation` is a node in a `tf.Graph` that takes zero or more `Tensor`
  objects as input, and produces zero or more `Tensor` objects as output.
  Objects of type `Operation` are created by calling a Python op constructor
  (such as `tf.matmul`) within a `tf.function` or under a `tf.Graph.as_default`
  context manager.

  For example, within a `tf.function`, `c = tf.matmul(a, b)` creates an
  `Operation` of type "MatMul" that takes tensors `a` and `b` as input, and
  produces `c` as output.

  If a `tf.compat.v1.Session` is used, an `Operation` of a `tf.Graph` can be
  executed by passing it to `tf.Session.run`. `op.run()` is a shortcut for
  calling `tf.compat.v1.get_default_session().run(op)`.
  """

  def __init__(self,
               node_def,
               g,
               inputs=None,
               output_types=None,
               control_inputs=None,
               input_types=None,
               original_op=None,
               op_def=None):
    r"""Creates an `Operation`.

    NOTE: This constructor validates the name of the `Operation` (passed
    as `node_def.name`). Valid `Operation` names match the following
    regular expression:

        [A-Za-z0-9.][A-Za-z0-9_.\\-/]*

    Args:
      node_def: `node_def_pb2.NodeDef`.  `NodeDef` for the `Operation`. Used for
        attributes of `node_def_pb2.NodeDef`, typically `name`, `op`, and
        `device`.  The `input` attribute is irrelevant here as it will be
        computed when generating the model.
      g: `Graph`. The parent graph.
      inputs: list of `Tensor` objects. The inputs to this `Operation`.
      output_types: list of `DType` objects.  List of the types of the `Tensors`
        computed by this operation.  The length of this list indicates the
        number of output endpoints of the `Operation`.
      control_inputs: list of operations or tensors from which to have a control
        dependency.
      input_types: List of `DType` objects representing the types of the tensors
        accepted by the `Operation`.  By default uses `[x.dtype.base_dtype for x
        in inputs]`.  Operations that expect reference-typed inputs must specify
        these explicitly.
      original_op: Optional. Used to associate the new `Operation` with an
        existing `Operation` (for example, a replica with the op that was
        replicated).
      op_def: Optional. The `op_def_pb2.OpDef` proto that describes the op type
        that this `Operation` represents.

    Raises:
      TypeError: if control inputs are not Operations or Tensors,
        or if `node_def` is not a `NodeDef`,
        or if `g` is not a `Graph`,
        or if `inputs` are not tensors,
        or if `inputs` and `input_types` are incompatible.
      ValueError: if the `node_def` name is not valid.
    """
    if not isinstance(g, Graph):
      raise TypeError(f"Argument g must be a Graph. "
                      f"Received an instance of type {type(g)}")

    # TODO(feyu): This message is redundant with the check below. We raise it
    # to help users to migrate. Remove this after 07/01/2022.
    if isinstance(node_def, pywrap_tf_session.TF_Operation):
      raise ValueError(
          "Calling Operation() with node_def of a TF_Operation is deprecated. "
          "Please switch to Operation.from_c_op.")

    if not isinstance(node_def, node_def_pb2.NodeDef):
      raise TypeError(f"Argument node_def must be a NodeDef. "
                      f"Received an instance of type: {type(node_def)}.")
    if node_def.ByteSize() >= (1 << 31) or node_def.ByteSize() < 0:
      raise ValueError(
          f"Cannot create a tensor proto whose content is larger than 2GB. "
          f"Size of tensor is {node_def.ByteSize()} bytes.")

    # TODO(mdan): This does not belong here. Graph::AddNode should handle it.
    if not _VALID_OP_NAME_REGEX.match(node_def.name):
      raise ValueError(
          f"`{node_def.name}` is not a valid node name. "
          f"Accepted names conform to Regex /{_VALID_OP_NAME_REGEX}/")

    # FIXME(b/225400189): output_types is unused. Consider remove it from
    # the argument list.
    del output_types

    if inputs is None:
      inputs = []
    elif not isinstance(inputs, list):
      raise TypeError(f"Argument inputs shall be a list of Tensors. "
                      f"Received an instance of type {type(inputs)}")
    for a in inputs:
      if not isinstance(a, Tensor):
        raise TypeError(f"Items of argument inputs shall be Tensor. "
                        f"Received an instance of type {type(a)}.")
    if input_types is None:
      input_types = [i.dtype.base_dtype for i in inputs]
    else:
      if not all(
          x.is_compatible_with(i.dtype) for i, x in zip(inputs, input_types)):
        raise TypeError("In op '%s', input types (%s) are not compatible "
                        "with expected types (%s)" %
                        (node_def.name, [i.dtype for i in inputs], input_types))

    # Build the list of control inputs.
    control_input_ops = []
    if control_inputs:
      for c in control_inputs:
        control_op = None
        if isinstance(c, Operation):
          control_op = c
        elif isinstance(c, (Tensor, IndexedSlices)):
          control_op = c.op
        else:
          raise TypeError(f"Control input must be an Operation, "
                          f"a Tensor, or IndexedSlices. "
                          f"Received an instance of type {type(c)}.")
        control_input_ops.append(control_op)

    # Initialize c_op from node_def and other inputs
    c_op = _create_c_op(g, node_def, inputs, control_input_ops, op_def=op_def)
    self._init_from_c_op(c_op=c_op, g=g)

    self._original_op = original_op

    # Post process for control flows.
    self._control_flow_post_processing(input_tensors=inputs)

    # Removes this frame from the Python traceback.
    # We adjust stacklevel directly to avoid triggering serialization.
    self.traceback._stacklevel += 1  # pylint: disable=protected-access
```
这个代码是创建一个Operation对象，核心的代码是

```py
    c_op = _create_c_op(g, node_def, inputs, control_input_ops, op_def=op_def)
    self._init_from_c_op(c_op=c_op, g=g)

    self._original_op = original_op

    # Post process for control flows.
    self._control_flow_post_processing(input_tensors=inputs)
```
_create_c_op、_init_from_c_op一一介绍

#### _create_c_op

```py
_create_c_op在tensorflow/python/framework/ops.py
def _create_c_op(graph,
                 node_def,
                 inputs,
                 control_inputs,
                 op_def=None,
                 extract_traceback=True):
  """Creates a TF_Operation.

  Args:
    graph: a `Graph`.
    node_def: `node_def_pb2.NodeDef` for the operation to create.
    inputs: A flattened list of `Tensor`s. This function handles grouping
      tensors into lists as per attributes in the `node_def`.
    control_inputs: A list of `Operation`s to set as control dependencies.
    op_def: Optional. `op_def_pb2.OpDef` for the operation to create. If not
      specified, is looked up from the `graph` using `node_def.op`.
    extract_traceback: if True, extract the current Python traceback to the
      TF_Operation.

  Returns:
    A wrapped TF_Operation*.
  """
  if op_def is None:
    op_def = graph._get_op_def(node_def.op)  # pylint: disable=protected-access
  # TODO(skyewm): op_def_library.apply_op() flattens the incoming inputs.
  # Refactor so we don't have to do this here.
  inputs = _reconstruct_sequence_inputs(op_def, inputs, node_def.attr)
  # pylint: disable=protected-access
  op_desc = pywrap_tf_session.TF_NewOperation(graph._c_graph,
                                              compat.as_str(node_def.op),
                                              compat.as_str(node_def.name))
  if node_def.device:
    pywrap_tf_session.TF_SetDevice(op_desc, compat.as_str(node_def.device))
  # Add inputs
  for op_input in inputs:
    if isinstance(op_input, (list, tuple)):
      pywrap_tf_session.TF_AddInputList(op_desc,
                                        [t._as_tf_output() for t in op_input])
    else:
      pywrap_tf_session.TF_AddInput(op_desc, op_input._as_tf_output())

  # Add control inputs
  for control_input in control_inputs:
    pywrap_tf_session.TF_AddControlInput(op_desc, control_input._c_op)
  # pylint: enable=protected-access

  # Add attrs
  for name, attr_value in node_def.attr.items():
    serialized = attr_value.SerializeToString()
    # TODO(skyewm): this creates and deletes a new TF_Status for every attr.
    # It might be worth creating a convenient way to re-use the same status.
    pywrap_tf_session.TF_SetAttrValueProto(op_desc, compat.as_str(name),
                                           serialized)

  try:
    c_op = pywrap_tf_session.TF_FinishOperation(op_desc)
  except errors.InvalidArgumentError as e:
    # Convert to ValueError for backwards compatibility.
    raise ValueError(e.message)

  # Record the current Python stack trace as the creating stacktrace of this
  # TF_Operation.
  if extract_traceback:
    tf_stack.extract_stack_for_op(c_op, stacklevel=3)

  return c_op
```
这个函数中，比较核心的是

```py
  op_desc = pywrap_tf_session.TF_NewOperation(graph._c_graph,
                                              compat.as_str(node_def.op),
                                              compat.as_str(node_def.name))
  if node_def.device:
    pywrap_tf_session.TF_SetDevice(op_desc, compat.as_str(node_def.device))
  # Add inputs
  for op_input in inputs:
    if isinstance(op_input, (list, tuple)):
      pywrap_tf_session.TF_AddInputList(op_desc,
                                        [t._as_tf_output() for t in op_input])
    else:
      pywrap_tf_session.TF_AddInput(op_desc, op_input._as_tf_output())

  # Add control inputs
  for control_input in control_inputs:
    pywrap_tf_session.TF_AddControlInput(op_desc, control_input._c_op)
  # pylint: enable=protected-access

  # Add attrs
  for name, attr_value in node_def.attr.items():
    serialized = attr_value.SerializeToString()
    # TODO(skyewm): this creates and deletes a new TF_Status for every attr.
    # It might be worth creating a convenient way to re-use the same status.
    pywrap_tf_session.TF_SetAttrValueProto(op_desc, compat.as_str(name),
                                           serialized)

  try:
    c_op = pywrap_tf_session.TF_FinishOperation(op_desc)
  except errors.InvalidArgumentError as e:
    # Convert to ValueError for backwards compatibility.
    raise ValueError(e.message)
```



#### _init_from_c_op

_init_from_c_op代码路径是tensorflow/python/framework/ops.py
```py
  def _init_from_c_op(self, c_op, g):
    """Initializes Operation from a TF_Operation."""

    if not isinstance(g, Graph):
      raise TypeError(f"Operation initialization requires a Graph, "
                      f"got {type(g)} for argument g.")

    if not isinstance(c_op, pywrap_tf_session.TF_Operation):
      raise TypeError(f"Operation initialization requires a TF_Operation, "
                      f"got {type(c_op)} for argument c_op.")

    self._original_op = None

    self._graph = g
    self._c_op = c_op

    # This will be set by self.inputs.
    self._inputs_val = None

    # List of _UserDevSpecs holding code location of device context manager
    # invocations and the users original argument to them.
    self._device_code_locations = None
    # Dict mapping op name to file and line information for op colocation
    # context managers.
    self._colocation_code_locations = None
    self._control_flow_context = g._get_control_flow_context()  # pylint: disable=protected-access

    # Gradient function for this op. There are three ways to specify gradient
    # function, and first available gradient gets used, in the following order.
    # 1. self._gradient_function
    # 2. Gradient name registered by "_gradient_op_type" attribute.
    # 3. Gradient name registered by op.type.
    self._gradient_function = None

    op_def = g._get_op_def(pywrap_tf_session.TF_OperationOpType(c_op))  # pylint: disable=protected-access

    self._is_stateful = op_def.is_stateful

    # Initialize self._outputs.
    num_outputs = pywrap_tf_session.TF_OperationNumOutputs(self._c_op)
    self._outputs = []
    for i in range(num_outputs):
      tf_output = c_api_util.tf_output(self._c_op, i)
      output_type = pywrap_tf_session.TF_OperationOutputType(tf_output)
      tensor = Tensor._create_with_tf_output(self, i, output_type, tf_output)  # pylint: disable=protected-access
      self._outputs.append(tensor)

    self._id_value = g._add_op(self, self.name)  # pylint: disable=protected-access
```

跳过前面冗长的赋值过程不看，核心的代码就是

```py
    op_def = g._get_op_def(pywrap_tf_session.TF_OperationOpType(c_op))  # pylint: disable=protected-access

    self._is_stateful = op_def.is_stateful

    # Initialize self._outputs.
    num_outputs = pywrap_tf_session.TF_OperationNumOutputs(self._c_op)
    self._outputs = []
    for i in range(num_outputs):
      tf_output = c_api_util.tf_output(self._c_op, i)
      output_type = pywrap_tf_session.TF_OperationOutputType(tf_output)
      tensor = Tensor._create_with_tf_output(self, i, output_type, tf_output)  # pylint: disable=protected-access
      self._outputs.append(tensor)

    self._id_value = g._add_op(self, self.name)  # pylint: disable=protected-access
```
### _create_op_helper


```py
 def _create_op_helper(self, op, compute_device=True):
    """Common logic for creating an op in this graph."""
    # Apply any additional attributes requested. Do not overwrite any existing
    # attributes.
    for key, value in self._attr_scope_map.items():
      try:
        op.get_attr(key)
      except ValueError:
        if callable(value):
          value = value(op.node_def)
          if not isinstance(value, (type(None), attr_value_pb2.AttrValue)):
            raise TypeError(
                "Callable for scope map key '%s' must return either None or "
                "an AttrValue protocol buffer; but it returned: %s" %
                (key, value))
        if value:
          op._set_attr(key, value)  # pylint: disable=protected-access

    # Apply a kernel label if one has been specified for this op type.
    try:
      kernel_label = self._op_to_kernel_label_map[op.type]
      op._set_attr("_kernel",  # pylint: disable=protected-access
                   attr_value_pb2.AttrValue(s=compat.as_bytes(kernel_label)))
    except KeyError:
      pass

    op._gradient_function = self._gradient_function_map.get(op.type)  # pylint: disable=protected-access

    # Apply the overriding op type for gradients if one has been specified for
    # this op type.
    try:
      mapped_op_type = self._gradient_override_map[op.type]
      op._set_attr("_gradient_op_type",  # pylint: disable=protected-access
                   attr_value_pb2.AttrValue(s=compat.as_bytes(mapped_op_type)))
    except KeyError:
      pass

    self._record_op_seen_by_control_dependencies(op)

    if compute_device:
      self._apply_device_functions(op)

    # Snapshot the colocation stack metadata before we might generate error
    # messages using it.  Note that this snapshot depends on the actual stack
    # and is independent of the op's _class attribute.
    # pylint: disable=protected-access
    op._colocation_code_locations = self._snapshot_colocation_stack_metadata()
    # pylint: enable=protected-access

    if self._colocation_stack:
      all_colocation_groups = []
      is_device_set = False
      for colocation_op in self._colocation_stack.peek_objs():
        try:
          all_colocation_groups.extend(colocation_op.colocation_groups())
        except AttributeError:
          pass
        if colocation_op.device and not is_device_set:
          # pylint: disable=protected-access
          op._set_device(colocation_op.device)
          # pylint: enable=protected-access
          is_device_set = True

      all_colocation_groups = sorted(set(all_colocation_groups))
      # pylint: disable=protected-access
      op._set_attr(
          "_class",
          attr_value_pb2.AttrValue(
              list=attr_value_pb2.AttrValue.ListValue(s=all_colocation_groups)))
      # pylint: enable=protected-access

    # Sets "container" attribute if
    # (1) self._container is not None
    # (2) "is_stateful" is set in OpDef
    # (3) "container" attribute is in OpDef
    # (4) "container" attribute is None
    if self._container and op._is_stateful:  # pylint: disable=protected-access
      try:
        container_attr = op.get_attr("container")
      except ValueError:
        # "container" attribute is not in OpDef
        pass
      else:
        if not container_attr:
          op._set_attr("container", attr_value_pb2.AttrValue(  # pylint: disable=protected-access
              s=compat.as_bytes(self._container)))
```
