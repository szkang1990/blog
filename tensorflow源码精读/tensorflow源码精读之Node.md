​
tensorflow中node是一个很重要的概念，在源码中相关的类有三种，分别是 Node， NodeDef， NodeProperties。介绍一下这些概念的区别。

## Node
node的概念出现在文件tensorflow/core/graph/graph.h  和tensorflow/core/graph/graph.cc

Node 是一个类，其定义如下：

```cpp
class Node {
 public:
  std::string DebugString() const;
  int id() const { return id_; }
  int cost_id() const { return cost_id_; }
  const std::string& name() const;
  void set_name(std::string name);
  const std::string& type_string() const;

  // def() provides the NodeDef the user supplied, but the specifics
  // of this Node may have changed due to placement, optimization, etc.
  // In particular:
  // * def().name() will match name();
  // * def().op() will match type_string() and op_def().name();
  // * def().input() is not reliable, use "in_edges()" below instead;
  // * def().device() is the "user's requested device" and may not match
  //   the actual assigned device, see assigned_device_name() below;
  // * def().attr() is authoritative.
  // TODO(irving): Replace with NodeInfo.
  const NodeDef& def() const;
  const OpDef& op_def() const;

  // TODO(mdan): This is only used by control_flow_deps_o_chains. Remove?
  NodeDef* mutable_def();

  // input and output types
  int32 num_inputs() const;
  DataType input_type(int32_t i) const;
  const DataTypeVector& input_types() const;

  int32 num_outputs() const;
  DataType output_type(int32_t o) const;
  const DataTypeVector& output_types() const;

  // The device requested by the user.  For the actual assigned device,
  // use assigned_device_name() below.
  const std::string& requested_device() const;

  // This changes the user requested device but not necessarily the device that
  // on which the operation will run.
  void set_requested_device(const std::string& device);

  // This gives the device the runtime has assigned this node to.  If
  // you want the device the user requested, use def().device() instead.
  // TODO(josh11b): Validate that the assigned_device, if not empty:
  // fully specifies a device, and satisfies def().device().
  // TODO(josh11b): Move assigned_device_name outside of Node into a
  // NodeId->DeviceName map.
  const std::string& assigned_device_name() const;
  void set_assigned_device_name(const std::string& device_name);
  bool has_assigned_device_name() const {
    return assigned_device_name_index_ > 0;
  }
  int assigned_device_name_index() const { return assigned_device_name_index_; }
  void set_assigned_device_name_index(int index);

  // Sets 'original_node_names' field of this node's DebugInfo proto to
  // 'names'.
  void set_original_node_names(const std::vector<string>& names);
  void set_original_func_names(const std::vector<string>& names);

  // Read only access to attributes
  AttrSlice attrs() const;

  // Inputs requested by the NodeDef.  For the actual inputs, use in_edges.
  const protobuf::RepeatedPtrField<string>& requested_inputs() const;

  // Get the neighboring nodes via edges either in or out of this node.  This
  // includes control edges.
  gtl::iterator_range<NeighborIter> in_nodes() const;
  gtl::iterator_range<NeighborIter> out_nodes() const;
  const EdgeSet& in_edges() const { return in_edges_; }
  const EdgeSet& out_edges() const { return out_edges_; }

  // Node type helpers.
  bool IsSource() const { return id() == 0; }
  bool IsSink() const { return id() == 1; }
  // Anything other than the special Source & Sink nodes.
  bool IsOp() const { return id() > 1; }

  // Node class helpers
  bool IsSwitch() const { return class_ == NC_SWITCH; }
  bool IsMerge() const { return class_ == NC_MERGE; }
  bool IsEnter() const { return class_ == NC_ENTER; }
  bool IsExit() const { return class_ == NC_EXIT; }
  bool IsNextIteration() const { return class_ == NC_NEXT_ITERATION; }
  bool IsLoopCond() const { return class_ == NC_LOOP_COND; }
  bool IsControlTrigger() const { return class_ == NC_CONTROL_TRIGGER; }
  bool IsSend() const { return class_ == NC_SEND || class_ == NC_HOST_SEND; }
  bool IsRecv() const { return class_ == NC_RECV || class_ == NC_HOST_RECV; }
  bool IsConstant() const { return class_ == NC_CONSTANT; }
  bool IsVariable() const { return class_ == NC_VARIABLE; }
  bool IsIdentity() const { return class_ == NC_IDENTITY; }
  bool IsGetSessionHandle() const { return class_ == NC_GET_SESSION_HANDLE; }
  bool IsGetSessionTensor() const { return class_ == NC_GET_SESSION_TENSOR; }
  bool IsDeleteSessionTensor() const {
    return class_ == NC_DELETE_SESSION_TENSOR;
  }
  bool IsControlFlow() const {
    return (class_ != NC_OTHER) &&  // Fast path
           (IsSwitch() || IsMerge() || IsEnter() || IsExit() ||
            IsNextIteration());
  }
  bool IsHostSend() const { return class_ == NC_HOST_SEND; }
  bool IsHostRecv() const { return class_ == NC_HOST_RECV; }
  bool IsScopedAllocator() const { return class_ == NC_SCOPED_ALLOCATOR; }
  bool IsCollective() const { return class_ == NC_COLLECTIVE; }

  bool IsMetadata() const { return class_ == NC_METADATA; }
  bool IsFakeParam() const { return class_ == NC_FAKE_PARAM; }
  bool IsPartitionedCall() const { return class_ == NC_PARTITIONED_CALL; }

  // Returns true if this node is any kind of function call node.
  //
  // NOTE: "function call nodes" include partitioned call ops, symbolic gradient
  // ops, and ops whose type_string is the name of a function ("function ops").
  bool IsFunctionCall() const {
    return class_ == NC_PARTITIONED_CALL || class_ == NC_FUNCTION_OP ||
           class_ == NC_SYMBOLIC_GRADIENT;
  }

  bool IsIfNode() const { return class_ == NC_IF; }
  bool IsWhileNode() const { return class_ == NC_WHILE; }
  bool IsCaseNode() const { return class_ == NC_CASE; }
  // Is this node a function input
  bool IsArg() const { return class_ == NC_ARG; }
  // Is this node a function output
  bool IsRetval() const { return class_ == NC_RETVAL; }

  bool IsDistributedCommunication() const {
    return op_def().is_distributed_communication();
  }

  template <typename T>
  void AddAttr(const std::string& name, const T& val) {
    SetAttrValue(val, AddAttrHelper(name));
    UpdateProperties();
  }

  void AddAttr(const std::string& name, std::vector<string>&& val) {
    MoveAttrValue(std::move(val), AddAttrHelper(name));
    UpdateProperties();
  }

  void ClearAttr(const std::string& name);

  // Returns into '*e' the edge connecting to the 'idx' input of this Node.
  Status input_edge(int idx, const Edge** e) const;

  // Returns into '*edges' the input data edges of this Node, indexed by input
  // number. Does not return control edges.
  Status input_edges(std::vector<const Edge*>* edges) const;

  // Returns into '*n' the node that has an output connected to the
  // 'idx' input of this Node.
  Status input_node(int idx, const Node** n) const;
  Status input_node(int idx, Node** n) const;

  // Returns into '*t' the idx-th input tensor of this node, represented as the
  // output tensor of input_node(idx).
  Status input_tensor(int idx, OutputTensor* t) const;

  WhileContext* while_ctx() const { return while_ctx_; }
  void set_while_ctx(WhileContext* while_ctx) {
    DCHECK(IsExit());
    DCHECK(while_ctx_ == nullptr);
    while_ctx_ = while_ctx;
  }

  std::shared_ptr<NodeProperties> properties() const { return props_; }

  // Sets the stack trace for the node. Assumes that getting and setting the
  // stack trace for a given node will not race.
  void SetStackTrace(const std::shared_ptr<AbstractStackTrace>& stack_trace) {
    stack_trace_ = stack_trace;
  }

  // Get the stack trace for when the node was instantiated.
  const std::shared_ptr<AbstractStackTrace>& GetStackTrace() const {
    return stack_trace_;
  }

  // Called after an attr has changed. Decides whether we need to update some
  // property of the node (stored in props_).
  void UpdateProperties();

  // Erases type information from the node.
  void ClearTypeInfo();

  // Called after an incident non-control edge has changed. Does nothing if not
  // all input edges are defined.
  void RunForwardTypeInference();

 private:
  // TODO(mdan): Drop this.
  friend class Graph;
  Node();

  // Stack trace for the user code for node instantiation. Can be shared across
  // multiple nodes (e.g. when inlining).
  std::shared_ptr<AbstractStackTrace> stack_trace_;

  // Releases memory from props_, in addition to restoring *this to its
  // uninitialized state.
  void Clear();

  // Make a copy of the Node's props_ if props_ is shared with
  // other nodes. This must be called before mutating properties,
  // e.g. in AddAttr.
  void MaybeCopyOnWrite();

  AttrValue* AddAttrHelper(const std::string& name);

  // A set of mutually exclusive classes for different kinds of nodes,
  // class_ is initialized in the Node::Initialize routine based on the
  // node's type_string().
  enum NodeClass {
    NC_UNINITIALIZED,
    NC_SWITCH,
    NC_MERGE,
    NC_ENTER,
    NC_EXIT,
    NC_NEXT_ITERATION,
    NC_LOOP_COND,
    NC_CONTROL_TRIGGER,
    NC_SEND,
    NC_HOST_SEND,
    NC_RECV,
    NC_HOST_RECV,
    NC_CONSTANT,
    NC_VARIABLE,
    NC_IDENTITY,
    NC_GET_SESSION_HANDLE,
    NC_GET_SESSION_TENSOR,
    NC_DELETE_SESSION_TENSOR,
    NC_METADATA,
    NC_SCOPED_ALLOCATOR,
    NC_COLLECTIVE,
    NC_FAKE_PARAM,
    NC_PARTITIONED_CALL,
    NC_FUNCTION_OP,
    NC_SYMBOLIC_GRADIENT,
    NC_IF,
    NC_WHILE,
    NC_CASE,
    NC_ARG,
    NC_RETVAL,
    NC_OTHER  // Not a special kind of node
  };
```

我们先来看Node的核心属性：

### Node的属性
```cpp
int id_; // -1 until Initialize() is called

NodeClass class_;   枚举值
enum NodeClass {
    NC_UNINITIALIZED,
    NC_SWITCH,
    NC_MERGE,
    NC_ENTER,
    NC_EXIT,
    NC_NEXT_ITERATION,
    NC_LOOP_COND,
    NC_CONTROL_TRIGGER,
    NC_SEND,
    NC_HOST_SEND,
    NC_RECV,
    NC_HOST_RECV,
    NC_CONSTANT,
    NC_VARIABLE,
    NC_IDENTITY,
    NC_GET_SESSION_HANDLE,
    NC_GET_SESSION_TENSOR,
    NC_DELETE_SESSION_TENSOR,
    NC_METADATA,
    NC_SCOPED_ALLOCATOR,
    NC_COLLECTIVE,
    NC_FAKE_PARAM,
    NC_PARTITIONED_CALL,
    NC_FUNCTION_OP,
    NC_SYMBOLIC_GRADIENT,
    NC_IF,
    NC_WHILE,
    NC_CASE,
    NC_ARG,
    NC_RETVAL,
    NC_OTHER  // Not a special kind of node
  };

EdgeSet in_edges_;
EdgeSet out_edges_;
// 获取输入输出的edge

std::shared_ptr<NodeProperties> props_;

//  在创建图的时候，每次创造一个op都会指定一个device， 
//  然后再Graph的属性中，有一个Graph::device_names_属性，
//  该属性是一个vector记录所有被指定的device。
//  assigned_device_name_index_是这个Node分配的device在device_names_ 中的下标
 int assigned_device_name_index_;

Graph* graph_;
// 表示拥有这个node的图
```

### Node的成员函数
Node的函数基本都是一些围绕属性的函数，例如获取输入输出edges的函数


```cpp
  const EdgeSet& in_edges() const { return in_edges_; }
  const EdgeSet& out_edges() const { return out_edges_; }
```

获取某个编号的input_edge和output_edge( 例如获取第2个输入edge）

```cpp
Status input_edge(int idx, const Edge** e) const; 
Status output_edge(int idx, const Edge** e) const; 
```

获取NodeDef的函数
```cpp
const NodeDef& def() const;
```
获取某个编号的输入node

```cpp
  Status input_node(int idx, const Node** n) const;
  Status input_node(int idx, Node** n) const;
```
获取某个编号的tensor

```cpp
  Status input_tensor(int idx, OutputTensor* t) const;
```

获取输入和输出的类型

```cpp
  // input and output types
  int32 num_inputs() const;
  DataType input_type(int32_t i) const;
  const DataTypeVector& input_types() const;

  int32 num_outputs() const;
  DataType output_type(int32_t o) const;
  const DataTypeVector& output_types() const;
```
获取NodeProperties的函数

```cpp
std::shared_ptr<NodeProperties> properties() const { return props_; }
```
获取OpDef的函数
```cpp
const OpDef& op_def() const;
```
获取基本Node属性的函数
```cpp
  std::string DebugString() const;
  int id() const { return id_; }
  int cost_id() const { return cost_id_; }
  const std::string& name() const;
  const std::string& type_string() const;
```

### Node的构造函数
Node有一个初始化函数，作用和构造函数一样重要，代码如下

```cpp
Node::Node()
    : id_(-1),
      cost_id_(-1),
      class_(NC_UNINITIALIZED),
      props_(nullptr),
      assigned_device_name_index_(0),
      while_ctx_(nullptr) {}
```
### Initialize函数
```cpp
void Node::Initialize(int id, int cost_id,
                      std::shared_ptr<NodeProperties> props,
                      Node::NodeClass node_class) {
  DCHECK_EQ(id_, -1);
  DCHECK(in_edges_.empty());
  DCHECK(out_edges_.empty());
  id_ = id;
  cost_id_ = cost_id;

  props_ = std::move(props);
  class_ = node_class;
}
```
Initialize函数用于从NodeProperties生成一个Node
## NodeProperties

从上面我们已经得知，NodeProperties 是Node的一个属性，NodeProperties 的定义在tensorflow\core\framework\node_properties.h   和  tensorflow\core\framework\node_properties.cc

```cpp
struct NodeProperties {
 public:
  NodeProperties(const OpDef* op_def, NodeDef node_def,
                 const DataTypeSlice inputs, const DataTypeSlice outputs)
      : NodeProperties(op_def, std::move(node_def),
                       DataTypeVector(inputs.begin(), inputs.end()),
                       DataTypeVector(outputs.begin(), outputs.end())) {}

  NodeProperties(const OpDef* _op_def, NodeDef&& _node_def,
                 DataTypeVector inputs, DataTypeVector outputs)
      : NodeProperties(_op_def, std::move(_node_def), inputs, outputs,
                       nullptr) {}

  NodeProperties(const OpDef* _op_def, NodeDef&& _node_def,
                 DataTypeVector inputs, DataTypeVector outputs,
                 ForwardTypeInferenceFn fwd_type_fn)
      : op_def(_op_def),
        node_def(std::move(_node_def)),
        input_types(std::move(inputs)),
        input_types_slice(input_types),
        output_types(std::move(outputs)),
        output_types_slice(output_types),
        fwd_type_fn(fwd_type_fn) {}

  // Resets the 'props' shared pointer to point to a new NodeProperties created
  // from the given NodeDef. 'op_registry' is used to look up the OpDef
  // corresponding to node_def.op(). Returns an error if OpDef lookup or
  // creation failed.
  static Status CreateFromNodeDef(NodeDef node_def,
                                  const OpRegistryInterface* op_registry,
                                  std::shared_ptr<const NodeProperties>* props);

  const OpDef* op_def;  // not owned.
  NodeDef node_def;
  DataTypeVector input_types;
  DataTypeSlice input_types_slice;
  DataTypeVector output_types;
  DataTypeSlice output_types_slice;
  ForwardTypeInferenceFn fwd_type_fn;
};

```

### NodeProperties的属性
核心的属性是：
```cpp
  const OpDef* op_def;  // not owned.
  NodeDef node_def;
  DataTypeVector input_types;
  DataTypeSlice input_types_slice;
  DataTypeVector output_types;
  DataTypeSlice output_types_slice;
```

应该注意的是，OpDef是一个const

其中DataTypeVector和DataTypeSlice的定义如下：在比较老版本的tensorflow中都是自定义的格式，源码可以参考：https://github.com/szkang1990/tensorflow_makefile/tree/master/tensorflow/core/lib/gtl

在比较新的版本中都换成了absl库，这些不用深究。在后面可以看到其实input_types，input_types_slice  本质上是一个东西。

```cpp
typedef gtl::InlinedVector<DataType, 4> DataTypeVector;
typedef gtl::ArraySlice<DataType> DataTypeSlice;
```

### CreateFromNodeDef函数

```cpp
Status NodeProperties::CreateFromNodeDef(
    NodeDef node_def, const OpRegistryInterface* op_registry,
    std::shared_ptr<const NodeProperties>* props) {
  const OpDef* op_def;
  TF_RETURN_IF_ERROR(op_registry->LookUpOpDef(node_def.op(), &op_def));
  DataTypeVector input_types;
  DataTypeVector output_types;
  TF_RETURN_IF_ERROR(
      InOutTypesForNode(node_def, *op_def, &input_types, &output_types));
  props->reset(new NodeProperties(op_def, std::move(node_def),
                                  std::move(input_types),
                                  std::move(output_types)));
  return OkStatus();
}
```
这个函数用于从NodeDef创建一个NodeProperty

###  NodeProperties的构造函数
 NodeProperties的构造函数也非常简单

```cpp
  NodeProperties(const OpDef* _op_def, NodeDef&& _node_def,
                 DataTypeVector inputs, DataTypeVector outputs,
                 ForwardTypeInferenceFn fwd_type_fn)
      : op_def(_op_def),
        node_def(std::move(_node_def)),
        input_types(std::move(inputs)),
        input_types_slice(input_types),
        output_types(std::move(outputs)),
        output_types_slice(output_types),
        fwd_type_fn(fwd_type_fn) {}
```

## NodeDef
上面的代码揭示了NodeDef是NodeProperties 的一个属性，node的基本对象结构，定义在一个proto中，文件路径：tensorflow/core/framework/node_def.proto

```proto
syntax = "proto3";

package tensorflow;

import "tensorflow/core/framework/attr_value.proto";
import "tensorflow/core/framework/full_type.proto";

option cc_enable_arenas = true;
option java_outer_classname = "NodeProto";
option java_multiple_files = true;
option java_package = "org.tensorflow.framework";
option go_package = "github.com/tensorflow/tensorflow/tensorflow/go/core/framework/node_def_go_proto";

message NodeDef {
  // The name given to this operator. Used for naming inputs,
  // logging, visualization, etc.  Unique within a single GraphDef.
  // Must match the regexp "[A-Za-z0-9.][A-Za-z0-9_>./]*".
  string name = 1;

  // The operation name.  There may be custom parameters in attrs.
  // Op names starting with an underscore are reserved for internal use.
  string op = 2;

  // Each input is "node:src_output" with "node" being a string name and
  // "src_output" indicating which output tensor to use from "node". If
  // "src_output" is 0 the ":0" suffix can be omitted.  Regular inputs
  // may optionally be followed by control inputs that have the format
  // "^node".
  repeated string input = 3;

  // A (possibly partial) specification for the device on which this
  // node should be placed.
  // The expected syntax for this string is as follows:
  //
  // DEVICE_SPEC ::= PARTIAL_SPEC
  //
  // PARTIAL_SPEC ::= ("/" CONSTRAINT) *
  // CONSTRAINT ::= ("job:" JOB_NAME)
  //              | ("replica:" [1-9][0-9]*)
  //              | ("task:" [1-9][0-9]*)
  //              | ("device:" [A-Za-z]* ":" ([1-9][0-9]* | "*") )
  //
  // Valid values for this string include:
  // * "/job:worker/replica:0/task:1/device:GPU:3"  (full specification)
  // * "/job:worker/device:GPU:3"                   (partial specification)
  // * ""                                    (no specification)
  //
  // If the constraints do not resolve to a single device (or if this
  // field is empty or not present), the runtime will attempt to
  // choose a device automatically.
  string device = 4;

  // Operation-specific graph-construction-time configuration.
  // Note that this should include all attrs defined in the
  // corresponding OpDef, including those with a value matching
  // the default -- this allows the default to change and makes
  // NodeDefs easier to interpret on their own.  However, if
  // an attr with a default is not specified in this list, the
  // default will be used.
  // The "names" (keys) must match the regexp "[a-z][a-z0-9_]+" (and
  // one of the names from the corresponding OpDef's attr field).
  // The values must have a type matching the corresponding OpDef
  // attr's type field.
  // TODO(josh11b): Add some examples here showing best practices.
  map<string, AttrValue> attr = 5;

  message ExperimentalDebugInfo {
    // Opaque string inserted into error messages created by the runtime.
    //
    // This is intended to store the list of names of the nodes from the
    // original graph that this node was derived. For example if this node, say
    // C, was result of a fusion of 2 nodes A and B, then 'original_node' would
    // be {A, B}. This information can be used to map errors originating at the
    // current node to some top level source code.
    repeated string original_node_names = 1;

    // This is intended to store the list of names of the functions from the
    // original graph that this node was derived. For example if this node, say
    // C, was result of a fusion of node A in function FA and node B in function
    // FB, then `original_funcs` would be {FA, FB}. If the node is in the top
    // level graph, the `original_func` is empty. This information, with the
    // `original_node_names` can be used to map errors originating at the
    // current ndoe to some top level source code.
    repeated string original_func_names = 2;
  }

  // This stores debug information associated with the node.
  ExperimentalDebugInfo experimental_debug_info = 6;

  // The complete type of this node. Experimental and subject to change.
  // Currently, the field only contains the return types of the node. That will
  // extend in the future to contain the entire signature of the node, as a
  // function type.
  FullTypeDef experimental_type = 7;
}
```

node的主要属性有四个name，op，input，device。

name：略

op： 指node绑定的op，一个node，一般只能绑定一个op

input： 并非具体的input数值，是类似 的格式，node:src_output其中，node是输入节点的名称，src_output 是上个节点的第几个输出，例如"node:0"。这个属性是为了创建node之间的连接关系。

device：节点所在的设备。

attr:  node的一些其他属性


### nodeDefBuilder
因为nodeDef由proto定义，所以没有显式的构造函数，所以与op，kernel类似，nodeDef也有一个构建类nodeDefBuilder
```cpp
NodeDefBuilder::NodeDefBuilder(StringPiece name, StringPiece op_name,
                               const OpRegistryInterface* op_registry,
                               const NodeDebugInfo* debug) {
  node_def_.set_name(string(name));
  const Status status = op_registry->LookUpOpDef(string(op_name), &op_def_);
  if (status.ok()) {
    Initialize();
  } else {
    errors_.push_back(status.error_message());
    inputs_specified_ = 0;
  }
  if (debug != nullptr) MergeDebugInfo(*debug, &node_def_);
}

NodeDefBuilder::NodeDefBuilder(StringPiece name, StringPiece op_name,
                               const NodeDebugInfo& debug)
    : NodeDefBuilder(name, op_name) {
  MergeDebugInfo(debug, &node_def_);
}

NodeDefBuilder::NodeDefBuilder(StringPiece name, const OpDef* op_def)
    : op_def_(op_def) {
  node_def_.set_name(string(name));
  Initialize();
}
```
nodeDefBuilder的三个构造函数说明node的最核心属性是name和op，其他属性在构造环节没有定义。但是在NodeDefBuilder的类方法中，包含设置device等属性的方法，而且与opDefBuilder一样，这些方法都是返回类本身，以方便用链式方法

```cpp
NodeDef node_def;
Status status = NodeDefBuilder(node_name, op_name)
                    .Input(...)
                    .Attr(...)
                    .Finalize(&node_def);
```

结论
通过分析源码，可以看到Node NodeProperties NodeDef 是层层包含的关系




​