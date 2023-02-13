
这节介绍 c_api中的几个重要的函数
# TF_NewOperation

首先明确一下和Operation相关的几个概念，在tensorflow/c/c_api_internal.h中定义了几个结构体

```cpp
struct TF_OperationDescription {
  TF_OperationDescription(TF_Graph* g, const char* op_type,
                          const char* node_name)
      : node_builder(node_name, op_type, g->graph.op_registry()), graph(g) {}

  tensorflow::NodeBuilder node_builder;
  TF_Graph* graph;
  std::set<tensorflow::string> colocation_constraints;
};

struct TF_Operation {
  tensorflow::Node node;
};

```

可见Operation 只是封装了一层结构体的Node，所以Operation和Node 可以理解陈没有任何区别。下面来看TF_newOperation函数
```cpp
TF_OperationDescription* TF_NewOperationLocked(TF_Graph* graph,
                                               const char* op_type,
                                               const char* oper_name)
    TF_EXCLUSIVE_LOCKS_REQUIRED(graph->mu) {
  return new TF_OperationDescription(graph, op_type, oper_name);
}

TF_OperationDescription* TF_NewOperation(TF_Graph* graph, const char* op_type,
                                         const char* oper_name) {
  mutex_lock l(graph->mu);
  return TF_NewOperationLocked(graph, op_type, oper_name);
}
```
TF_NewOperation调用了TF_NewOperationLocked，TF_NewOperationLocked又调用了TF_OperationDescription的构造函数，TF_OperationDescription也是一个结构体，代码在上面已经给出

TF_OperationDescription有两个比较重要的属性tensorflow::NodeBuilder node_builder; 和 TF_Graph* graph; 这两个属性都在构造函数中被初始化。
## NodeBuilder
NodeBuilder是一个创建Node的工具类，
核心的属性是：

```cpp
  NodeDefBuilder def_builder_;
  const OpRegistryInterface* op_registry_;
  std::vector<NodeOut> inputs_;
  std::vector<Node*> control_inputs_;
  std::vector<string> errors_;
  string assigned_device_;
```

构造函数：

```cpp
NodeBuilder::NodeBuilder(StringPiece name, StringPiece op_name,
                         const OpRegistryInterface* op_registry,
                         const NodeDebugInfo* debug)
    : def_builder_(name, op_name, op_registry, debug) {}

NodeBuilder::NodeBuilder(StringPiece name, const OpDef* op_def)
    : def_builder_(name, op_def) {}

NodeBuilder::NodeBuilder(const NodeDefBuilder& def_builder)
    : def_builder_(def_builder) {}
```

除了构造函数以后还有一些比较重要的函数是Fanalize

```cpp
StatusOr<Node*> NodeBuilder::Finalize(Graph* graph, bool consume) {
  Node* out;
  TF_RETURN_IF_ERROR(Finalize(graph, &out, consume));
  return out;
}

Status NodeBuilder::Finalize(Graph* graph, Node** created_node, bool consume) {
  // In case of error, set *created_node to nullptr.
  if (created_node != nullptr) {
    *created_node = nullptr;
  }
  if (!errors_.empty()) {
    return errors::InvalidArgument(absl::StrJoin(errors_, "\n"));
  }

  NodeDef node_def;
  TF_RETURN_IF_ERROR(def_builder_.Finalize(&node_def, consume));
  TF_RETURN_IF_ERROR(ValidateNodeDef(node_def, def_builder_.op_def()));
  TF_RETURN_IF_ERROR(
      CheckOpDeprecation(def_builder_.op_def(), graph->versions().producer()));

  TF_ASSIGN_OR_RETURN(Node * node, graph->AddNode(std::move(node_def)));

  node->set_assigned_device_name(assigned_device_);

  for (size_t i = 0; i < inputs_.size(); ++i) {
    if (inputs_[i].node != nullptr) {  // Skip back edges.
      graph->AddEdge(inputs_[i].node, inputs_[i].index, node, i);
    }
  }
  for (Node* control_input : control_inputs_) {
    graph->AddControlEdge(control_input, node);
  }

  if (created_node != nullptr) *created_node = node;

  return Status::OK();
}
```
Finalize用于生成一个Node并且赋值给入参，这一点和讲解OP时的用法非常类似。有一点不同的是，NodeBuilder的Finalize不仅生成了一个Node而且这个node会被添加到图中，同时创建和其他node之间的edge。

Node的其它函数都比较平常，基本都是一些对属性的常规操作，不赘述




# TF_SetDevice

用于给node赋值，源码如下

```cpp
void TF_SetDevice(TF_OperationDescription* desc, const char* device) {
  desc->node_builder.Device(device);
}
```
入参是TF_OperationDescription，上面刚刚讲过TF_OperationDescription的属性之一是NodeBuilder。这个函数调用了node_builder的Device函数，Device函数如下：

```cpp
NodeBuilder& NodeBuilder::Device(StringPiece device_spec) {
  def_builder_.Device(device_spec);
  return *this;
}
```

# TF_AddInputList

TF_AddInputList 用于给Node添加输入,具体用法和TF_SetDevice类似,不再赘述

```cpp
void TF_AddInputList(TF_OperationDescription* desc, const TF_Output* inputs,
                     int num_inputs) {
  std::vector<NodeBuilder::NodeOut> input_list;
  input_list.reserve(num_inputs);
  for (int i = 0; i < num_inputs; ++i) {
    input_list.emplace_back(&inputs[i].oper->node, inputs[i].index);
  }
  desc->node_builder.Input(input_list);
}

NodeBuilder& NodeBuilder::Input(NodeOut src) {
  if (src.error) {
    AddIndexError(src.node, src.index);
  } else {
    inputs_.emplace_back(src.node, src.index);
    def_builder_.Input(src.name, src.index, src.dt);
  }
  return *this;
}
```

# TF_FinishOperation

TF_FinishOperation做了两件事
* 调用NodeBuilder中的Finalize函数生成一个Node，然后吧这个Node加入到图中，并且创建edge
* 调用Shape_refiner 的addNode创建形状推理相关信息


```cpp
TF_Operation* TF_FinishOperation(TF_OperationDescription* desc,
                                 TF_Status* status) {
  mutex_lock l(desc->graph->mu);
  return TF_FinishOperationLocked(desc, status);
}

TF_Operation* TF_FinishOperationLocked(TF_OperationDescription* desc,
                                       TF_Status* status)
    TF_EXCLUSIVE_LOCKS_REQUIRED(desc->graph->mu) {
  Node* ret = nullptr;

  if (desc->graph->name_map.count(desc->node_builder.node_name())) {
    status->status = InvalidArgument("Duplicate node name in graph: '",
                                     desc->node_builder.node_name(), "'");
  } else {
    if (!desc->colocation_constraints.empty()) {
      desc->node_builder.Attr(
          tensorflow::kColocationAttrName,
          std::vector<string>(desc->colocation_constraints.begin(),
                              desc->colocation_constraints.end()));
    }
    status->status = desc->node_builder.Finalize(&desc->graph->graph, &ret,
                                                 /*consume=*/true);

    if (status->status.ok()) {
      // Run shape inference function for newly added node.
      status->status = desc->graph->refiner.AddNode(ret);
    }
    if (status->status.ok()) {
      // Add the node to the name-to-node mapping.
      desc->graph->name_map[ret->name()] = ret;
    } else if (ret != nullptr) {
      desc->graph->graph.RemoveNode(ret);
      ret = nullptr;
    }
  }

  delete desc;

  return ToOperation(ret);
}

```
# TF_OperationOpType
返回node的nodeDef 的op name
```cpp
const char* TF_OperationOpType(TF_Operation* oper) {
  return oper->node.type_string().c_str();
}
```
type_string函数源码如下：

```cpp
const std::string& Node::type_string() const { return props_->node_def.op(); }
```

# TF_OperationNumOutputs
获取Operation的output的数量
```cpp
int TF_OperationNumOutputs(TF_Operation* oper) {
  return oper->node.num_outputs();
}
```

# TF_Output
TF_Output是一个结构体，结构体代码如下

```cpp
typedef struct TF_Output {
  TF_Operation* oper;
  int index;  // The index of the output within oper.
} TF_Output;
```

# TF_OperationOutputType

这个函数返回输出的类型

```cpp
TF_DataType TF_OperationOutputType(TF_Output oper_out) {
  return static_cast<TF_DataType>(
      oper_out.oper->node.output_type(oper_out.index));
}
```



