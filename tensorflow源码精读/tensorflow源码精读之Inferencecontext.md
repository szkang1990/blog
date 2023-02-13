# ExtendedInferenceContext
ExtendedInferenceContext的代码在路径tensorflow/core/common_runtime/shape_refiner.h中，代码量很小

```cpp
class ExtendedInferenceContext {
 public:
  ExtendedInferenceContext(
      std::unique_ptr<shape_inference::InferenceContext> ic, const Node* node)
      : inference_context_(std::move(ic)), op_(node->name()) {
    input_types_.reserve(node->num_inputs());
    for (int i = 0; i < node->num_inputs(); i++) {
      input_types_.push_back(node->input_type(i));
    }
    output_types_.reserve(node->num_outputs());
    for (int i = 0; i < node->num_outputs(); i++) {
      output_types_.push_back(node->output_type(i));
    }
  }

  DataType input_type(int64_t idx) const { return input_types_[idx]; }
  DataType output_type(int64_t idx) const { return output_types_[idx]; }

  shape_inference::InferenceContext* get_context() {
    return inference_context_.get();
  }

  std::string op() const { return op_; }

 private:
  std::unique_ptr<shape_inference::InferenceContext> inference_context_;
  std::string op_;
  std::vector<DataType> input_types_;
  std::vector<DataType> output_types_;

  TF_DISALLOW_COPY_AND_ASSIGN(ExtendedInferenceContext);
};
```
核心的属性只有4个，其中inference_context_后面会做详细的介绍。其他的属性都一目了然。
```cpp
  std::unique_ptr<shape_inference::InferenceContext> inference_context_;
  std::string op_;
  std::vector<DataType> input_types_;
  std::vector<DataType> output_types_;
```

构造函数也很简单，接收两个入参：InferenceContext 和node， 其中InferenceContext 和  nodeName 直接赋值给inference_context_ 和op， 然后把Node的输入输出类型赋值给input_types_， output_types_。 
几个函数也是比较简单的，不赘述。总的来说ExtendedInferenceContext就是对InferenceContext进行了一次封装。

# InferenceContext

InferenceContext 是形状推断中的核心类。代码路径是tensorflow/core/framework/shape_inference.h



## 基本属性
基本属性如下：

```cpp

  static constexpr int64_t kUnknownDim = -1;
  static constexpr int32_t kUnknownRank = -1;

  ShapeManager shape_manager_;

  // inputs_, outputs_, and input_tensors_as_shapes_ refer to values from
  // `shape_manager_`.
  std::vector<ShapeHandle> inputs_;
  std::vector<const Tensor*> input_tensors_;
  std::vector<bool> requested_input_tensor_;
  std::vector<ShapeHandle> outputs_;
  // Can have fewer elements than inputs_.
  std::vector<ShapeHandle> input_tensors_as_shapes_;
  std::vector<bool> requested_input_tensor_as_partial_shape_;

  // input_handle_shapes_and_types_[i] is the list of shape/type pairs available
  // through the resource handle passed along input i of the node.
  //
  // Values may be NULL.
  std::vector<std::unique_ptr<std::vector<ShapeAndType>>>
      input_handle_shapes_and_types_;

  // output_handle_shapes_and_types_[i] is the list of shape/type pairs
  // available through the resource handle passed along output i of the node.
  //
  // Values may be NULL.
  std::vector<std::unique_ptr<std::vector<ShapeAndType>>>
      output_handle_shapes_and_types_;

  // Return types for the node this context is associated with. This information
  // is to eventually consolidate all the dtype and shape info, allowing for
  // output_handle_shapes_and_types_ to be removed.
  FullTypeDef ret_types_;

  const int graph_def_version_;
  AttrSlice attrs_;
  NameRangeMap input_name_map_;
  NameRangeMap output_name_map_;

  // An error set during construction. TODO(cwhipkey): remove when test
  // constructor is removed.
  Status construction_status_;

  // Pair of shape or dim handles that are equivalent, ie that represent the
  // same underlying shape of dimension. Note that for each pair at least one of
  // the handles must contain an unknown shape, since we don't keep track of
  // known shapes or dims here.
  std::vector<std::pair<ShapeHandle, ShapeHandle>> merged_shapes_;
  std::vector<std::pair<DimensionHandle, DimensionHandle>> merged_dims_;
  ```
属性中有两个constexpr 修饰的常量表示该常量在编译期间就会赋值，
ShapeManager 是对形状的封装，具体在tensorshape中有介绍
NameRangeMap 是一个map 定义在tensorflow/core/framework/node_def_util.h中， 表示代码是
```cpp
typedef gtl::FlatMap<StringPiece, std::pair<int, int>, hash<StringPiece>>
    NameRangeMap;
```
NameRangeMap的value是一个std::pair<int, int> ，用于表示输入和输出的位置以及名称。例如一个op的输入入参有2个，分别是input1，input2。而input1，是一个list长度是3， input2 是一个tensor，那么Namerange就是{input1:<0,2>,input2:2}


FullTypeDef ret_types_; 定义在Proto文件tensorflow/core/framework/full_type.proto，代码如下：
```cpp
message FullTypeDef {
  // The principal type represented by this object. This may be a concrete type
  // (Tensor, Dataset) a type variable (used for dependent types) a type
  // symbol (Any, Union). See FullTypeId for details.
  FullTypeId type_id = 1;

  repeated FullTypeDef args = 2;

  // Literal values of this type object, if the the type admits one.
  // For example, a type variable admits a string attribute - its name.
  // Shape-related types may admit int attributes - their static shape values.
  // Fields for more data types to be added as needed.
  oneof attr {
    string s = 3;
    int64 i = 4;
    // TODO(mdan): list/tensor, map? Need to reconcile with TFT_RECORD, etc.
  }
}
```
FullTypeId 就是对一些常见的数据类型的抽象，所以FullTypeDef 可以认为是对数据类型的封装。



## 构造函数
我们先来看他的构造函数
```cpp
InferenceContext::InferenceContext(
    int graph_def_version, const AttrSlice& attrs, const OpDef& op_def,
    const std::vector<PartialTensorShape>& input_shapes,
    const std::vector<const Tensor*>& input_tensors,
    const std::vector<PartialTensorShape>& input_tensors_as_shapes,
    const std::vector<
        std::unique_ptr<std::vector<std::pair<PartialTensorShape, DataType>>>>&
        input_handle_shapes_and_types)
    : graph_def_version_(graph_def_version), attrs_(attrs) {
  std::vector<ShapeHandle> input_tensors_as_shape_handles;
  input_tensors_as_shape_handles.reserve(input_tensors_as_shapes.size());
  for (const PartialTensorShape& p : input_tensors_as_shapes) {
    ShapeHandle shape;
    construction_status_.Update(MakeShapeFromPartialTensorShape(p, &shape));
    if (!construction_status_.ok()) {
      return;
    }
    input_tensors_as_shape_handles.push_back(shape);
  }
  PreInputInit(op_def, input_tensors, input_tensors_as_shape_handles);
  if (!construction_status_.ok()) return;
  inputs_.reserve(input_shapes.size());
  for (const PartialTensorShape& p : input_shapes) {
    ShapeHandle shape;
    construction_status_.Update(MakeShapeFromPartialTensorShape(p, &shape));
    if (!construction_status_.ok()) {
      return;
    }
    inputs_.push_back(shape);
  }
  std::vector<std::unique_ptr<std::vector<ShapeAndType>>> handle_data(
      input_shapes.size());
  for (int i = 0, end = input_handle_shapes_and_types.size(); i < end; ++i) {
    const auto& v = input_handle_shapes_and_types[i];
    if (v == nullptr) {
      continue;
    }
    handle_data[i].reset(new std::vector<ShapeAndType>(v->size()));
    auto& new_v = *handle_data[i];
    for (int j = 0, end = v->size(); j < end; ++j) {
      const auto& p = (*v)[j];
      construction_status_.Update(
          MakeShapeFromPartialTensorShape(p.first, &new_v[j].shape));
      if (!construction_status_.ok()) {
        return;
      }
      new_v[j].dtype = p.second;
    }
  }
  PostInputInit(std::move(handle_data));
}

InferenceContext::InferenceContext(
    int graph_def_version, const AttrSlice& attrs, const OpDef& op_def,
    const std::vector<ShapeHandle>& input_shapes,
    const std::vector<const Tensor*>& input_tensors,
    const std::vector<ShapeHandle>& input_tensors_as_shapes,
    std::vector<std::unique_ptr<std::vector<ShapeAndType>>>
        input_handle_shapes_and_types)
    : graph_def_version_(graph_def_version), attrs_(attrs) {
  PreInputInit(op_def, input_tensors, input_tensors_as_shapes);
  if (!construction_status_.ok()) return;
  inputs_ = input_shapes;

  PostInputInit(std::move(input_handle_shapes_and_types));
}
```
InferenceContext 有两个构造函数，两个函数入参的区别是第4个入参是PartialTensorShape 还是 ShapeHandle。
前者表示我们只知道输入的部分形状，后者表示我们知道输入的所有形状。如果入参是PartialTensorShape，则先生成ShapeHandle。
然后调用PreInputInit
在shape_refiner.cc中， InferenceContext通常只有前4个入参传入数据，其他都是传入一个空值。

```cpp
  std::unique_ptr<InferenceContext> ic(new InferenceContext(
      graph_def_version_, node->def(), node->op_def(),
      std::vector<ShapeHandle>(node->num_inputs()), {}, {}, {}));

```

构造函数虽然看着复杂，拆开看其实比较简单，inputs_ 被直接赋值。还有两个核心函数就是PreInputInit， PostInputInit。
PreInputInit是为了给属性赋值
input_tensors_， input_tensors_as_shapes_：直接赋值
input_name_map_， output_name_map_：调用NameRangesForNode从op_def中获取
outputs_， output_handle_shapes_and_types_：获取最大的输入和输出个数，开辟最大的输入和输出个数的内存空间。
PostInputInit主要给input_handle_shapes_and_types_，input_tensors_， requested_input_tensor_，预留空间这几个变量并没有被直接赋值而是resize了大小。以便形状的推断。

### PreInputInit函数

PreInputInit代码如下：
```cpp

void InferenceContext::PreInputInit(
    const OpDef& op_def, const std::vector<const Tensor*>& input_tensors,
    const std::vector<ShapeHandle>& input_tensors_as_shapes) {
  // TODO(mdan): This is also done at graph construction. Run only here instead?
  Status s = full_type::SpecializeType(attrs_, op_def, ret_types_);
  if (!s.ok()) {
    construction_status_ = s;
    return;
  }

  input_tensors_ = input_tensors;
  input_tensors_as_shapes_ = input_tensors_as_shapes;

  construction_status_ =
      NameRangesForNode(attrs_, op_def, &input_name_map_, &output_name_map_);
  if (!construction_status_.ok()) return;

  int num_outputs = 0;
  for (const auto& e : output_name_map_) {
    num_outputs = std::max(num_outputs, e.second.second);
  }
  outputs_.assign(num_outputs, nullptr);
  output_handle_shapes_and_types_.resize(num_outputs);
}
```
其中调用了NameRangesForNode函数来为input_name_map_， output_name_map_ 赋值，input_name_map_， output_name_map_ 都是NamerangeMap，前面已有介绍。
同时output_name_map_从获取了输出的个数num_outputs，给outputs_和output_handle_shapes_and_types_开辟了num_outputs的空间。
#### NameRangesForNode

NameRangesForNode的定义在tensorflow/core/framework/node_def_util.cc。代码如下,NameRangesForNode还调用了NameRangesHelper和 ComputeArgRange ，都是为了生成NameRangeMap：
```cpp
Status NameRangesForNode(const AttrSlice& attrs, const OpDef& op_def,
                         NameRangeMap* inputs, NameRangeMap* outputs) {
  if (inputs != nullptr) {
    TF_RETURN_IF_ERROR(
        NameRangesHelper(attrs, op_def.input_arg(), op_def, inputs));
  }
  if (outputs != nullptr) {
    return NameRangesHelper(attrs, op_def.output_arg(), op_def, outputs);
  }
  return Status::OK();
}

Status NameRangesHelper(const AttrSlice& attrs,
                        const protobuf::RepeatedPtrField<OpDef::ArgDef>& args,
                        const OpDef& op_def, NameRangeMap* result) {
  int start = 0;
  int num;
  for (const auto& arg : args) {
    TF_RETURN_IF_ERROR(ComputeArgRange(attrs, arg, op_def, &num));
    (*result)[arg.name()] = std::make_pair(start, start + num);
    start += num;
  }
  return Status::OK();
}

Status ComputeArgRange(const AttrSlice& attrs, const OpDef::ArgDef& arg_def,
                       const OpDef& op_def, int* num) {
  if (!arg_def.number_attr().empty()) {
    // Same type repeated "num" times.
    return GetNodeAttr(attrs, arg_def.number_attr(), num);
  } else if (!arg_def.type_list_attr().empty()) {
    const AttrValue* attr_value;
    TF_RETURN_IF_ERROR(attrs.Find(arg_def.type_list_attr(), &attr_value));
    *num = attr_value->list().type_size();
  } else if (!arg_def.type_attr().empty() || arg_def.type() != DT_INVALID) {
    *num = 1;
  } else {
    return errors::InvalidArgument(
        "Argument '", arg_def.name(),
        "' incorrectly specified in op definition: ", SummarizeOpDef(op_def));
  }
  return Status::OK();
}

```

### PostInputInit

```cpp
void InferenceContext::PostInputInit(
    std::vector<std::unique_ptr<std::vector<ShapeAndType>>> input_handle_data) {
  int num_inputs_from_node_def = 0;
  for (const auto& e : input_name_map_) {
    num_inputs_from_node_def =
        std::max(num_inputs_from_node_def, e.second.second);
  }

  // Allow passing empty shapes/dtypes to avoid changing every single test.
  if (input_handle_data.empty()) {
    input_handle_shapes_and_types_.resize(inputs_.size());
  } else {
    if (input_handle_data.size() != inputs_.size()) {
      construction_status_ = errors::InvalidArgument(
          "Wrong number of handle shapes passed; expected ", inputs_.size(),
          " got ", input_handle_data.size());
      return;
    }
    input_handle_shapes_and_types_ = std::move(input_handle_data);
  }
  const int inputs_size = inputs_.size();
  if (inputs_size != num_inputs_from_node_def) {
    construction_status_ = errors::InvalidArgument(
        "Wrong number of inputs passed: ", inputs_.size(), " while ",
        num_inputs_from_node_def, " expected based on NodeDef");
    return;
  }

  CHECK_LE(input_tensors_.size(), inputs_.size());
  input_tensors_.resize(inputs_.size());
  requested_input_tensor_.resize(inputs_.size());
  requested_input_tensor_as_partial_shape_.resize(inputs_.size());
}
```


### set_output函数
前面介绍了output_name_map_的定义，所以这个理set_output就很好解释了。我们希望设置名为output的输出的形状，所以第一个入参就是我们希望设置的output的名称，第二个是我们要设置的形状。当然，output可能布置一个tensor所以要设置的output的长度和我们给定的形状的长度必须要一致。


```cpp
Status InferenceContext::set_output(StringPiece output_name,
                                    const std::vector<ShapeHandle>& shapes) {
  auto result = output_name_map_.find(output_name);
  if (result == output_name_map_.end()) {
    return errors::InvalidArgument("Unknown output name: ", output_name);
  } else {
    const int start = result->second.first;
    const int size = result->second.second - start;
    const int shapes_size = shapes.size();
    if (size != shapes_size) {
      return errors::InvalidArgument("Must have exactly ", shapes.size(),
                                     " shapes.");
    }
    for (int i = 0; i < shapes_size; ++i) {
      outputs_[i + start] = shapes[i];
    }
  }
  return Status::OK();
}
```
在一些早期的版本tensorflow中，可以通过output的位置设置形状，代码如下：
```cpp
Status InferenceContext::set_output(int idx,
                                    const <ShapeHandle>& shape)
```
即给第idx个输出设置形状，非常简单。


### input

input函数的作用是名为input_name的所有输入tensor的形状，同样因为input_name可能有多个输入，所以输出是一个vector，赋值给入参output。

```cpp
Status InferenceContext::input(StringPiece input_name,
                               std::vector<ShapeHandle>* output) const {
  const auto result = input_name_map_.find(input_name);
  if (result == input_name_map_.end()) {
    return errors::InvalidArgument("Unknown input name: ", input_name);
  } else {
    output->clear();
    for (int i = result->second.first; i < result->second.second; ++i) {
      output->push_back(inputs_[i]);
    }
  }
  return Status::OK();
}
```

同样的，在早期版本的tensorflow中，有通过位置获取形状的函数input，代码如下，

```cpp
Status InferenceContext::input(int idx) 
```
表示获取第idx个输入。


### output 
output函数和input非常类似，不做赘述
```cpp
Status InferenceContext::output(StringPiece output_name,
                                std::vector<ShapeHandle>* output) const {
  const auto result = output_name_map_.find(output_name);
  if (result == output_name_map_.end()) {
    return errors::InvalidArgument("Unknown output name: ", output_name);
  } else {
    output->clear();
    for (int i = result->second.first; i < result->second.second; ++i) {
      output->push_back(outputs_[i]);
    }
  }
  return Status::OK();
}
```



###   Rank && RankKnown && Value && ValueKnown
Rank: 获取入参shapehandle的rank_属性 ，具体是指一个tensor的维度的长度，例如形状是[2,2,2]的tensor rank_ 是3
RankKnown:判断入参的rank_是否明确
Value ： 获取入参DimensionOrConstant的属性，具体是指一个维度，例如形状是[2,2,2]的tensor，该函数就是要获取2
ValueKnown ： 获取的tensor维度是否是未知的。
```cpp
static int32 Rank(ShapeHandle s) {
    return s.IsSet() ? s->rank_ : kUnknownRank;
  }

  static bool RankKnown(ShapeHandle s) {
    return (s.IsSet() && (Rank(s) != kUnknownRank));
  }
  static inline int64_t Value(DimensionOrConstant d) {
    return d.dim.IsSet() ? d.dim->value_ : d.val;
  }
  static inline bool ValueKnown(DimensionOrConstant d) {
    return Value(d) != kUnknownDim;
  }
```

### merge
对于给定的两个DimensionHandle的输入d0和d1，如果d0和d1的维度相同，那么直接返回d0，d0d1其中一个维度不确定，那么返回确定的那个，把d0d1写入merged_dims_， merged_dims_是一个存储DimensionHandle 对的vector。
```cpp
Status InferenceContext::Merge(DimensionHandle d0, DimensionHandle d1,
                               DimensionHandle* out) {
  if (d0.SameHandle(d1)) {
    *out = d0;
    return Status::OK();
  } else if (!ValueKnown(d1)) {
    *out = d0;
    merged_dims_.emplace_back(d0, d1);
    return Status::OK();
  } else if (!ValueKnown(d0)) {
    *out = d1;
    merged_dims_.emplace_back(d0, d1);
    return Status::OK();
  } else if (Value(d0) == Value(d1)) {
    *out = d0;
    return Status::OK();
  } else {
    *out = nullptr;
    return errors::InvalidArgument("Dimensions must be equal, but are ",
                                   Value(d0), " and ", Value(d1));
  }
}
```


### UnknownDim
这个函数接受一个

```cpp
  inline DimensionHandle UnknownDim() { return MakeDim(kUnknownDim); }

  inline DimensionHandle MakeDim(DimensionOrConstant d) {
    return shape_manager_.MakeDim(d);
  }


```

### withRank

对于给定的shapehandle 和rank
如果shpehandle的rank_和入参rank相等，那么直接返回入参ShapeHandle。如果ShapeHandle 没有设置rank_
```cpp
Status InferenceContext::WithRank(ShapeHandle shape, int64_t rank,
                                  ShapeHandle* out) {
  if (rank > kint32max) {
    return errors::InvalidArgument("Rank cannot exceed kint32max");
  }
  const int32_t existing = Rank(shape);
  if (existing == rank) {
    *out = shape;
    return Status::OK();
  }
  if (existing == kUnknownRank) {
    std::vector<DimensionHandle> dims;
    dims.reserve(rank);
    for (int i = 0; i < rank; ++i) {
      dims.push_back(UnknownDim());
    }
    ShapeHandle shp = shape_manager_.MakeShape(dims);
    return Merge(shape, shp, out);
  }
  *out = nullptr;

  return errors::InvalidArgument("Shape must be rank ", rank, " but is rank ",
                                 existing);
}
```