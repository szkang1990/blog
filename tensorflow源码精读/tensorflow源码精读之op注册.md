&emsp;在tensorflow最新的版本中，op注册的代码有了一些修改，比如删除了opregisterreceiver对象。修改以后代码复用性更好，但是也更难理解了。

## 注册一个算子

&emsp;当我们定义一个tensorflow的算子，首先我们需要tensorflow知道这个算子，也就是说我们要把这个算子注册到tensorflow的算子库中。一般算子注册的方法如下：
```cpp
REGISTER_OP("ZeroOutKsz")
    .Input("to_zero: int32")
    .Output("zeroed: int32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      return Status::OK();
    });
```
## REGISTER_OP
&emsp;我们来看第一行REGISTER_OP("ZeroOutKsz") ，REGISTER_OP是一个宏，定义在tensorflow/core/framework/op.h ，这是个非常重要的文件，op注册苏需要的所有接口，对象，函数全部写在op.h 和  op.cc中。这个宏写的很复杂，中间有环境检测，通过编译工具展开以后，可以看到是
```cpp
static ::tensorflow::InitOnStartupMarker const register_op0 __attribute__((unused)) = (::std::integral_constant<bool, !(false || true)>::value) ? ::tensorflow::InitOnStartupMarker{} : ::tensorflow::InitOnStartupMarker {} << ::tensorflow::register_op::OpDefBuilderWrapper("test")
```

&emsp;这个代码看着很复杂，我们来一点一点拆开

### 等号左边

```cpp
static ::tensorflow::InitOnStartupMarker const register_op0 __attribute__((unused))
```

&emsp;返回值是一个const类型的InitOnStartupMarker， __attribute__((unused)) 表名该常量可能没用上，所以忽略警告。

### 等号右边

```cpp
(::std::integral_constant<bool, !(false || true)>::value) ? ::tensorflow::InitOnStartupMarker{} : ::tensorflow::InitOnStartupMarker {} << ::tensorflow::register_op::OpDefBuilderWrapper("test")
```

>本质上这是一个三元表达式：cond？true_expr：false_expr \
>我们来看三元表达式的逻辑判断部分
>```cpp
> ::std::integral_constant<bool, !(false || true)>::value
>```
>std::integral_constant  用于包装特定的数据类型，这里不用深究，直接就当成 !(false || true) 处理就行，因此条件是false。我们直接看false_expr
>```cpp
> ::tensorflow::InitOnStartupMarker {} << ::tensorflow::register_op::OpDefBuilderWrapper("test")
>```
这里需要分别分析InitOnStartupMarker 和 OpDefBuilderWrapper 源码
## OpDefBuilderWrapper

&emsp;先来看OpDefBuilderWrapper。OpDefBuilderWrapper的代码如下

```cpp
namespace register_op {
 
class OpDefBuilderWrapper {
 public:
  explicit OpDefBuilderWrapper(const char name[]) : builder_(name) {}
  OpDefBuilderWrapper& Attr(std::string spec) {
    builder_.Attr(std::move(spec));
    return *this;
  }
  OpDefBuilderWrapper& Input(std::string spec) {
    builder_.Input(std::move(spec));
    return *this;
  }
  OpDefBuilderWrapper& Output(std::string spec) {
    builder_.Output(std::move(spec));
    return *this;
  }
  OpDefBuilderWrapper& SetIsCommutative() {
    builder_.SetIsCommutative();
    return *this;
  }
  OpDefBuilderWrapper& SetIsAggregate() {
    builder_.SetIsAggregate();
    return *this;
  }
  OpDefBuilderWrapper& SetIsStateful() {
    builder_.SetIsStateful();
    return *this;
  }
  OpDefBuilderWrapper& SetDoNotOptimize() {
    // We don't have a separate flag to disable optimizations such as constant
    // folding and CSE so we reuse the stateful flag.
    builder_.SetIsStateful();
    return *this;
  }
  OpDefBuilderWrapper& SetAllowsUninitializedInput() {
    builder_.SetAllowsUninitializedInput();
    return *this;
  }
  OpDefBuilderWrapper& Deprecated(int version, std::string explanation) {
    builder_.Deprecated(version, std::move(explanation));
    return *this;
  }
  OpDefBuilderWrapper& Doc(std::string text) {
    builder_.Doc(std::move(text));
    return *this;
  }
  OpDefBuilderWrapper& SetShapeFn(OpShapeInferenceFn fn) {
    builder_.SetShapeFn(std::move(fn));
    return *this;
  }
  OpDefBuilderWrapper& SetIsDistributedCommunication() {
    builder_.SetIsDistributedCommunication();
    return *this;
  }
 
  OpDefBuilderWrapper& SetTypeConstructor(OpTypeConstructor fn) {
    builder_.SetTypeConstructor(std::move(fn));
    return *this;
  }
 
  OpDefBuilderWrapper& SetForwardTypeFn(ForwardTypeInferenceFn fn) {
    builder_.SetForwardTypeFn(std::move(fn));
    return *this;
  }
 
  const ::tensorflow::OpDefBuilder& builder() const { return builder_; }
 
  InitOnStartupMarker operator()();
 
 private:
  mutable ::tensorflow::OpDefBuilder builder_;
};
 
}
```

在构造函数中，入参是字符串name，构造函数创建一个OpDefBuilder类builder_， 然后后面所有的类函数都是直接传给builder_， 然后类函数返回OpDefBuilderWrapper 本身，以方面用上面的链式定义。

主要用到的类函数包括

Attr：attr输入为一个字符串，attr的用法是
```cpp
REGISTER_OP("opName")
    .Attr("reduction: {'min', 'max', 'prod', 'sum'}")
    .Attr("T: {half, float, float64, int32, int64}")
    .Attr("num_devices: int")
    .Attr("shared_name: string")
    .Attr("XlaCompile: bool=true")
```
 "Attr"是一个允许自定义的值, 比如XLA引擎就根据自身需求提供了"XlaCompile", 如果一个Op将该值设置为true, 就会强制XLA引擎将其编译. 当然, 也可以设置一些无用的值, 就像函数声明里有一个并没有实际使用的参数, 除了浪费存储空间没有其他用途.

Input: 输入为一个字符串，input用于设置算子的输入

```cpp
REGISTER_OP("ZeroOutKsz")
    .Input("input1: int32")
    .Input("input2: int32")
```

Output: 输入为一个字符串，用于设置算子的输出

```cpp
REGISTER_OP("ZeroOutKsz")
    .Output("zeroed: int32")
```

 SetShapeFn:  输入是一个接口或者函数，用于设置输出的形状，例如下面的例子就是输入了一个lambda 函数为了在创建图的时候就能够实现tensor形状的自洽。改接口经常以shape_inference::InferenceContext 为输入， InferenceContext后面会单独讲

从上面的分析可以看到REGISTER_OP 的主要过程就是通过几层宏，生成了一个OpDefBuilderWrapper，而OpDefBuilderWrapper的构造函数和几个主要函数都是为了生成与修改OpDefBuilder。OpDefBuilder 的定义如下：

```cpp
class OpDefBuilder {
 public:
  // Constructs an OpDef with just the name field set.
  explicit OpDefBuilder(std::string op_name);
 
  // Adds an attr to this OpDefBuilder (and returns *this). The spec has
  // format "<name>:<type>" or "<name>:<type>=<default>"
  // where <name> matches regexp [a-zA-Z][a-zA-Z0-9_]*
  // (by convention only using capital letters for attrs that can be inferred)
  // <type> can be:
  //   "string", "int", "float", "bool", "type", "shape", or "tensor"
  //   "numbertype", "realnumbertype", "quantizedtype"
  //       (meaning "type" with a restriction on valid values)
  //   "{int32,int64}" or {realnumbertype,quantizedtype,string}"
  //       (meaning "type" with a restriction containing unions of value types)
  //   "{\"foo\", \"bar\n baz\"}", or "{'foo', 'bar\n baz'}"
  //       (meaning "string" with a restriction on valid values)
  //   "list(string)", ..., "list(tensor)", "list(numbertype)", ...
  //       (meaning lists of the above types)
  //   "int >= 2" (meaning "int" with a restriction on valid values)
  //   "list(string) >= 2", "list(int) >= 2"
  //       (meaning "list(string)" / "list(int)" with length at least 2)
  // <default>, if included, should use the Proto text format
  // of <type>.  For lists use [a, b, c] format.
  //
  // Note that any attr specifying the length of an input or output will
  // get a default minimum of 1 unless the >= # syntax is used.
  //
  // TODO(josh11b): Perhaps support restrictions and defaults as optional
  // extra arguments to Attr() instead of encoding them in the spec string.
  // TODO(josh11b): Would like to have better dtype handling for tensor attrs:
  // * Ability to say the type of an input/output matches the type of
  //   the tensor.
  // * Ability to restrict the type of the tensor like the existing
  //   restrictions for type attrs.
  // Perhaps by linking the type of the tensor to a type attr?
  OpDefBuilder& Attr(std::string spec);
 
  // Adds an input or output to this OpDefBuilder (and returns *this).
  // The spec has form "<name>:<type-expr>" or "<name>:Ref(<type-expr>)"
  // where <name> matches regexp [a-z][a-z0-9_]* and <type-expr> can be:
  // * For a single tensor: <type>
  // * For a sequence of tensors with the same type: <number>*<type>
  // * For a sequence of tensors with different types: <type-list>
  // Where:
  //   <type> is either one of "float", "int32", "string", ...
  //                 or the name of an attr (see above) with type "type".
  //   <number> is the name of an attr with type "int".
  //   <type-list> is the name of an attr with type "list(type)".
  // TODO(josh11b): Indicate Ref() via an optional argument instead of
  // in the spec?
  // TODO(josh11b): SparseInput() and SparseOutput() matching the Python
  // handling?
  OpDefBuilder& Input(std::string spec);
  OpDefBuilder& Output(std::string spec);
 
  // Turns on the indicated boolean flag in this OpDefBuilder (and
  // returns *this).
  OpDefBuilder& SetIsCommutative();
  OpDefBuilder& SetIsAggregate();
  OpDefBuilder& SetIsStateful();
  OpDefBuilder& SetAllowsUninitializedInput();
  OpDefBuilder& SetIsDistributedCommunication();
 
  // Deprecate the op at a certain GraphDef version.
  OpDefBuilder& Deprecated(int version, std::string explanation);
 
  // Adds docs to this OpDefBuilder (and returns *this).
  // Docs have the format:
  //   <1-line summary>
  //   <rest of the description>
  //   <name>: <description of name>
  //   <name>: <description of name>
  //     <if long, indent the description on subsequent lines>
  // Where <name> is the name of an attr, input, or output.  Please
  // wrap docs at 72 columns so that it may be indented in the
  // generated output.  For tensor inputs or outputs (not attrs), you
  // may start the description with an "=" (like name:= <description>)
  // to suppress the automatically-generated type documentation in
  // generated output.
  OpDefBuilder& Doc(std::string text);
 
  // Sets the function to be used as type constructor.
  // See OpRegistrationData::type_ctor.
  OpDefBuilder& SetTypeConstructor(OpTypeConstructor c);
 
  // Sets the function to be used for forward type inference.
  // See OpRegistrationData::fwd_type_fn.
  OpDefBuilder& SetForwardTypeFn(ForwardTypeInferenceFn f);
 
  // Sets the shape function to be used for shape inference.
  //
  // Note that currently (October 2016), python code still requires a
  // RegisterShape call to invoke this; see call_cpp_shape_fn in
  // python/framework/common_shapes.py
  OpDefBuilder& SetShapeFn(OpShapeInferenceFn fn);
 
  // Allows the `<type>` in calls to `Attr()` to be "any".
  // This is used by PythonAPIWrapper for pass-through parameters.
  OpDefBuilder& AllowAttrTypeAny();
 
  // Sets op_reg_data->op_def to the requested OpDef and
  // op_reg_data->shape_inference_fn to the requested shape inference function,
  // or returns an error.
  // Must be called after all of the above methods.
  //
  // Note that OpDefBuilder only reports parsing errors.  You should also
  // call ValidateOpDef() to detect other problems.
  Status Finalize(OpRegistrationData* op_reg_data) const;
 
 private:
  friend class FunctionDefHelper;
 
  // Adds control output to this OpDefBuilder (and returns *this).
  // The <name> must be a valid node name (matches regexp
  // [a-zA-Z][a-zA-Z0-9_]*). Named control output can only exist for functions.
  OpDefBuilder& ControlOutput(std::string name);
 
  OpDef* op_def() { return &op_reg_data_.op_def; }
 
  OpRegistrationData op_reg_data_;
  std::vector<string> attrs_;
  std::vector<string> inputs_;
  std::vector<string> outputs_;
  std::vector<string> control_outputs_;
  std::string doc_;
  std::vector<string> errors_;
  bool allow_attr_type_any_ = false;
};
 
```

OpDefBuilder几个主要的类成员都是我们最一开始的时候设置op的时候设置的那些属性

```cpp
  std::vector<string> attrs_;
  std::vector<string> inputs_;
  std::vector<string> outputs_;
  std::vector<string> control_outputs_;
```

下面这些类函数也都是用于设置这些属性。

```cpp
Attr(std::string spec);
OpDefBuilder& Input(std::string spec);
OpDefBuilder& Output(std::string spec);
```

 其中值得注意的类成员和类函数是

类函数

OpDef* op_def() { return &op_reg_data_.op_def; }

Status Finalize(OpRegistrationData* op_reg_data) const;

OpDefBuilder& SetShapeFn(OpShapeInferenceFn fn);

类成员

OpRegistrationData op_reg_data_;

 其中OpRegistrationData 是用于op注册的数据，是一个结构体定义如下：

struct OpRegistrationData {
 public:
  OpRegistrationData() {}
  OpRegistrationData(const OpDef& def) : op_def(def) {}
  OpRegistrationData(const OpDef& def, const OpShapeInferenceFn& fn,
                     bool is_function = false)
      : op_def(def), shape_inference_fn(fn), is_function_op(is_function) {}
 
  OpDef op_def;
  OpShapeInferenceFn shape_inference_fn;
 
  // Type constructor. This callable initializes the type of this op.
  // It is provided as a programmatic mechanism for defining an op's
  // type, as part of its registration. It is to be eventually replaced by a
  // textual language.
  //
  // Important: historically, op registrations only contained partial
  // input/output type information in non-standardized attribute declarations
  // (e.g. typically, input types were held in a `dtype` attribute). The type
  // constructor currently duplicates such attribute information, with the aim
  // of entirely subsuming it, and eventually deprecating all type-related
  // attributes.
  //
  // Since ops are typically parametrized, the type created by this constructor
  // is also parametric.
  //
  // Example: for an op `Foo(x: T) -> Bar[T]`:
  //
  //  * typically, its op registration included a single attribute `T: type`;
  //    then the respective input was defined as `x: T`; the output type `Bar`
  //    was implied by the op name.
  //  * the type constructor creates a FullType object containing `Bar[T]`; this
  //    still relies on the `T` attribute which it references.
  //  * in the future, the type constructor will create a FullType containing
  //    `Callable[(x: T), Bar[T]]`, and the attribute `T` will be deprecated.
  OpTypeConstructor type_ctor;
 
  // Forward type inference function. This callable infers the return type of an
  // op based on its input types.
  //
  // Note that the type constructor and forward inference functions need not be
  // mutually exclusive: if there is some static information that can be set
  // based on attributes, then that should be set in the constructor. If more
  // information can be extracted from inputs, that should be done in the
  // forward inference function.
  //
  // This is similar to the shape function, but is more general, and applied
  // directly to NodeDefs, rather than working on the ShapeAndType structures.
  // Note that the op input/output declarations may specify some implicit type
  // constraints through attribute references (i.e. two inputs pointing to the
  // same type attribute). Those constraints may duplicate what this function
  // specifies in its body. That's intended, for a gradual transition to a more
  // formal type system.
  //
  // These type inference functions are intermediate solutions as well: once the
  // op registration has a complete, formal type definition, along with
  // a solver-based type inference, it will replace these functions.
  //
  // TODO(mdan): Merge with shape inference.
  // TODO(mdan): Replace with a union-based type inference algorithm.
  ForwardTypeInferenceFn fwd_type_fn;
 
  bool is_function_op = false;
};

我们忽略其冗长的注释，他的核心就两个变量

  OpDef op_def;
  OpShapeInferenceFn shape_inference_fn;
其中opdef是由tensorflow/core/framework/op_def.proto 定义的对象

proto内容如下：

syntax = "proto3";
 
package tensorflow;
 
import "tensorflow/core/framework/attr_value.proto";
import "tensorflow/core/framework/full_type.proto";
import "tensorflow/core/framework/resource_handle.proto";
import "tensorflow/core/framework/types.proto";
 
option cc_enable_arenas = true;
option java_outer_classname = "OpDefProtos";
option java_multiple_files = true;
option java_package = "org.tensorflow.framework";
option go_package = "github.com/tensorflow/tensorflow/tensorflow/go/core/framework/op_def_go_proto";
 
// Defines an operation. A NodeDef in a GraphDef specifies an Op by
// using the "op" field which should match the name of a OpDef.
// LINT.IfChange
message OpDef {
  // Op names starting with an underscore are reserved for internal use.
  // Names should be CamelCase and match the regexp "[A-Z][a-zA-Z0-9>_]*".
  string name = 1;
 
  // For describing inputs and outputs.
  message ArgDef {
    // Name for the input/output.  Should match the regexp "[a-z][a-z0-9_]*".
    string name = 1;
 
    // Human readable description.
    string description = 2;
 
    // Describes the type of one or more tensors that are accepted/produced
    // by this input/output arg.  The only legal combinations are:
    // * For a single tensor: either the "type" field is set or the
    //   "type_attr" field is set to the name of an attr with type "type".
    // * For a sequence of tensors with the same type: the "number_attr"
    //   field will be set to the name of an attr with type "int", and
    //   either the "type" or "type_attr" field will be set as for
    //   single tensors.
    // * For a sequence of tensors, the "type_list_attr" field will be set
    //   to the name of an attr with type "list(type)".
    DataType type = 3;
    string type_attr = 4;    // if specified, attr must have type "type"
    string number_attr = 5;  // if specified, attr must have type "int"
    // If specified, attr must have type "list(type)", and none of
    // type, type_attr, and number_attr may be specified.
    string type_list_attr = 6;
 
    // The handle data for resource inputs.
    repeated ResourceHandleProto.DtypeAndShape handle_data = 7;
 
    // For inputs: if true, the inputs are required to be refs.
    //   By default, inputs can be either refs or non-refs.
    // For outputs: if true, outputs are refs, otherwise they are not.
    bool is_ref = 16;
 
    // Experimental. Full type declaration for this argument.
    // The full type specification combines type, type_attr, type_list_attr,
    // etc. into a unified representation.
    // This declaration may contain non-concrete types (for example,
    // Tensor<TypeVar<'T'>> is a valid type declaration.
    //
    // Note: this is a transient field. The long-term aim is to represent the
    // entire OpDef as a single type: a callable. In that context, this field is
    // just the type of a single argument.
    FullTypeDef experimental_full_type = 17;
  }
 
  // Description of the input(s).
  repeated ArgDef input_arg = 2;
 
  // Description of the output(s).
  repeated ArgDef output_arg = 3;
 
  // Named control outputs for this operation. Useful only for composite
  // operations (i.e. functions) which want to name different control outputs.
  repeated string control_output = 20;
 
  // Description of the graph-construction-time configuration of this
  // Op.  That is to say, this describes the attr fields that will
  // be specified in the NodeDef.
  message AttrDef {
    // A descriptive name for the argument.  May be used, e.g. by the
    // Python client, as a keyword argument name, and so should match
    // the regexp "[a-z][a-z0-9_]+".
    string name = 1;
 
    // One of the type names from attr_value.proto ("string", "list(string)",
    // "int", etc.).
    string type = 2;
 
    // A reasonable default for this attribute if the user does not supply
    // a value.  If not specified, the user must supply a value.
    AttrValue default_value = 3;
 
    // Human-readable description.
    string description = 4;
 
    // TODO(josh11b): bool is_optional?
 
    // --- Constraints ---
    // These constraints are only in effect if specified.  Default is no
    // constraints.
 
    // For type == "int", this is a minimum value.  For "list(___)"
    // types, this is the minimum length.
    bool has_minimum = 5;
    int64 minimum = 6;
 
    // The set of allowed values.  Has type that is the "list" version
    // of the "type" field above (uses the "list" field of AttrValue).
    // If type == "type" or "list(type)" above, then the "type" field
    // of "allowed_values.list" has the set of allowed DataTypes.
    // If type == "string" or "list(string)", then the "s" field of
    // "allowed_values.list" has the set of allowed strings.
    AttrValue allowed_values = 7;
  }
  repeated AttrDef attr = 4;
 
  // Optional deprecation based on GraphDef versions.
  OpDeprecation deprecation = 8;
 
  // One-line human-readable description of what the Op does.
  string summary = 5;
 
  // Additional, longer human-readable description of what the Op does.
  string description = 6;
 
  // -------------------------------------------------------------------------
  // Which optimizations this operation can participate in.
 
  // True if the operation is commutative ("op(a,b) == op(b,a)" for all inputs)
  bool is_commutative = 18;
 
  // If is_aggregate is true, then this operation accepts N >= 2
  // inputs and produces 1 output all of the same type.  Should be
  // associative and commutative, and produce output with the same
  // shape as the input.  The optimizer may replace an aggregate op
  // taking input from multiple devices with a tree of aggregate ops
  // that aggregate locally within each device (and possibly within
  // groups of nearby devices) before communicating.
  // TODO(josh11b): Implement that optimization.
  bool is_aggregate = 16;  // for things like add
 
  // Other optimizations go here, like
  //   can_alias_input, rewrite_when_output_unused, partitioning_strategy, etc.
 
  // -------------------------------------------------------------------------
  // Optimization constraints.
 
  // Ops are marked as stateful if their behavior depends on some state beyond
  // their input tensors (e.g. variable reading op) or if they have
  // a side-effect (e.g. printing or asserting ops). Equivalently, stateless ops
  // must always produce the same output for the same input and have
  // no side-effects.
  //
  // By default Ops may be moved between devices.  Stateful ops should
  // either not be moved, or should only be moved if that state can also
  // be moved (e.g. via some sort of save / restore).
  // Stateful ops are guaranteed to never be optimized away by Common
  // Subexpression Elimination (CSE).
  bool is_stateful = 17;  // for things like variables, queue
 
  // -------------------------------------------------------------------------
  // Non-standard options.
 
  // By default, all inputs to an Op must be initialized Tensors.  Ops
  // that may initialize tensors for the first time should set this
  // field to true, to allow the Op to take an uninitialized Tensor as
  // input.
  bool allows_uninitialized_input = 19;  // for Assign, etc.
 
  // Indicates whether the op implementation uses distributed communication.
  // If True, the op is allowed to return errors for network disconnection and
  // trigger TF network failure handling logics.
  bool is_distributed_communication = 21;
}
// LINT.ThenChange(
//     https://www.tensorflow.org/code/tensorflow/core/framework/op_def_util.cc)
 
// Information about version-dependent deprecation of an op
message OpDeprecation {
  // First GraphDef version at which the op is disallowed.
  int32 version = 1;
 
  // Explanation of why it was deprecated and what to use instead.
  string explanation = 2;
}
 
// A collection of OpDefs
message OpList {
  repeated OpDef op = 1;
}

内容非常简单，就是我们设置的input，output，attr

 OpShapeInferenceFn是一个接口

typedef std::function<Status(shape_inference::InferenceContext* c)>
    OpShapeInferenceFn;
看到这里我们就明白要弄出一个OpRegistrationData 来注册op了，因为我们在一开始定义一个op的时候不仅提供了input，output，attr，还提供了形状推断的方法，而opDef由 proto定义，无法定义接口信息，所以我们专门做了一个接口OpShapeInferenceFn来存储形状推断方法。然后把opDef和OpShapeInferenceFn 一起当做op注册数据。

明白这一点以后，我们再来看这几个类函数

OpDef* op_def() { return &op_reg_data_.op_def; }

Status Finalize(OpRegistrationData* op_reg_data) const;

OpDefBuilder& SetShapeFn(OpShapeInferenceFn fn);

 OpDefBuilder& SetShapeFn(OpShapeInferenceFn fn); 用于设置形状推断方法，所以设置op_reg_data_

OpDefBuilder& OpDefBuilder::SetShapeFn(OpShapeInferenceFn fn) {
  if (op_reg_data_.shape_inference_fn != nullptr) {
    errors_.push_back(
        strings::StrCat("SetShapeFn called twice for Op ", op_def()->name()));
  } else {
    op_reg_data_.shape_inference_fn = OpShapeInferenceFn(fn);
  }
  return *this;
}
 OpDef* op_def() { return &op_reg_data_.op_def; }  非常好理解，只有op_reg_data_ 存储了op_def数据

 Status Finalize(OpRegistrationData* op_reg_data) const;

OpDefBuilder只是用于创建op的类，真正用于注册是OpRegistrationData。我们来回想，我们通过多个宏传入参数，定义了一个OpDefBuilderWrapper ，OpDefBuilderWrapper 里面定义了一个 OpDefBuilder。但是前面整个过程我们并没有定义一个OpRegistrationData。所以Finalize的作用是定义一个OpRegistrationData。从名称也能看出来只有当我们注册信息全部就绪以后，我们才会最终注册一个op，所以名字叫Finalize。

其源码如下，具体做法是，首先把op_reg_data_ 赋给了入参，一个指针op_reg_data。然后把OpDefBuilder 中的所有类成员，如attr，input， output都赋给op_reg_data 中的op_def。

Status OpDefBuilder::Finalize(OpRegistrationData* op_reg_data) const {
  std::vector<string> errors = errors_;
  *op_reg_data = op_reg_data_;
 
  OpDef* op_def = &op_reg_data->op_def;
  for (StringPiece attr : attrs_) {
    FinalizeAttr(attr, allow_attr_type_any_, op_def, &errors);
  }
  for (StringPiece input : inputs_) {
    FinalizeInputOrOutput(input, false, op_def, &errors);
  }
  for (StringPiece output : outputs_) {
    FinalizeInputOrOutput(output, true, op_def, &errors);
  }
  for (StringPiece control_output : control_outputs_) {
    FinalizeControlOutput(control_output, op_def, &errors);
  }
  FinalizeDoc(doc_, op_def, &errors);
 
  if (op_reg_data->type_ctor != nullptr) {
    TF_RETURN_IF_ERROR(op_reg_data->type_ctor(op_def));
  }
 
  if (errors.empty()) return Status::OK();
  return errors::InvalidArgument(absl::StrJoin(errors, "\n"));
}

这里一定要记住Finalize 的入参是空的！是把 OpDefBuilder 的数据全部赋给了入参！！这在后面非常重要。

至此，我们已经完成了op注册的所有数据准备，生成了一个OpRegistrationData。

但是，还没有注册行为。注册行为在OpDefBuilderWrapper对象中的operator()() 中实现。没错，OpDefBuilderWrapper重载了符号"()" , 是一个函数对象，operator()()具体的代码在tensorflow/core/framework/op.cc

InitOnStartupMarker OpDefBuilderWrapper::operator()() {
  OpRegistry::Global()->Register(
      [builder =
           std::move(builder_)](OpRegistrationData* op_reg_data) -> Status {
        return builder.Finalize(op_reg_data);
      });
  return {};
}
 这个代码介绍了 “->” 符号的两种用法 1. 对象指针获取方法，2： lambda函数的固定格式。

我们来看这个函数主体其实就是调用了  OpRegistry::Global() 的返回值的Register函数，只不过Register函数的输入是一个接口或者lambda函数。看名字就知道，我们已经到了op注册的环节了，所以我们不妨把这几个对象和函数都研究一下：

OpRegistry  对象：op注册的核心对象

代码如下：

class OpRegistry : public OpRegistryInterface {
 public:
  typedef std::function<Status(OpRegistrationData*)> OpRegistrationDataFactory;
 
  OpRegistry();
  ~OpRegistry() override;
 
  void Register(const OpRegistrationDataFactory& op_data_factory);
 
  Status LookUp(const std::string& op_type_name,
                const OpRegistrationData** op_reg_data) const override;
 
  // Returns OpRegistrationData* of registered op type, else returns nullptr.
  const OpRegistrationData* LookUp(const std::string& op_type_name) const;
 
  // Fills *ops with all registered OpDefs (except those with names
  // starting with '_' if include_internal == false) sorted in
  // ascending alphabetical order.
  void Export(bool include_internal, OpList* ops) const;
 
  // Returns ASCII-format OpList for all registered OpDefs (except
  // those with names starting with '_' if include_internal == false).
  std::string DebugString(bool include_internal) const;
 
  // A singleton available at startup.
  static OpRegistry* Global();
 
  // Get all registered ops.
  void GetRegisteredOps(std::vector<OpDef>* op_defs);
 
  // Get all `OpRegistrationData`s.
  void GetOpRegistrationData(std::vector<OpRegistrationData>* op_data);
 
  // Registers a function that validates op registry.
  void RegisterValidator(
      std::function<Status(const OpRegistryInterface&)> validator) {
    op_registry_validator_ = std::move(validator);
  }
 
  // Watcher, a function object.
  // The watcher, if set by SetWatcher(), is called every time an op is
  // registered via the Register function. The watcher is passed the Status
  // obtained from building and adding the OpDef to the registry, and the OpDef
  // itself if it was successfully built. A watcher returns a Status which is in
  // turn returned as the final registration status.
  typedef std::function<Status(const Status&, const OpDef&)> Watcher;
 
  // An OpRegistry object has only one watcher. This interface is not thread
  // safe, as different clients are free to set the watcher any time.
  // Clients are expected to atomically perform the following sequence of
  // operations :
  // SetWatcher(a_watcher);
  // Register some ops;
  // op_registry->ProcessRegistrations();
  // SetWatcher(nullptr);
  // Returns a non-OK status if a non-null watcher is over-written by another
  // non-null watcher.
  Status SetWatcher(const Watcher& watcher);
 
  // Process the current list of deferred registrations. Note that calls to
  // Export, LookUp and DebugString would also implicitly process the deferred
  // registrations. Returns the status of the first failed op registration or
  // Status::OK() otherwise.
  Status ProcessRegistrations() const;
 
  // Defer the registrations until a later call to a function that processes
  // deferred registrations are made. Normally, registrations that happen after
  // calls to Export, LookUp, ProcessRegistrations and DebugString are processed
  // immediately. Call this to defer future registrations.
  void DeferRegistrations();
 
  // Clear the registrations that have been deferred.
  void ClearDeferredRegistrations();
 
 private:
  // Ensures that all the functions in deferred_ get called, their OpDef's
  // registered, and returns with deferred_ empty.  Returns true the first
  // time it is called. Prints a fatal log if any op registration fails.
  bool MustCallDeferred() const TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);
 
  // Calls the functions in deferred_ and registers their OpDef's
  // It returns the Status of the first failed op registration or Status::OK()
  // otherwise.
  Status CallDeferred() const TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);
 
  // Add 'def' to the registry with additional data 'data'. On failure, or if
  // there is already an OpDef with that name registered, returns a non-okay
  // status.
  Status RegisterAlreadyLocked(const OpRegistrationDataFactory& op_data_factory)
      const TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);
 
  const OpRegistrationData* LookUpSlow(const std::string& op_type_name) const;
 
  mutable mutex mu_;
  // Functions in deferred_ may only be called with mu_ held.
  mutable std::vector<OpRegistrationDataFactory> deferred_ TF_GUARDED_BY(mu_);
  // Values are owned.
  mutable std::unordered_map<string, const OpRegistrationData*> registry_
      TF_GUARDED_BY(mu_);
  mutable bool initialized_ TF_GUARDED_BY(mu_);
 
  // Registry watcher.
  mutable Watcher watcher_ TF_GUARDED_BY(mu_);
 
  std::function<Status(const OpRegistryInterface&)> op_registry_validator_;
};

核心的成员和函数是

static OpRegistry* Global();

OpRegistry* OpRegistry::Global() {
  static OpRegistry* global_op_registry = new OpRegistry;
  return global_op_registry;
}
这里需要注意的是，这里创建是OpRegistry 是静态的，创建了一个全局唯一的OpRegistry用于op注册。具体的定义方法是

OpRegistry::OpRegistry()
    : initialized_(false), op_registry_validator_(DefaultValidator) {}
初始化了initialized_ ，这个在后面注册op时会用到。

 Register

void OpRegistry::Register(const OpRegistrationDataFactory& op_data_factory) {
  mutex_lock lock(mu_);
  if (initialized_) {
    TF_QCHECK_OK(RegisterAlreadyLocked(op_data_factory));
  } else {
    deferred_.push_back(op_data_factory);
  }
}
deferred_ 是一个线程安全的vector,  用于存储op注册的函数

mutable std::vector<OpRegistrationDataFactory> deferred_ TF_GUARDED_BY(mu_);
虽然有一个条件判断，但是其本质是一样的，deferred_存储op注册的函数以后，在

Export,GetOpRegistrationData,GetOpRegistrationData,LookUpSlow,ProcessRegistrations等函数中，都会遍历deferred_ ，对deferred_每个元素调用RegisterAlreadyLocked。

Status OpRegistry::RegisterAlreadyLocked(
    const OpRegistrationDataFactory& op_data_factory) const {
  std::unique_ptr<OpRegistrationData> op_reg_data(new OpRegistrationData);
  Status s = op_data_factory(op_reg_data.get());
  if (s.ok()) {
    s = ValidateOpDef(op_reg_data->op_def);
    if (s.ok() &&
        !gtl::InsertIfNotPresent(&registry_, op_reg_data->op_def.name(),
                                 op_reg_data.get())) {
      s = errors::AlreadyExists("Op with name ", op_reg_data->op_def.name());
    }
  }
  Status watcher_status = s;
  if (watcher_) {
    watcher_status = watcher_(s, op_reg_data->op_def);
  }
  if (s.ok()) {
    op_reg_data.release();
  } else {
    op_reg_data.reset();
  }
  return watcher_status;
}

RegisterAlreadyLocked 首先创建了一个空的OpRegistrationData op_reg_data， 这里用到了智能指针unique_ptr。 然后把这个空OpRegistrationData 赋给了operator中的lambda函数：

 [builder = std::move(builder_)](OpRegistrationData* op_reg_data) -> 
  Status { return builder.Finalize(op_reg_data); }
前面已经说了，这个函数是把opDefBuilder的数据全部赋给了入参。因此前面创建的空op_reg_data就有了数据。然后调下面的函数把op_reg_data 写入registry_

!gtl::InsertIfNotPresent(&registry_, op_reg_data->op_def.name(),
   op_reg_data.get())) 
其中registry_ 是一个线程安全的map

mutable std::unordered_map<string, const OpRegistrationData*> registry_
      TF_GUARDED_BY(mu_);
所以InsertIfNotPresent本质上就是把(op_def.name, op_reg_data) 插入到这个map中。

除此之外还有lookup系列的函数，都是从registry_ 中查找 op的过程，都比较简单，这里就不赘述了。

OpDefBuilderWrapper的部分讲完了，总结一下：

我们提供op的定义，例如input，output等信息给REGISTER_OP 这个宏，REGISTER_OP宏展开以后是一个三元表达式，会创建一个OpDefBuilderWrapper ，而OpDefBuilderWrapper 有类成员OpDefBuilder ，OpDefBuilderWrapper 接收所有的op属性都会传给OpDefBuilder。 同时OpDefBuilderWrapper 是一个函数对象，调用OpDefBuilderWrapper以后会触发 一个静态对象OpRegistry的注册函数Register，注册函数Register 的入参是一个lambda函数，这个lambda函数需要一个入参OpRegistrationData格式的op_reg_data，op_reg_data 中包含了opDef和OpShapeInferenceFn， 即节点定义和形状推断方法。同时会捕捉OpDefBuilderWrapper 成员OpDefBuilder，在函数体主体中把OpDefBuilder所有的数据都传给入参op_reg_data。

注册函数Register会创建一个空OpRegistrationData   op_reg_data ，然后把它传给lambda函数，使得这个op_reg_data 被写入数据，然后把op_reg_data 写入 registry_，完成注册。

上面所有的过程都依赖一个过程：调用OpDefBuilderWrapper这个函数对象，这个调用过程需要分析InitOnStartupMarker 来完成。

我们先来看看前面InitOnStartupMarker的用法：

:tensorflow::InitOnStartupMarker {} << ::tensorflow::register_op::OpDefBuilderWrapper("test")

看着有点怪异，尤其是<< , 这个符号很有可能被重载了。

我们来看一下源码：

struct InitOnStartupMarker {
  constexpr InitOnStartupMarker operator<<(InitOnStartupMarker) const {
    return *this;
  }
 
  template <typename T>
  constexpr InitOnStartupMarker operator<<(T&& v) const {
    return std::forward<T>(v)();
  }
};
InitOnStartupMarker 是一个结构，而且如我们所料，在InitOnStartupMarker的两个构造函数中都有对 << 的重载。源码中的用法应该是第二个构造函数的用法，在这个构造函数中， << 接收一个右值（T&& V, 是右值的泛型）。std::forward 是完美转发，表示不改变输入是左值还是右值。相当于:tensorflow::register_op::OpDefBuilderWrapper("test")()  在这里完成了OpDefBuilderWrapper这个函数对象的调用。触发了op的注册。

InferenceContext
InferenceContext用于在注册op时提前做形状推断时的输入，这里有一个概念非常容易混淆，一定要弄清楚。不同于OpDefBuilder， OpDefBuilderWrapper 这些在算子注册时候的类，InferenceContext是一个在算子调用的时候生成的类。也就是说InferenceContext 不在算子注册的时候生成，而是用户在调用OpDefBuilder 的时候生成！！重要的事说三遍

InferenceContext 不在算子注册的时候生成，而是用户在调用OpDefBuilder 的时候生成！

InferenceContext 不在算子注册的时候生成，而是用户在调用OpDefBuilder 的时候生成！

InferenceContext 不在算子注册的时候生成，而是用户在调用OpDefBuilder 的时候生成！

这也好理解，SetShapeFn 本身就是一个接口，肯定是有具体数据传进来的时候才有意义。理解了这一点，后面才能理解。

InferenceContext和很多已经定义好的接口都属作用域shape_inference，shape_inference中包含了很多接口和函数，例如已经定义好的shape_inference::MatMulShape，这些也包含了形状推断用到的所有信息。要想更好的了解InferenceContext 必须要要了解shape_inference 中的一些对象。

shape_inference 的具体代码可以见：tensorflow/core/framework/shape_inference.h， 其中包含6个主要对象： shapehandle，Shape，DimenionHandle，Dimension ； InferenceContext， shapemanager。

其中shapehandle，Shape，DimenionHandle，Dimension 四个对象的包含关系如下：



ShapeHandle：只有一个主要的属性就ptr，就是 Shape的指针

Shape 有两个主要属性，rank和dims， rank 是int类型表示tesnor的有多少个维度，dim表示类型是vector，表示这三个维度的宽度对象DimensionHandle

DimensionHandle 也只有一个主要属性，ptr，是指向Dimension的指针

Dimension ： 则只有一个主要属性是int，表示宽度。

这样层层嵌套的关系有点像tensorflow的中的featurecolumn的结构。

举例来说  一个tensor的维度是[3,4,5] 那么 这个tensor的shape类的rank 就是3， DimensionHandle 本质就是三个指针，这三个指针指向的值就是3,4,5。而整个shape的指针就是shapehandle。

上面铺垫完了，正式讲InferenceContext，InferenceContext是一个对象，构造函数中，最重要的输入是

const std::vector<ShapeHandle>& input_shapes
const std::vector<const Tensor*>& input_tensors
其中input_shapes表示输入的形状，input_tensors 表示输入的tensor。

最重要的几个属性如下

  std::vector<ShapeHandle> inputs_;
  std::vector<const Tensor*> input_tensors_;
  std::vector<bool> requested_input_tensor_;
  std::vector<ShapeHandle> outputs_;
  // Can have fewer elements than inputs_.
  std::vector<ShapeHandle> input_tensors_as_shapes_;
  std::vector<bool> requested_input_tensor_as_partial_shape_;

inputs_和outputs_ 是最核心的两个属性，inputs_在构造函数中被赋值，就是上面传入的input_shapes。 outputs_一开始为空，整个SetShapeFn 就是为了给outputs_  赋值，赋值以后形状推断即结束，outputs_ 就是输出的形状。这个形状会流向下一个节点，由下一个节点判断形状是否有问题。

算子生成和算子注册
我们自定义一个算子的本质是为了，利用这个算子生成一个op，所以必须要知道我们自定义的算子是怎么生成op的。

对于前面的过程我们已经理清楚了，过程是这样的：

调用REGISTER_OP 生成一个 OpDefBuilderWrapper 类，给OpDefBuilderWrapper 传入我们算子的信息：函数名，输入输出名称、格式，形状推断方法。OpDefBuilderWrapper 回生成一个OpDefBuilder，在load动态链接库时自动注册到tensorflow算子库中。

然后用户在Python侧调用这个函数的时候，就等于在调用相应的OpDefBuilder，这里需要注意的是，OpDefBuilder不是简单地生成一个OpDef，而是会生成一个结构体OpRegistrationData， 这个结构体包括两个成员：OpDef， OpShapeInferenceFn，其中OpShapeInferenceFn 是一个以InferenceContext 为输入的函数。用户会输入具体的tensor，输入的内容传给OpDef， OpShapeInferenceFn 完成op的定义以及形状的推断

借用 

HaoBBNuanMM
https://blog.csdn.net/HaoBBNuanMM

的一张图片就是 



op对象以proto的形式定义，定义如下：

syntax = "proto3";
 
package tensorflow;
 
import "tensorflow/core/framework/attr_value.proto";
import "tensorflow/core/framework/full_type.proto";
import "tensorflow/core/framework/resource_handle.proto";
import "tensorflow/core/framework/types.proto";
 
option cc_enable_arenas = true;
option java_outer_classname = "OpDefProtos";
option java_multiple_files = true;
option java_package = "org.tensorflow.framework";
option go_package = "github.com/tensorflow/tensorflow/tensorflow/go/core/framework/op_def_go_proto";
 
// Defines an operation. A NodeDef in a GraphDef specifies an Op by
// using the "op" field which should match the name of a OpDef.
// LINT.IfChange
message OpDef {
  // Op names starting with an underscore are reserved for internal use.
  // Names should be CamelCase and match the regexp "[A-Z][a-zA-Z0-9>_]*".
  string name = 1;
 
  // For describing inputs and outputs.
  message ArgDef {
    // Name for the input/output.  Should match the regexp "[a-z][a-z0-9_]*".
    string name = 1;
 
    // Human readable description.
    string description = 2;
 
    // Describes the type of one or more tensors that are accepted/produced
    // by this input/output arg.  The only legal combinations are:
    // * For a single tensor: either the "type" field is set or the
    //   "type_attr" field is set to the name of an attr with type "type".
    // * For a sequence of tensors with the same type: the "number_attr"
    //   field will be set to the name of an attr with type "int", and
    //   either the "type" or "type_attr" field will be set as for
    //   single tensors.
    // * For a sequence of tensors, the "type_list_attr" field will be set
    //   to the name of an attr with type "list(type)".
    DataType type = 3;
    string type_attr = 4;    // if specified, attr must have type "type"
    string number_attr = 5;  // if specified, attr must have type "int"
    // If specified, attr must have type "list(type)", and none of
    // type, type_attr, and number_attr may be specified.
    string type_list_attr = 6;
 
    // The handle data for resource inputs.
    repeated ResourceHandleProto.DtypeAndShape handle_data = 7;
 
    // For inputs: if true, the inputs are required to be refs.
    //   By default, inputs can be either refs or non-refs.
    // For outputs: if true, outputs are refs, otherwise they are not.
    bool is_ref = 16;
 
    // Experimental. Full type declaration for this argument.
    // The full type specification combines type, type_attr, type_list_attr,
    // etc. into a unified representation.
    // This declaration may contain non-concrete types (for example,
    // Tensor<TypeVar<'T'>> is a valid type declaration.
    //
    // Note: this is a transient field. The long-term aim is to represent the
    // entire OpDef as a single type: a callable. In that context, this field is
    // just the type of a single argument.
    FullTypeDef experimental_full_type = 17;
  }
 
  // Description of the input(s).
  repeated ArgDef input_arg = 2;
 
  // Description of the output(s).
  repeated ArgDef output_arg = 3;
 
  // Named control outputs for this operation. Useful only for composite
  // operations (i.e. functions) which want to name different control outputs.
  repeated string control_output = 20;
 
  // Description of the graph-construction-time configuration of this
  // Op.  That is to say, this describes the attr fields that will
  // be specified in the NodeDef.
  message AttrDef {
    // A descriptive name for the argument.  May be used, e.g. by the
    // Python client, as a keyword argument name, and so should match
    // the regexp "[a-z][a-z0-9_]+".
    string name = 1;
 
    // One of the type names from attr_value.proto ("string", "list(string)",
    // "int", etc.).
    string type = 2;
 
    // A reasonable default for this attribute if the user does not supply
    // a value.  If not specified, the user must supply a value.
    AttrValue default_value = 3;
 
    // Human-readable description.
    string description = 4;
 
    // TODO(josh11b): bool is_optional?
 
    // --- Constraints ---
    // These constraints are only in effect if specified.  Default is no
    // constraints.
 
    // For type == "int", this is a minimum value.  For "list(___)"
    // types, this is the minimum length.
    bool has_minimum = 5;
    int64 minimum = 6;
 
    // The set of allowed values.  Has type that is the "list" version
    // of the "type" field above (uses the "list" field of AttrValue).
    // If type == "type" or "list(type)" above, then the "type" field
    // of "allowed_values.list" has the set of allowed DataTypes.
    // If type == "string" or "list(string)", then the "s" field of
    // "allowed_values.list" has the set of allowed strings.
    AttrValue allowed_values = 7;
  }
  repeated AttrDef attr = 4;
 
  // Optional deprecation based on GraphDef versions.
  OpDeprecation deprecation = 8;
 
  // One-line human-readable description of what the Op does.
  string summary = 5;
 
  // Additional, longer human-readable description of what the Op does.
  string description = 6;
 
  // -------------------------------------------------------------------------
  // Which optimizations this operation can participate in.
 
  // True if the operation is commutative ("op(a,b) == op(b,a)" for all inputs)
  bool is_commutative = 18;
 
  // If is_aggregate is true, then this operation accepts N >= 2
  // inputs and produces 1 output all of the same type.  Should be
  // associative and commutative, and produce output with the same
  // shape as the input.  The optimizer may replace an aggregate op
  // taking input from multiple devices with a tree of aggregate ops
  // that aggregate locally within each device (and possibly within
  // groups of nearby devices) before communicating.
  // TODO(josh11b): Implement that optimization.
  bool is_aggregate = 16;  // for things like add
 
  // Other optimizations go here, like
  //   can_alias_input, rewrite_when_output_unused, partitioning_strategy, etc.
 
  // -------------------------------------------------------------------------
  // Optimization constraints.
 
  // Ops are marked as stateful if their behavior depends on some state beyond
  // their input tensors (e.g. variable reading op) or if they have
  // a side-effect (e.g. printing or asserting ops). Equivalently, stateless ops
  // must always produce the same output for the same input and have
  // no side-effects.
  //
  // By default Ops may be moved between devices.  Stateful ops should
  // either not be moved, or should only be moved if that state can also
  // be moved (e.g. via some sort of save / restore).
  // Stateful ops are guaranteed to never be optimized away by Common
  // Subexpression Elimination (CSE).
  bool is_stateful = 17;  // for things like variables, queue
 
  // -------------------------------------------------------------------------
  // Non-standard options.
 
  // By default, all inputs to an Op must be initialized Tensors.  Ops
  // that may initialize tensors for the first time should set this
  // field to true, to allow the Op to take an uninitialized Tensor as
  // input.
  bool allows_uninitialized_input = 19;  // for Assign, etc.
 
  // Indicates whether the op implementation uses distributed communication.
  // If True, the op is allowed to return errors for network disconnection and
  // trigger TF network failure handling logics.
  bool is_distributed_communication = 21;
}
// LINT.ThenChange(
//     https://www.tensorflow.org/code/tensorflow/core/framework/op_def_util.cc)
 
// Information about version-dependent deprecation of an op
message OpDeprecation {
  // First GraphDef version at which the op is disallowed.
  int32 version = 1;
 
  // Explanation of why it was deprecated and what to use instead.
  string explanation = 2;
}
 
// A collection of OpDefs
message OpList {
  repeated OpDef op = 1;
}

————————————————
版权声明：本文为CSDN博主「kangshuangzhu」的原创文章，遵循CC 4.0 BY-SA版权协议，转载请附上原文出处链接及本声明。
原文链接：https://blog.csdn.net/kangshuangzhu/article/details/128636437