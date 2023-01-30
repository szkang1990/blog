

&ensp;&ensp;本节介绍tensorflow中的graph，在c_api.cc中有创建graph的例子，我们从这个为切入点，探索graph的使用。
# 创建一个图
&ensp;&ensp;在c_api.cc中，创建**graph**的代码如下：
```cpp
TF_Graph* TF_NewGraph() { return new TF_Graph; }
TF_Graph::TF_Graph()
    : graph(tensorflow::OpRegistry::Global()),
      refiner(graph.versions().producer(), graph.op_registry()),
      delete_requested(false),
      parent(nullptr),
      parent_inputs(nullptr) {
  // Tell the shape refiner to also run shape inference on functions.
  refiner.set_function_library_for_shape_inference(&graph.flib_def());
```
&emsp;在上面的代码中，创建通过TF_Graph()创建一个图，TF_Graph的代码文件路径是 *'tensorflow/c/c_api_internal.h'*
# TF_Graph
&emsp; TF_Graph的源码如下：
```cpp
struct TF_Graph {
  TF_Graph();
 
  mutable tensorflow::mutex mu;
  tensorflow::Graph graph TF_GUARDED_BY(mu);
 
  // Runs shape inference.
  tensorflow::ShapeRefiner refiner TF_GUARDED_BY(mu);
 
  // Maps from name of an operation to the Node* in 'graph'.
  std::unordered_map<tensorflow::string, tensorflow::Node*> name_map
      TF_GUARDED_BY(mu);
 
  // The keys of this map are all the active sessions using this graph. Each
  // value records whether the graph has been mutated since the corresponding
  // session has been run (this is detected in RecordMutation function). If the
  // string is empty, no mutation has occurred. Otherwise the string is a
  // description of the mutation suitable for returning to the user.
  //
  // Sessions are added to this map in TF_NewSession, and removed in
  // TF_DeleteSession.
  // TF_Graph may only / must be deleted when
  //   sessions.size() == 0 && delete_requested == true
  //
  // TODO(b/74949947): mutations currently trigger a warning instead of a bad
  // status, this should be reverted when possible.
  tensorflow::gtl::FlatMap<TF_Session*, tensorflow::string> sessions
      TF_GUARDED_BY(mu);
  bool delete_requested TF_GUARDED_BY(mu);  // set true by TF_DeleteGraph
 
  // Used to link graphs contained in TF_WhileParams to the parent graph that
  // will eventually contain the full while loop.
  TF_Graph* parent;
  TF_Output* parent_inputs;
};
```
TF_Graph是一个结构体，核心属性只有两个：

> <font size = 3>tensorflow::Graph graph </font>
> <font size = 3>tensorflow::ShapeRefiner refiner</font>

首先介绍Graph
## Graph
graph的源码在tensorflow/core/graph/graph.cc 和 tensorflow/core/graph/graph.h，源码如下：
```cpp
class Graph {
 public:
  // Constructs a graph with a single SOURCE (always id kSourceId) and a
  // single SINK (always id kSinkId) node, and an edge from SOURCE->SINK.
  //
  // The graph can hold ops found in the registry. `ops`s lifetime must be at
  // least that of the constructed graph's.
  explicit Graph(const OpRegistryInterface* ops);
 
  // Constructs a graph with a single SOURCE (always id kSourceId) and a
  // single SINK (always id kSinkId) node, and an edge from SOURCE->SINK.
  //
  // The graph can hold ops found in `flib_def`. Unlike the constructor taking
  // an OpRegistryInterface, this constructor copies the function definitions in
  // `flib_def` so its lifetime may be shorter than that of the graph's. The
  // OpRegistryInterface backing `flib_def` must still have the lifetime of the
  // graph though.
  explicit Graph(const FunctionLibraryDefinition& flib_def);
 
  ~Graph();
 
  // Clone the current graph into a new one.
  std::unique_ptr<Graph> Clone();
 
  static const int kControlSlot;
 
  // The GraphDef version range of this graph (see graph.proto).
  const VersionDef& versions() const;
  void set_versions(const VersionDef& versions);
 
  // Adds a new node to this graph, and returns it. Infers the Op and
  // input/output types for the node. *this owns the returned instance.
  // Returns nullptr and sets *status on error.
  Node* AddNode(NodeDef node_def, Status* status);
 
  // Same as above, but using StatusOr. This method is always preferred.
  StatusOr<Node*> AddNode(NodeDef node_def);
 
  // Copies *node, which may belong to another graph, to a new node,
  // which is returned.  Does not copy any edges.  *this owns the
  // returned instance.
  Node* CopyNode(const Node* node);
 
  // Removes a node from this graph, including all edges from or to it.
  // *node should not be accessed after calling this function.
  // REQUIRES: node->IsOp()
  void RemoveNode(Node* node);
 
  void Copy(const Graph& src);
 
  // Removes all nodes from this graph, including all edges from or to them.
  // No Node* references to the Graph are valid post.
  void Clear();
 
  // Adds an edge that connects the xth output of `source` to the yth input of
  // `dest` and returns it. Does not update dest's NodeDef.
  const Edge* AddEdge(Node* source, int x, Node* dest, int y);
 
  // Adds a control edge (no data flows along this edge) that connects `source`
  // to `dest`. If `dest`s NodeDef is missing the corresponding control input,
  // adds the control input.
  //
  // If such a control edge already exists and `allow_duplicates` is false, no
  // edge is added and the function returns nullptr. Otherwise the edge is
  // unconditionally created and returned. The NodeDef is not updated if
  // `allow_duplicates` is true.
  // TODO(skyewm): // TODO(skyewm): allow_duplicates is needed only by
  // graph_partition.cc. Figure out if we can do away with it.
  const Edge* AddControlEdge(Node* source, Node* dest,
                             bool allow_duplicates = false);
 
  // Removes edge from the graph. Does not update the destination node's
  // NodeDef.
  // REQUIRES: The edge must exist.
  void RemoveEdge(const Edge* edge);
 
  // Removes control edge `edge` from the graph. Note that this also updates
  // the corresponding NodeDef to reflect the change.
  // REQUIRES: The control edge must exist.
  void RemoveControlEdge(const Edge* e);
 
  // Updates the input to a node.  The existing edge to `dst` is removed and an
  // edge from `new_src` to `dst` is created. The NodeDef associated with `dst`
  // is also updated.
  Status UpdateEdge(Node* new_src, int new_src_index, Node* dst, int dst_index);
 
  // Like AddEdge but updates dst's NodeDef. Used to add an input edge to a
  // "While" op during gradient construction, see AddInputWhileHack in
  // python_api.h for more details.
  Status AddWhileInputHack(Node* new_src, int new_src_index, Node* dst);
 
  // Adds the function and gradient definitions in `fdef_lib` to this graph's op
  // registry. Ignores duplicate functions, and returns a bad status if an
  // imported function differs from an existing function or op with the same
  // name.
  Status AddFunctionLibrary(const FunctionDefLibrary& fdef_lib);
 
  // The number of live nodes in the graph.
  //
  // Because nodes can be removed from the graph, num_nodes() is often
  // smaller than num_node_ids(). If one needs to create an array of
  // nodes indexed by node ids, num_node_ids() should be used as the
  // array's size.
  int num_nodes() const { return num_nodes_; }
 
  // The number of live nodes in the graph, excluding the Source and Sink nodes.
  int num_op_nodes() const {
    DCHECK_GE(num_nodes_, 2);
    return num_nodes_ - 2;
  }
 
  // The number of live edges in the graph.
  //
  // Because edges can be removed from the graph, num_edges() is often
  // smaller than num_edge_ids(). If one needs to create an array of
  // edges indexed by edge ids, num_edge_ids() should be used as the
  // array's size.
  int num_edges() const { return num_edges_; }
 
  // Serialize the nodes starting at `from_node_id` to a GraphDef.
  void ToGraphDefSubRange(GraphDef* graph_def, int from_node_id) const;
 
  // Serialize to a GraphDef.
  void ToGraphDef(GraphDef* graph_def) const;
 
  // This version can be called from debugger to inspect the graph content.
  // Use the previous version outside debug context for efficiency reasons.
  //
  // Note: We do not expose a DebugString() API, since GraphDef.DebugString() is
  // not defined in some TensorFlow builds.
  GraphDef ToGraphDefDebug() const;
 
  // Generate new node name with the specified prefix that is unique
  // across this graph.
  std::string NewName(StringPiece prefix);
 
  // Access to the list of all nodes.  Example usage:
  //   for (Node* node : graph.nodes()) { ... }
  gtl::iterator_range<NodeIter> nodes() const;
 
  // Access to the list of all nodes, excluding the Source and Sink nodes.
  gtl::iterator_range<NodeIter> op_nodes() const;
 
  // Returns one more than the maximum id assigned to any node.
  int num_node_ids() const { return nodes_.size(); }
 
  // Returns the node associated with an id, or nullptr if no node
  // with that id (the node with that id was removed and the id has
  // not yet been re-used). *this owns the returned instance.
  // REQUIRES: 0 <= id < num_node_ids().
  Node* FindNodeId(int id) const { return nodes_[id]; }
 
  // Returns one more than the maximum id assigned to any edge.
  int num_edge_ids() const { return edges_.size(); }
 
  // Returns the Edge associated with an id, or nullptr if no edge
  // with that id (the edge with that id was removed and the id has
  // not yet been re-used). *this owns the returned instance.
  // REQUIRES: 0 <= id < num_edge_ids().
  const Edge* FindEdgeId(int id) const { return edges_[id]; }
 
  // Access to the set of all edges.  Example usage:
  //   for (const Edge* e : graph.edges()) { ... }
  GraphEdgesIterable edges() const { return GraphEdgesIterable(edges_); }
 
  // The pre-defined nodes.
  enum { kSourceId = 0, kSinkId = 1 };
  Node* source_node() const { return FindNodeId(kSourceId); }
  Node* sink_node() const { return FindNodeId(kSinkId); }
 
  const OpRegistryInterface* op_registry() const { return &ops_; }
  const FunctionLibraryDefinition& flib_def() const { return ops_; }
 
  // TODO(mdan): This is only used by control_flow_deps_o_chains. Remove?
  FunctionLibraryDefinition* mutable_flib_def() { return &ops_; }
 
  void CheckDeviceNameIndex(int index) {
    DCHECK_GE(index, 0);
    DCHECK_LT(index, static_cast<int>(device_names_.size()));
  }
 
  int InternDeviceName(const std::string& device_name);
 
  const std::string& get_assigned_device_name(const Node& node) const {
    return device_names_[node.assigned_device_name_index()];
  }
 
  void set_assigned_device_name_index(Node* node, int device_name_index) {
    CheckDeviceNameIndex(device_name_index);
    node->assigned_device_name_index_ = device_name_index;
  }
 
  void set_assigned_device_name(Node* node, const std::string& device_name) {
    node->assigned_device_name_index_ = InternDeviceName(device_name);
  }
 
  // Returns OK if `node` is non-null and belongs to this graph
  Status IsValidNode(const Node* node) const;
 
  // Returns OK if IsValidNode(`node`) and `idx` is a valid output.  Does not
  // accept control outputs.
  Status IsValidOutputTensor(const Node* node, int idx) const;
 
  // Returns OK if IsValidNode(`node`) and `idx` a valid input.  Does not accept
  // control inputs.
  Status IsValidInputTensor(const Node* node, int idx) const;
 
  // Create and return a new WhileContext owned by this graph. This is called
  // when a new while loop is created. `frame_name` must be unique among
  // WhileContexts in this graph.
  Status AddWhileContext(StringPiece frame_name, std::vector<Node*> enter_nodes,
                         std::vector<Node*> exit_nodes,
                         OutputTensor cond_output,
                         std::vector<OutputTensor> body_inputs,
                         std::vector<OutputTensor> body_outputs,
                         WhileContext** result);
 
  // Builds a node name to node pointer index for all nodes in the graph.
  std::unordered_map<string, Node*> BuildNodeNameIndex() const;
 
  absl::optional<std::vector<bool>>& GetConstArgIndicesCache() const {
    return const_arg_indices_cache_;
  }
 
  // TODO(kkb): Add to the constructor when it becomes managable.
  // Sets the graph construction context.
  void SetConstructionContext(ConstructionContext construction_context) {
    construction_context_ = construction_context;
  }
 
  // TODO(kkb): Rename to `GetConstructionContext` once we're comfortable
  // making this stable and make it available widely.
  // Returns the graph construction context. It's `kUnknown` if not set.
  ConstructionContext GetConstructionContextInternal() const {
    return construction_context_;
  }
 
  // TODO(josh11b): uint64 hash() const;
 
 private:
  // If cost_node is non-null, then cost accounting (in CostModel)
  // will be associated with that node rather than the new one being
  // created.
  //
  // Ownership of the returned Node is not transferred to caller.
  Node* AllocateNode(std::shared_ptr<NodeProperties> props,
                     const Node* cost_node, Node::NodeClass node_class);
  void ReleaseNode(Node* node);
  // Insert edge in free_edges_ for possible reuse.
  void RecycleEdge(const Edge* edge);
  // Registry of all known ops, including functions.
  FunctionLibraryDefinition ops_;
 
  // GraphDef versions
  const std::unique_ptr<VersionDef> versions_;
 
  // Allocator which will give us good locality.
  core::Arena arena_;
 
  // Map from node ids to allocated nodes.  nodes_[id] may be nullptr if
  // the node with that id was removed from the graph.
  std::vector<Node*> nodes_;
 
  // Number of nodes alive.
  int64_t num_nodes_ = 0;
 
  // Map from edge ids to allocated edges.  edges_[id] may be nullptr if
  // the edge with that id was removed from the graph.
  std::vector<Edge*> edges_;
 
  // The number of entries in edges_ that are not nullptr.
  int num_edges_ = 0;
 
  // Allocated but free nodes and edges.
  std::vector<Node*> free_nodes_;
  std::vector<Edge*> free_edges_;
 
  // For generating unique names.
  int name_counter_ = 0;
 
  // In most graphs, the number of unique values used for the
  // Node::assigned_device_name() property is quite small.  If the graph is
  // large, then this duplication of values can consume a significant amount of
  // memory.  Instead, we represent the same information using an interning
  // table, which consists of a vector of unique strings (device_names_), as
  // well a map (device_names_map_) from unique strings to indices within the
  // unique string table.
  //
  // The InternDeviceName() method handles adding a new entry into the table,
  // or locating the index of an existing entry.
  //
  // The fact that Node::assigned_device_name() is implemented using an
  // interning table is intentionally public.  This allows algorithms that
  // frequently access this field to do so efficiently, especially for the case
  // where the assigned_device_name of one Node is copied directly from that
  // of another Node.
 
  // A table of the unique assigned device names.  Indices do NOT correspond
  // to node IDs.  Index 0 is always the empty string.
  std::vector<string> device_names_;
 
  // Maps unique device names to indices within device_names_[i].
  std::unordered_map<string, int> device_names_map_;
 
  // All the while contexts owned by this graph, keyed by frame name,
  // corresponding to all the while loops contained in this graph (including
  // nested loops). The stored contexts are usually accessed via
  // AddWhileContext() or Node::while_ctx(), but this manages the lifetime.
  std::map<string, WhileContext> while_ctxs_;
 
  // Cache of the indices of the arguments which need to be constant for the XLA
  // compilation.
  mutable absl::optional<std::vector<bool>> const_arg_indices_cache_;
 
  // Indicates the context that this Graph instance is constructed.
  ConstructionContext construction_context_ = ConstructionContext::kNotTracked;
 
  TF_DISALLOW_COPY_AND_ASSIGN(Graph);
};
```
其核心的属性为
> const std::unique_ptr<VersionDef> versions_;
core::Arena arena_;
std::vector<Node*> nodes_;
FunctionLibraryDefinition ops_;
std::vector<Edge*> edges_;

&emsp;这几个属性都很好理解，一个graph核心的属性就是graph的Node，edge，此外还有graph的版本这个和tensorflow的迭代有关，arena用于给graph分配内存，op用于添加op。op的数据类型是对象FunctionLibraryDefinition，后面会做详细介绍。

&emsp;graph的构造函数如下：
```cpp

Graph::Graph(const OpRegistryInterface* ops)
    : ops_(ops, FunctionDefLibrary()),
      versions_(new VersionDef),
      arena_(8 << 10 /* 8kB */) {
  versions_->set_producer(TF_GRAPH_DEF_VERSION);
  versions_->set_min_consumer(TF_GRAPH_DEF_VERSION_MIN_CONSUMER);
 
  // Initialize the name interning table for assigned_device_name.
  device_names_.push_back("");
  DCHECK_EQ(0, InternDeviceName(""));
 
  // Source and sink have no endpoints, just control edges.
  NodeDef def;
  def.set_name("_SOURCE");
  def.set_op("NoOp");
  Status status;
  Node* source = AddNode(def, &status);
  TF_CHECK_OK(status);
  CHECK_EQ(source->id(), kSourceId);
 
  def.set_name("_SINK");
  Node* sink = AddNode(def, &status);
  TF_CHECK_OK(status);
  CHECK_EQ(sink->id(), kSinkId);
 
  AddControlEdge(source, sink);
}
```
&emsp;构造函数中只传入了三个入参
> const std::unique_ptr<VersionDef> versions_;
core::Arena arena_;
std::vector<Node*> nodes_;

&emsp;同时再tensorflow的图中，必要要有一个起点source node和终点sink node。所以在构造函数的函数体中，通过addNode添加了两个Node：“_SINK”和“_SOURCE”

&emsp;addNode是Graph中一个非常重要的函数，这里着重介绍一下，addNode的源码如下：
```cpp
Node* Graph::AddNode(NodeDef node_def, Status* status) {
  const OpRegistrationData* op_reg_data;
  status->Update(ops_.LookUp(node_def.op(), &op_reg_data));
  if (!status->ok()) return nullptr;
 
  DataTypeVector inputs;
  DataTypeVector outputs;
  status->Update(
      InOutTypesForNode(node_def, op_reg_data->op_def, &inputs, &outputs));
  if (!status->ok()) {
    *status = AttachDef(*status, node_def);
    return nullptr;
  }
 
  Node::NodeClass node_class = op_reg_data->is_function_op
                                   ? Node::NC_FUNCTION_OP
                                   : Node::GetNodeClassForOp(node_def.op());
 
  if (node_def.has_experimental_type()) {
    VLOG(3) << "AddNode: node has type set, skipping type constructor "
            << node_def.name();
  } else {
    if (op_reg_data->type_ctor != nullptr) {
      VLOG(3) << "AddNode: found type constructor for " << node_def.name();
      Status s =
          full_type::SpecializeType(AttrSlice(node_def), op_reg_data->op_def,
                                    *(node_def.mutable_experimental_type()));
      if (!s.ok()) {
        *status = errors::InvalidArgument("type error: ", s.ToString());
        VLOG(3) << "AddNode: type inference failed for " << node_def.name()
                << ": " << s;
        return nullptr;
      }
    } else {
      VLOG(3) << "AddNode: no type constructor for " << node_def.name();
    }
  }
 
  Node* node = AllocateNode(std::make_shared<NodeProperties>(
                                &op_reg_data->op_def, std::move(node_def),
                                inputs, outputs, op_reg_data->fwd_type_fn),
                            nullptr, node_class);
  return node;
}
 
 
Node* Graph::AllocateNode(std::shared_ptr<NodeProperties> props,
                          const Node* cost_node, Node::NodeClass node_class) {
  Node* node = nullptr;
  if (free_nodes_.empty()) {
    node = new (arena_.Alloc(sizeof(Node))) Node;  // placement new
  } else {
    node = free_nodes_.back();
    free_nodes_.pop_back();
  }
  node->graph_ = this;
  const int id = nodes_.size();
  int cost_id = cost_node ? cost_node->cost_id() : id;
  node->Initialize(id, cost_id, std::move(props), node_class);
  nodes_.push_back(node);
  ++num_nodes_;
  return node;
}
```
大概过程是：

1. 首先查找该nodedef 是否注册，如果已被注册，则将注册信息取出，赋给一个空OpRegistrationData

2. 获取op的nodeclass，这个结果在后面的环节中要用到，nodeclass是一个枚举值，在[tensorflow之Node,NodeProperties,NodeDef](https://blog.csdn.net/kangshuangzhu/article/details/128675923) 中有介绍

3.  通过调用AllocateNode添加一个新的node， AllocateNode的入参主要一个NodeProperties  和 node_class。 NodeProperties同样在[tensorflow之Node,NodeProperties,NodeDef](https://blog.csdn.net/kangshuangzhu/article/details/128675923) 
有介绍。
4.  AllocateNode添加node的时候，首先创建一个空的node，开辟一个新的node大小的内存空间，或者利用free_nodes_的内存，free_nodes_是那些被释放清空的node。然后给这个node设置id，graph，cost_id等属性。设置属性以后通过Initialize初始化这个node， 最后把node压进graph的 nodes_ 中，并且num_nodes_计数加1.

可以看到，AllocateNode只是把node加入到graph的nodes_中，而并没有添加相应的edges

### FunctionLibraryDefinition
&emsp;Graph的构造函数中，只接受一个入参，格式是OpRegistryInterface，这个格式我们在[tensorflow之op](https://blog.csdn.net/kangshuangzhu/article/details/128636437) 有过接触，这是一个接口，在注册op的时候，我们会继承出来一个对象OpRegistry ，并且在其中存储所有tensorflow自有和用户自定义的op。在文章的最一开始便是传入了OpRegistry类。
&emsp;这个唯一的入参的入参被FunctionLibraryDefinition格式的op接受，生成一个FunctionLibraryDefinition。FunctionLibraryDefinition的源码位置在 *tensorflow/core/framework/function.h* 和 *tensorflow/core/framework/function.h* 。源码如下：
```cpp
class FunctionLibraryDefinition : public OpRegistryInterface {
 public:
  // Ops created for function arguments bear the name given by `kArgOp`; those
  // created for return values bear the name given by `kRetOp`.
  static constexpr const char* const kArgOp = "_Arg";
  static constexpr const char* const kDeviceArgOp = "_DeviceArg";
  static constexpr const char* const kRetOp = "_Retval";
  static constexpr const char* const kDeviceRetOp = "_DeviceRetval";
  static constexpr const char* const kIntsOnDeviceAttr =
      "experimental_ints_on_device";
  static constexpr const char* const kSharedRendezvousAttr =
      "shared_rendezvous";

  static constexpr const char* const kGradientOp = "SymbolicGradient";
  static constexpr const char* const kFuncAttr = "f";

  // Note: This constructor grabs `lib_def`'s lock in shared mode.
  FunctionLibraryDefinition(const FunctionLibraryDefinition& lib_def);
  FunctionLibraryDefinition(const OpRegistryInterface* default_registry,
                            const FunctionDefLibrary& lib_def = {});
  ~FunctionLibraryDefinition() override;

  FunctionLibraryDefinition& operator=(const FunctionLibraryDefinition&) =
      delete;

  // Returns True if the library contains `func`, False otherwise.
  bool Contains(const std::string& func) const;

  // Returns nullptr if "func" is not defined in "lib_def". Otherwise,
  // returns its definition proto.
  //
  // NB: This function returns a borrowed pointer, which can be invalidated by a
  // subsequent call to `ReplaceFunction()` with the given name.
  const FunctionDef* Find(const std::string& func) const TF_LOCKS_EXCLUDED(mu_);

  // Adds function definition 'fdef' to this function library.
  // Returns status 'ok' on success, or error otherwise. This is a no-op if
  // 'fdef' already exists in this function library.
  // If 'fdef' is successfully added to the library, it will be accessible
  // from 'LookUp' and included in the proto returned by 'ToProto'.
  // This operation is atomic.
  //
  // Associates `graph` with a function `func_name`. Lifetime assumption:
  // `graph` has to outlive all instantiated graphs.
  Status AddFunctionDef(const FunctionDef& fdef,
                        const StackTracesMap& stack_traces = {})
      TF_LOCKS_EXCLUDED(mu_);

  // Adds gradient definition 'grad' to this function library.
  // This is a no-op if 'grad' already exists in this function library.
  // If 'grad' is successfully added, it will be accessible via 'FindGradient'
  // and included in the proto returned by 'ToProto'.
  // This operation is atomic.
  Status AddGradientDef(const GradientDef& grad) TF_LOCKS_EXCLUDED(mu_);

  // Replaces the function corresponding to `func` with `fdef`. Returns
  // a non-OK status if "func" was not found in the library, OK otherwise.
  // Please be careful when replacing function: make sure all previous pointers
  // returned by `Find()` are no longer in use.
  Status ReplaceFunction(const std::string& func, const FunctionDef& fdef,
                         const StackTracesMap& stack_traces = {})
      TF_LOCKS_EXCLUDED(mu_);

  // Replaces the gradient corresponding to `grad.function_name()`. Returns
  // a non-OK status if "grad.function_name()" was not found in the library, OK
  // otherwise.
  Status ReplaceGradient(const GradientDef& grad) TF_LOCKS_EXCLUDED(mu_);

  // Removes the function corresponding to 'func'. Returns a non-OK status if
  // 'func' was not found in the library, OK otherwise.
  // Please be careful when removing function: make sure there are no other
  // nodes using the function, and all previous pointers returned by `Find()`
  // are no longer in use.
  Status RemoveFunction(const std::string& func) TF_LOCKS_EXCLUDED(mu_);

  // Removes all the functions and gradient functions.
  void Clear() TF_LOCKS_EXCLUDED(mu_);

  // Adds the functions and gradients in 'other' to this function library.
  // Duplicate functions and gradients are ignored.
  // This operation is atomic.
  Status AddLibrary(const FunctionLibraryDefinition& other)
      TF_LOCKS_EXCLUDED(mu_);

  // Adds the functions and gradients in 'lib_def' to this function library.
  // Duplicate functions and gradients are ignored.
  // This operation is atomic.
  Status AddLibrary(const FunctionDefLibrary& lib_def) TF_LOCKS_EXCLUDED(mu_);

  // If the gradient function for 'func' is specified explicitly in
  // the library, returns the gradient function name.  Otherwise,
  // returns an empty string.
  std::string FindGradient(const std::string& func) const
      TF_LOCKS_EXCLUDED(mu_);

  // OpRegistryInterface method. Useful for constructing a Graph.
  //
  // If "op" is defined in the library, returns its signature.
  // Otherwise, assume "op" is a primitive op and returns its op
  // signature and shape inference function.
  //
  // NB: This function outputs a borrowed pointer, which can be invalidated by a
  // subsequent call to `ReplaceFunction()` with the given name.
  Status LookUp(const std::string& op_type_name,
                const OpRegistrationData** op_reg_data) const override
      TF_LOCKS_EXCLUDED(mu_);

  // Generates new function name with the specified prefix that is unique
  // across this library.
  std::string UniqueFunctionName(StringPiece prefix) const
      TF_LOCKS_EXCLUDED(mu_);

  // Given a node def 'ndef', inspects attributes of the callee
  // function to derive the attribute 'value' for 'attr'. Returns OK
  // iff the attribute is given by the function's definition.
  // TODO(irving): Remove; keep only the const Node& version.
  template <typename T>
  Status GetAttr(const NodeDef& ndef, const std::string& attr, T* value) const;

  // Given a node, inspects attributes of the callee function to derive the
  // attribute 'value' for 'attr'. Returns OK iff the attribute is given by the
  // function's definition.
  template <typename T>
  Status GetAttr(const Node& node, const std::string& attr, T* value) const;

  // Returns a proto representation of the state of this function library.
  FunctionDefLibrary ToProto() const TF_LOCKS_EXCLUDED(mu_);

  size_t num_functions() const {
    tf_shared_lock l(mu_);
    return function_defs_.size();
  }

  // Returns all the function names in the FunctionLibraryDefinition.
  std::vector<string> ListFunctionNames() const TF_LOCKS_EXCLUDED(mu_);

  const OpRegistryInterface* default_registry() const {
    return default_registry_;
  }
  void set_default_registry(const OpRegistryInterface* registry) {
    default_registry_ = registry;
  }

  // Returns a copy of `*this` with only the subset of functions that are
  // reachable from the nodes of `graph` or `func`.
  FunctionLibraryDefinition ReachableDefinitions(const GraphDef& graph) const;
  FunctionLibraryDefinition ReachableDefinitions(const FunctionDef& func) const;

  // Copies the function named `func` from `other` to this
  // FunctionLibraryDefinition.
  // REQUIRES: `this->default_registry() == other.default_registry()`.
  // Returns OK on success, or error otherwise. This is a no-op if a function
  // name `func` already exists in this function library, and has the same
  // implementation as in `other`. If the implementations conflict, an invalid
  // argument error is returned.
  Status CopyFunctionDefFrom(const std::string& func,
                             const FunctionLibraryDefinition& other)
      TF_LOCKS_EXCLUDED(mu_);

  // Returns graph with debug stack traces for the given function, or `nullptr`
  // if none found.
  const StackTracesMap& GetStackTraces(const std::string& func_name) const {
    tf_shared_lock l(mu_);
    std::shared_ptr<FunctionDefAndOpRegistration> entry = FindHelper(func_name);
    if (entry) {
      return entry->stack_traces;
    }
    static const auto* empty_map = new StackTracesMap;
    return *empty_map;
  }

 private:
  // Shape inference for functions is handled separately by ShapeRefiner.

  struct FunctionDefAndOpRegistration {
    explicit FunctionDefAndOpRegistration(
        const FunctionDef& fdef_in, const StackTracesMap& stack_traces = {});

    const FunctionDef fdef;
    const OpRegistrationData op_registration_data;
    const StackTracesMap stack_traces;
  };

  std::shared_ptr<FunctionDefAndOpRegistration> FindHelper(
      const string& func) const TF_SHARED_LOCKS_REQUIRED(mu_);
  std::string FindGradientHelper(const std::string& func) const
      TF_SHARED_LOCKS_REQUIRED(mu_);

  Status AddHelper(std::shared_ptr<FunctionDefAndOpRegistration> registration,
                   bool* added) TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Same as AddFunctionDef/AddGradientDef except these methods set
  // `added` to true if the `fdef`/`grad` were actually added to this.
  Status AddFunctionDefHelper(const FunctionDef& fdef,
                              const StackTracesMap& stack_traces, bool* added)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);
  Status AddGradientDefHelper(const GradientDef& grad, bool* added)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Helper function for GetAttr. Returns the FunctionDef* to get the
  // attr from.
  const FunctionDef* GetAttrImpl(const NodeDef& ndef) const
      TF_LOCKS_EXCLUDED(mu_);

  // Remove all functions in `funcs` and all gradients of functions in
  // `funcs_with_grads` from this library.
  Status Remove(const std::vector<string>& funcs,
                const std::vector<string>& funcs_with_grads)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Remove `func` from the library. Returns non-OK Status unless `func` is in
  // the library. This should only be called when there is a guarantee that the
  // function being removed hasn't been retrieved with `Find`.
  Status RemoveFunctionHelper(const std::string& func)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  // Remove gradient of function `func` from the library. Returns non-OK Status
  // unless `func` has a gradient.
  Status RemoveGradient(const std::string& func)
      TF_EXCLUSIVE_LOCKS_REQUIRED(mu_);

  mutable mutex mu_;
  const OpRegistryInterface* default_registry_;
  gtl::FlatMap<string, std::shared_ptr<FunctionDefAndOpRegistration>>
      function_defs_ TF_GUARDED_BY(mu_);
  gtl::FlatMap<string, string> func_grad_ TF_GUARDED_BY(mu_);
};
```
&emsp;可以看到FunctionLibraryDefinition和OpRegistry都是继承自OpRegistryInterface，所以用法应该是类似的。实际上graph还有一个构造函数，就可以直接接受一个FunctionLibraryDefinition类型额入参：
```cpp
Graph::Graph(const FunctionLibraryDefinition& flib_def)
    : Graph(flib_def.default_registry()) {
  // Need a new-enough consumer to support the functions we add to the graph.
  if (flib_def.num_functions() > 0 && versions_->min_consumer() < 12) {
    versions_->set_min_consumer(12);
  }
  Status s = ops_.AddLibrary(flib_def);
  CHECK(s.ok()) << s.error_message();
}
```

核心属性
> const OpRegistryInterface* default_registry_;
> gtl::FlatMap<string, std::shared_ptr\<FunctionDefAndOpRegistration\>>&ensp;function_defs_

在上面的两个核心属性中，都是FunctionLibraryDefinition的构造函数用到的属性，构造函数如下：
```cpp
FunctionLibraryDefinition::FunctionLibraryDefinition(
    const OpRegistryInterface* default_registry,
    const FunctionDefLibrary& def_lib)
    : default_registry_(default_registry),
      function_defs_(def_lib.function_size()) {
  for (const auto& fdef : def_lib.function()) {
    // The latter function definition wins.
    auto& ptr = function_defs_[fdef.signature().name()];
    ptr.reset(new FunctionDefAndOpRegistration(fdef));
  }
  for (const auto& grad : def_lib.gradient()) {
    func_grad_[grad.function_name()] = grad.gradient_func();
  }
}
```
在grap的构造函数中，闯入的形参分别是
>ops_(ops, FunctionDefLibrary())
>ops是静态变量OpRegistry，记录了所有的op注册信息。

&emsp;这里可能有一些困惑，我们在[tensorflow源码精读之op](https://mp.csdn.net/mp_blog/creation/editor/128636437)中已经介绍了，op在导入的tensorflow的时候已经把所有的op都注册在OpRegistry了。为什么要把这个静止变量赋给default_registry_呢？我是这么理解的，在一个图的生命周期内，我们必须要保证op的定义是一致的，否则创建一个op的时候形状是2x2， 而运行的时候这个op被修改成了3x3，那就会报错。而OpRegistry在任何时间都可以被修改。所以为了防止op在graph的生命周期中被修改，在创建图的时候就把OpRegistry复制到graph中，且是const形式，不允许修改。

### Arena
&emsp;Arean是一个针对graph的内存分配的对象，用于graph的内存使用记录和内存分配（不是tensor的内存分配）。这个对象一般不需要修改，自己开发也基本用不到。因为在在函数allocateNode中有用到，简单记录一下源码如下：
```cpp
class Arena {
 public:
  // Allocates a thread-compatible arena with the specified block size.
  explicit Arena(const size_t block_size);
  ~Arena();

  char* Alloc(const size_t size) {
    return reinterpret_cast<char*>(GetMemory(size, 1));
  }

  char* AllocAligned(const size_t size, const size_t alignment) {
    return reinterpret_cast<char*>(GetMemory(size, alignment));
  }

  void Reset();

// This should be the worst-case alignment for any type.  This is
// good for IA-32, SPARC version 7 (the last one I know), and
// supposedly Alpha.  i386 would be more time-efficient with a
// default alignment of 8, but ::operator new() uses alignment of 4,
// and an assertion will fail below after the call to MakeNewBlock()
// if you try to use a larger alignment.
#ifdef __i386__
  static const int kDefaultAlignment = 4;
#else
  static constexpr int kDefaultAlignment = 8;
#endif

 protected:
  bool SatisfyAlignment(const size_t alignment);
  void MakeNewBlock(const uint32 alignment);
  void* GetMemoryFallback(const size_t size, const int align);
  void* GetMemory(const size_t size, const int align) {
    assert(remaining_ <= block_size_);                  // an invariant
    if (size > 0 && size < remaining_ && align == 1) {  // common case
      void* result = freestart_;
      freestart_ += size;
      remaining_ -= size;
      return result;
    }
    return GetMemoryFallback(size, align);
  }

  size_t remaining_;

 private:
  struct AllocatedBlock {
    char* mem;
    size_t size;
  };

  // Allocate new block of at least block_size, with the specified
  // alignment.
  // The returned AllocatedBlock* is valid until the next call to AllocNewBlock
  // or Reset (i.e. anything that might affect overflow_blocks_).
  AllocatedBlock* AllocNewBlock(const size_t block_size,
                                const uint32 alignment);

  const size_t block_size_;
  char* freestart_;  // beginning of the free space in most recent block
  char* freestart_when_empty_;  // beginning of the free space when we're empty
  // STL vector isn't as efficient as it could be, so we use an array at first
  size_t blocks_alloced_;  // how many of the first_blocks_ have been alloced
  AllocatedBlock first_blocks_[16];  // the length of this array is arbitrary
  // if the first_blocks_ aren't enough, expand into overflow_blocks_.
  std::vector<AllocatedBlock>* overflow_blocks_;

  void FreeBlocks();  // Frees all except first block

  TF_DISALLOW_COPY_AND_ASSIGN(Arena);
};
```
核心属性是

>const size_t block_size_;
>size_t remaining_;
>AllocatedBlock* AllocNewBlock(const size_t block_size, const uint32 alignment);

核心函数是
```cpp
 char* Alloc(const size_t size) {
    return reinterpret_cast<char*>(GetMemory(size, 1));
  }
```
```cpp
  void* GetMemory(const size_t size, const int align) {
    assert(remaining_ <= block_size_);                  // an invariant
    if (size > 0 && size < remaining_ && align == 1) {  // common case
      void* result = freestart_;
      freestart_ += size;
      remaining_ -= size;
      return result;
    }
    return GetMemoryFallback(size, align);
  }
  ```
 ```cpp
 void* Arena::GetMemoryFallback(const size_t size, const int alignment) {
  if (0 == size) {
    return nullptr;  // stl/stl_alloc.h says this is okay
  }

  // alignment must be a positive power of 2.
  CHECK(alignment > 0 && 0 == (alignment & (alignment - 1)));

  // If the object is more than a quarter of the block size, allocate
  // it separately to avoid wasting too much space in leftover bytes.
  if (block_size_ == 0 || size > block_size_ / 4) {
    return AllocNewBlock(size, alignment)->mem;
  }

  // Enforce alignment on freestart_ then check for adequate space,
  // which may require starting a new block.
  if (!SatisfyAlignment(alignment) || size > remaining_) {
    MakeNewBlock(alignment);
  }
  CHECK_LE(size, remaining_);

  remaining_ -= size;
  void* result = freestart_;
  freestart_ += size;

  return result;
}
```
&emsp;其中有一个reinterpret_cast的用法，这是c++中相当自由的格式转换的方法，可以把任意格式转换成任意格式，由于在使用中非常容易出错，所以一般不建议使用。


## ShapeRefiner


