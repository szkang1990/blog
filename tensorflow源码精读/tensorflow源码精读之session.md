&emsp;session是连接tensorflow客户端(这里的客户端是指tensorflow的上层接口，例如我们用到的PythonAPI就可以看做是客户端)和底层的桥梁。Session 对象使我们能够访问本地机器中的设备和使用分布式 TensorFlow 运行时的远程设备。它还可缓存关于Graph 的信息，使您能够多次高效地运行同一计算。Session接受Graph参数和Options选项参数，Options参数可以指定使用的设备等信息。

&emsp;在c_api.cc中有创建的session的过程，我们从c_api.cc入手，学习一下session的源代码


## TF_Session

结构体TF_Session是创建session的重要中间变量，TF_Session的结构比较简单：
```cpp
struct TF_Session {
  TF_Session(tensorflow::Session* s, TF_Graph* g);

  tensorflow::Session* session;
  TF_Graph* const graph;

  tensorflow::mutex mu TF_ACQUIRED_AFTER(TF_Graph::mu);
  int last_num_graph_nodes;

  // If true, TF_SessionRun and similar methods will call
  // ExtendSessionGraphHelper before running the graph (this is the default
  // public behavior). Can be set to false if the caller needs to call
  // ExtendSessionGraphHelper manually.
  std::atomic<bool> extend_before_run;
};

```
TF_Graph可以看graph中的一章。

TF_Session的构造函数如下，也非常简单，不多说了

```cpp
TF_Session::TF_Session(tensorflow::Session* s, TF_Graph* g)
    : session(s), graph(g), last_num_graph_nodes(0), extend_before_run(true) {}
```

## 创建一个session
在c_api.cc和c_api.h的中创建Session的代码如下：

```cpp
TF_Session* TF_NewSession(TF_Graph* graph, const TF_SessionOptions* opt,
                          TF_Status* status) {
  Session* session;
  status->status = NewSession(opt->options, &session);
  if (status->status.ok()) {
    TF_Session* new_session = new TF_Session(session, graph);
    if (graph != nullptr) {
      mutex_lock l(graph->mu);
      graph->sessions[new_session] = "";
    }
    return new_session;
  } else {
    LOG(ERROR) << status->status;
    DCHECK_EQ(nullptr, session);
    return nullptr;
  }
}
```
### NewSession函数
在上面的代码中借助NewSession函数创建了一个session，这函数的源码在tensorflow/tensorflow/core/common_runtime/session.cc和 tensorflow/core/common_runtime/session.cc和tensorflow/core/public/session.h

```cpp
Status NewSession(const SessionOptions& options, Session** out_session) {
  SessionFactory* factory;
  Status s = SessionFactory::GetFactory(options, &factory);
  if (!s.ok()) {
    *out_session = nullptr;
    LOG(ERROR) << "Failed to get session factory: " << s;
    return s;
  }
  // Starts exporting metrics through a platform-specific monitoring API (if
  // provided). For builds using "tensorflow/core/platform/default", this is
  // currently a no-op.
  session_created->GetCell()->Set(true);
  s = factory->NewSession(options, out_session);
  if (!s.ok()) {
    *out_session = nullptr;
    LOG(ERROR) << "Failed to create session: " << s;
  }
  return s;
}
```

应该注意的是，在sessfacotry中也有NewSession函数(就是上面函数体中的NewSession函数），具体可以见sessonFactory一章，这两个函数是不同的，要注意区分开。

在上面的代码中，首先调用SessionFactory::GetFactory(options, &factory)，获得和options匹配的SessionFactory，然后调用SessionFactory的NewSession函数，新建一个Session，并且赋值给入参out_session






NewSession的入参有两个：
>const SessionOptions& options \
>Session** out_session

很明显分别对应了上面的opt->options 和  &session。其中opt是TF_SessionOptions，我们可以联想到我们在写python的训练代码的时候经常会写这样的配置。
```py
graph_options = tf.GraphOptions(enable_bfloat16_sendrecv=True)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9, allow_growth=True)
config = tf.ConfigProto(graph_options=graph_options, gpu_options=gpu_options, log_device_placement=False,
                        allow_soft_placement=True)
with tf.Session(config=config) as sess:
    pass
```
这个配置就是TF_SessionOptions在python中的写法。


我们先来底层代码中看看TF_SessionOptions是什么
### TF_SessionOptions
&emsp;TF_SessionOptions的定义在tensorflow/c/c_api_internal.h中。在最新版本中的源码如下：

```cpp
struct TF_SessionOptions {
  tensorflow::SessionOptions options;
};
```

他是一个结构体而且只有一个属性就是SessionOptions，我不知道这样设计的目的是什么，也许在更早的版本中有其他的属性，然后都慢慢都迭代掉了吧。

SessionOptions的定义在tensorflow/core/common_runtime/session_options.cc 和 tensorflow/core/public/session_options.h(这里又是一个头文件和code不在同一路径下的)

```cpp
struct SessionOptions {
  /// The environment to use.
  Env* env;

  /// \brief The TensorFlow runtime to connect to.
  ///
  /// If 'target' is empty or unspecified, the local TensorFlow runtime
  /// implementation will be used.  Otherwise, the TensorFlow engine
  /// defined by 'target' will be used to perform all computations.
  ///
  /// "target" can be either a single entry or a comma separated list
  /// of entries. Each entry is a resolvable address of the
  /// following format:
  ///   local
  ///   ip:port
  ///   host:port
  ///   ... other system-specific formats to identify tasks and jobs ...
  ///
  /// NOTE: at the moment 'local' maps to an in-process service-based
  /// runtime.
  ///
  /// Upon creation, a single session affines itself to one of the
  /// remote processes, with possible load balancing choices when the
  /// "target" resolves to a list of possible processes.
  ///
  /// If the session disconnects from the remote process during its
  /// lifetime, session calls may fail immediately.
  std::string target;

  /// Configuration options.
  ConfigProto config;

  SessionOptions();
};
```

SessionOptions的结构也相当简单，只有三个属性

```cpp
Env* env;
std::string target;
ConfigProto config;
```

SessionOptions的构造函数几乎没有任何内容，这里不贴了，可以直接读代码。



最核心的自然是ConfigProto这个属性，我们Python中就有用到了。ConfigProto是由proto定义的对象,proto路径是tensorflow/core/protobuf/config.proto，其中包含了很多对设备和graph的配置，具体可以去看代码这里不贴了。

## Session

Session是Session的基类, 代码路径在tensorflow/core/public/session.h和tensorflow/core/common_runtime/session.cc。
session没有属性，绝大部分的函数都是虚函数，等到子函数去实现。少数几个实现的函数例如Session* NewSession(const SessionOptions& options) 前面都有介绍，所以就不赘述了，主要分析一下子类DirectSession，grpcSession是分布式会话，等到学习分布式的时候再做介绍。

## DirectSession

DirectSession的源码定义在tensorflow/core/common_runtime/direct_session.cc 和 tensorflow/core/common_runtime/direct_session.h


```cpp
  const SessionOptions options_;

  // Device structures.
  const std::unique_ptr<const DeviceMgr> device_mgr_;
  std::vector<Device*> devices_;  // not owned
  DeviceSet device_set_;

  // Unique session identifier.
  string session_handle_;
  mutex graph_state_lock_;
  bool graph_created_ TF_GUARDED_BY(graph_state_lock_) = false;
  bool finalized_ TF_GUARDED_BY(graph_state_lock_) = false;

  // The thread-pools to use for running ops, with a bool indicating if the pool
  // is owned.
  std::vector<std::pair<thread::ThreadPool*, bool>> thread_pools_;

  Status init_error_;  // Set to an error if construction failed.

  // If true, blocks until device has finished all queued operations in a step.
  bool sync_on_finish_ = true;

  std::vector<std::unique_ptr<FunctionInfo>> functions_
      TF_GUARDED_BY(executor_lock_);

  mutex executor_lock_;  // protects executors_
  // Holds mappings from signature to the executors that process
  // it. The reason for a level of indirection around mapped_type is
  // to guarantee address stability.
  // The map value is a shared_ptr since multiple map keys can point to the
  // same ExecutorsAndKey object.
  std::unordered_map<string, std::shared_ptr<ExecutorsAndKeys>> executors_
      TF_GUARDED_BY(executor_lock_);

  class RunCallableCallFrame;
  struct Callable {
    std::shared_ptr<ExecutorsAndKeys> executors_and_keys;
    std::shared_ptr<FunctionInfo> function_info;
    ~Callable();
  };
  mutex callables_lock_;
  int64_t next_callable_handle_ TF_GUARDED_BY(callables_lock_) = 0;
  std::unordered_map<int64_t, Callable> callables_
      TF_GUARDED_BY(callables_lock_);

  // Holds mappings from handle to partial run state.
  std::unordered_map<string, std::unique_ptr<PartialRunState>> partial_runs_
      TF_GUARDED_BY(executor_lock_);

  // This holds all the tensors that are currently alive in the session.
  SessionState session_state_;

  DirectSessionFactory* const factory_;  // not owned
  CancellationManager* cancellation_manager_;
  std::unique_ptr<CollectiveExecutorMgrInterface> collective_executor_mgr_;

  // Map of placed stateful nodes, i.e. nodes for which is_stateful()
  // is true, such as "params" and "queue" nodes.  Once placed these
  // nodes can not be moved to a different device.  Maps node names to
  // device names.
  std::unordered_map<string, string> stateful_placements_
      TF_GUARDED_BY(graph_state_lock_);

  // Execution_state; used when placing the entire graph.
  std::unique_ptr<GraphExecutionState> execution_state_
      TF_GUARDED_BY(graph_state_lock_);

  // The function library, before any rewrites or optimizations have been
  // performed. In particular, CreateGraphs() may need to modify the function
  // library; it copies and modifies the function library.
  std::unique_ptr<FunctionLibraryDefinition> flib_def_;

  // true if the Session has been Closed.
  mutex closed_lock_;
  bool closed_ TF_GUARDED_BY(closed_lock_) = false;

  // For generating unique names for this session instance.
  std::atomic<int64_t> edge_name_counter_ = {0};
  std::atomic<int64_t> handle_name_counter_ = {0};

  // For generating step ids that are unique among all sessions.
  static std::atomic_int_fast64_t step_id_counter_;

  // Global timeout for all blocking operations in this session.
  const int64_t operation_timeout_in_ms_ = 0;

  // Manages all the cost models for the graphs executed in this session.
  CostModelManager cost_model_manager_;

  // For testing collective graph key generation.
  mutex collective_graph_key_lock_;
  int64_t collective_graph_key_ TF_GUARDED_BY(collective_graph_key_lock_) = -1;

  // Run in caller's thread if RunOptions.inter_op_thread_pool is negative or
  // all of following conditions are met:
  // 1. This session doesn't own any thread pool.
  // 2. RunOptions.inter_op_thread_pool is unspecified or 0.
  // 3. This session has a single executor.
  // 4. config.inter_op_parallelism_threads is specified to negative explicitly
  //    or through environment variable TF_NUM_INTEROP_THREADS.
  // 5. RunOptions.experimental.use_run_handler_pool is unspecified or false.
  // Otherwise run in global thread pool, session owned thread pool or handler
  // pool according to other specifications of RunOptions and ConfigProto.
  bool run_in_caller_thread_ = false;
```

### DirectSession构造函数

构造函数如下：

```cpp
DirectSession::DirectSession(const SessionOptions& options,
                             const DeviceMgr* device_mgr,
                             DirectSessionFactory* const factory)
    : options_(options),
      device_mgr_(device_mgr),
      factory_(factory),
      cancellation_manager_(new CancellationManager()),
      operation_timeout_in_ms_(options_.config.operation_timeout_in_ms()) {
  const int thread_pool_size =
      options_.config.session_inter_op_thread_pool_size();
  if (thread_pool_size > 0) {
    for (int i = 0; i < thread_pool_size; ++i) {
      thread::ThreadPool* pool = nullptr;
      bool owned = false;
      init_error_.Update(NewThreadPoolFromThreadPoolOptions(
          options_, options_.config.session_inter_op_thread_pool(i), i, &pool,
          &owned));
      thread_pools_.emplace_back(pool, owned);
    }
  } else if (options_.config.use_per_session_threads()) {
    thread_pools_.emplace_back(NewThreadPoolFromSessionOptions(options_),
                               true /* owned */);
  } else {
    thread_pools_.emplace_back(GlobalThreadPool(options), false /* owned */);
    // Run locally if environment value of TF_NUM_INTEROP_THREADS is negative
    // and config.inter_op_parallelism_threads is unspecified or negative.
    static const int env_num_threads = NumInterOpThreadsFromEnvironment();
    if (options_.config.inter_op_parallelism_threads() < 0 ||
        (options_.config.inter_op_parallelism_threads() == 0 &&
         env_num_threads < 0)) {
      run_in_caller_thread_ = true;
    }
  }
  // The default value of sync_on_finish will be flipped soon and this
  // environment variable will be removed as well.
  const Status status =
      ReadBoolFromEnvVar("TF_SYNC_ON_FINISH", true, &sync_on_finish_);
  if (!status.ok()) {
    LOG(ERROR) << status.error_message();
  }
  session_handle_ =
      strings::StrCat("direct", strings::FpToString(random::New64()));
  int devices_added = 0;
  if (options.config.log_device_placement()) {
    const string mapping_str = device_mgr_->DeviceMappingString();
    string msg;
    if (mapping_str.empty()) {
      msg = "Device mapping: no known devices.";
    } else {
      msg = strings::StrCat("Device mapping:\n", mapping_str);
    }
    if (!logging::LogToListeners(msg)) {
      LOG(INFO) << msg;
    }
  }
  for (auto d : device_mgr_->ListDevices()) {
    devices_.push_back(d);
    device_set_.AddDevice(d);
    d->op_segment()->AddHold(session_handle_);

    // The first device added is special: it is the 'client device' (a
    // CPU device) from which we feed and fetch Tensors.
    if (devices_added == 0) {
      device_set_.set_client_device(d);
    }
    ++devices_added;
  }
}
```
在反转实验中，初始化了属性：
options_(options),
device_mgr_(device_mgr),
factory_(factory),
cancellation_manager_(new CancellationManager()),
operation_timeout_in_ms_(options_.config.operation_timeout_in_ms())


#### DeviceMgr

DirectSession的构造函数的 入参之一是DeviceMgr。实际上用到的是DeviceMgr的子类StaticDeviceMgr， DeviceMgr本质上是对std::vector<std::unique_ptr<Device>> devices的一层封装，把device做成各种格式的集合，以方便快速的查询或者操作。例如

```cpp
  const std::vector<std::unique_ptr<Device>> devices_;

  StringPiece CopyToBackingStore(StringPiece s);

  absl::flat_hash_set<int64_t> device_incarnation_set_;
  std::unordered_map<StringPiece, Device*, StringPieceHasher> device_map_;
  core::Arena name_backing_store_;  // Storage for keys in device_map_
  std::unordered_map<string, int> device_type_counts_;
  Device* cpu_device_;
```
这几个属性的作用一目了然，就不赘述了。

DeviceMgr的函数也是服务于对device的操作，例如：
```cpp
  void ListDeviceAttributes(
      std::vector<DeviceAttributes>* devices) const override;
  std::vector<Device*> ListDevices() const override;
  Status LookupDevice(StringPiece name, Device** device) const override;
  bool ContainsDevice(int64_t device_incarnation) const override;
  int NumDeviceType(const string& type) const override;
```
这几个函数甚至不用看代码就能猜到是在做什么，
ListDeviceAttributes 是列举素有device的DeviceAttributes到入参
ListDevices是列举所有device
LookupDevice 是查找某个设备
等。这里也不再详细看代码了。



