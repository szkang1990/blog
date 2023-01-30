&emsp;session是连接tensorflow客户端(这里的客户端是指tensorflow的上层接口，例如我们用到的PythonAPI就可以看做是客户端)和底层的桥梁。Session 对象使我们能够访问本地机器中的设备和使用分布式 TensorFlow 运行时的远程设备。它还可缓存关于Graph 的信息，使您能够多次高效地运行同一计算。Session接受Graph参数和Options选项参数，Options参数可以指定使用的设备等信息。

&emsp;在c_api.cc中有创建的session的过程，我们从c_api.cc入手，学习一下session的源代码
## 创建一个session

```cpp
TF_Session::TF_Session(tensorflow::Session* s, TF_Graph* g)
    : session(s), graph(g), last_num_graph_nodes(0), extend_before_run(true) {}

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

c_api.cc创建session的时候主要用到了三个函数
> TF_NewSession  
> NewSession  
> TF_Session  

&emsp;我们来一一分析，TF_NewSession调用了NewSession，TF_Session。调用NewSession语句分别是

```cpp
  Session* session;
  status->status = NewSession(opt->options, &session);
  // 其中opt是TF_SessionOptions
```

&emsp;这一行的直观上是比较好理解的，首先创建一个空的session，然后在NewSession函数读取session的配置，写入session。我们在Python中，定义一个session的时候经常会这么写：

```py
graph_options = tf.GraphOptions(enable_bfloat16_sendrecv=True)
gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.9, allow_growth=True)
config = tf.ConfigProto(graph_options=graph_options, gpu_options=gpu_options, log_device_placement=False,
                        allow_soft_placement=True)
with tf.Session(config=config) as sess:
    pass
```

这里的session中的配置就是用来生成session的参数。

NewSession的定义在tensorflow/core/common_runtime/session.cc和tensorflow/core/public/session.h(这里注意一下，session的头文件和code没有在一个文件夹中)

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

NewSession的入参有两个：
>const SessionOptions& options \
>Session** out_session

很明显分别对应了上面的opt->options 和  &session。其中opt是TF_SessionOptions，所以我们先来看看TF_SessionOptions 和 SessionOptions这两个对象。
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

到此就很清晰了创建一个session的需要传入一个TF_SessionOptions，这个结构体只有一个属性SessionOptions，而SessionOptions中最核心的属性就是ConfigProto，因此本质上创建一个session就是需要传入一个ConfigProto。

接下来继续来看NewSession的源码，把其中最核心的部分难出来，可以看到就是先创建了一个SessionFactory ，然后调用了SessionFactory的NewSession函数把传入的空session：out_session填满。

```cpp
  SessionFactory* factory;
  Status s = SessionFactory::GetFactory(options, &factory);
  s = factory->NewSession(options, out_session);
```

GetFactory的源码如下

```cpp
Status SessionFactory::GetFactory(const SessionOptions& options,
                                  SessionFactory** out_factory) {
  mutex_lock l(*get_session_factory_lock());  // could use reader lock

  std::vector<std::pair<string, SessionFactory*>> candidate_factories;
  for (const auto& session_factory : *session_factories()) {
    if (session_factory.second->AcceptsOptions(options)) {
      VLOG(2) << "SessionFactory type " << session_factory.first
              << " accepts target: " << options.target;
      candidate_factories.push_back(session_factory);
    } else {
      VLOG(2) << "SessionFactory type " << session_factory.first
              << " does not accept target: " << options.target;
    }
  }

  if (candidate_factories.size() == 1) {
    *out_factory = candidate_factories[0].second;
    return Status::OK();
  } else if (candidate_factories.size() > 1) {
    // NOTE(mrry): This implementation assumes that the domains (in
    // terms of acceptable SessionOptions) of the registered
    // SessionFactory implementations do not overlap. This is fine for
    // now, but we may need an additional way of distinguishing
    // different runtimes (such as an additional session option) if
    // the number of sessions grows.
    // TODO(mrry): Consider providing a system-default fallback option
    // in this case.
    std::vector<string> factory_types;
    factory_types.reserve(candidate_factories.size());
    for (const auto& candidate_factory : candidate_factories) {
      factory_types.push_back(candidate_factory.first);
    }
    return errors::Internal(
        "Multiple session factories registered for the given session "
        "options: {",
        SessionOptionsToString(options), "} Candidate factories are {",
        absl::StrJoin(factory_types, ", "), "}. ",
        RegisteredFactoriesErrorMessageLocked());
  } else {
    return errors::NotFound(
        "No session factory registered for the given session options: {",
        SessionOptionsToString(options), "} ",
        RegisteredFactoriesErrorMessageLocked());
  }
}
```