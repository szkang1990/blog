&emsp;从这一节开始学一下session，session是一个比较难的地方，涉及的代码量非常大。会分成几部分讲，这一节先讲SessionFactory。

&emsp;SessionFactory是用于生成session的对象，session的类型有两种，分别是DirectionSession和GrpcSession，分别对应单机训练和分布式训练。因此SessionFactory也有两种分别是DirectionSessionFactory和GrpcSessionFactory。

&emsp;DirectionSessionFactory和GrpcSessionFactory都继承自SessionFactory。各个对象之间的关系如下图所示。

![avatar](https://github.com/szkang1990/blog/blob/main/tensorflow%E6%BA%90%E7%A0%81%E7%B2%BE%E8%AF%BB/image/v2-674be7969c2aaf9726b1a43da27af45e_r.jpeg?raw=true)

&emsp;首先来看SessionFactory的源码

```cpp
class SessionFactory {
 public:
  // Creates a new session and stores it in *out_session, or fails with an error
  // status if the Session could not be created. Caller takes ownership of
  // *out_session if this returns Status::OK().
  virtual Status NewSession(const SessionOptions& options,
                            Session** out_session) = 0;

  virtual bool AcceptsOptions(const SessionOptions& options) = 0;

  // Abort and close all existing sessions, disconnecting their resources from
  // future sessions.
  //
  // Reset() allows misbehaving or slow sessions to be aborted and closed, and
  // causes their resources eventually to be released.  Reset() does not wait
  // for the computations in old sessions to cease; it merely starts the
  // process of tearing them down.  However, if a new session is started after
  // a Reset(), the new session is isolated from changes that old sessions
  // (started prior to the Reset()) may continue to make to resources, provided
  // all those resources are in containers listed in "containers".
  //
  // Old sessions may continue to have side-effects on resources not in
  // containers listed in "containers", and thus may affect future
  // sessions' results in ways that are hard to predict.  Thus, if well-defined
  // behavior is desired, is it recommended that all containers be listed in
  // "containers".
  //
  // If the "containers" vector is empty, the default container is assumed.
  // If the "containers" vector is non-empty, the default container should be
  // listed explicitly.
  //
  // Sessions that support resource containers should override this function.
  virtual Status Reset(const SessionOptions& options,
                       const std::vector<string>& containers) {
    return errors::Unimplemented("Reset()");
  }

  virtual ~SessionFactory() {}
  static void Register(const string& runtime_type, SessionFactory* factory);
  static Status GetFactory(const SessionOptions& options,
                           SessionFactory** out_factory);
};
```

核心的函数是
> 构造函数 \
>NewSession \
>AcceptsOptions \
>GetFactory 

除了对象的成员函数之外，还有一些比较重要的变量

```cpp
typedef std::unordered_map<string, SessionFactory*> SessionFactories;
SessionFactories* session_factories() {
  static SessionFactories* factories = new SessionFactories;
  return factories;
}
```
SessionFactories是一个记录SessionFactory的map。key是DIRECT_SESSION或者是GRPC_SESSION
