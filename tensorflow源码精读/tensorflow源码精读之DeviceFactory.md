&emsp;这一节我们一起学习一下tensorflow的device。经过前面的介绍，我们能看出来tensorflow使用了很多factory设计模式，Device同样沿用这种设计，tensorflow设计了各种device的factory，以生成这种device对象。

tensorflow中有很多的DeviceFactory, 各个DeviceFactory的继承关系如下：

![avatar](https://github.com/szkang1990/blog/blob/main/tensorflow%E6%BA%90%E7%A0%81%E7%B2%BE%E8%AF%BB/image/deviceFactory.png?raw=true)

其中DeviceFactory是所有所有factory的父类，由其集成出的各类factory中用的最多的就是
>ThreadPoolDeviceFactory ：用于创建ThreadPoolDevice，是CPU Device 的实现 \
GPUCompatibleCPUDevice ：用于生成GPUCompatibleCPUDevice \
BaseGPUDeviceFactory && GPUDeviceFactory： 用于生成BaseGPUDevice和GPUDevice


其他factory比较少用到，而且从上面的三个就能大概明白factory的工作原理了。


首先看一下父类DeviceFactory的定义，DeviceFactory的定义放在tensorflow\core\framework\device_factory.h和tensorflow\core\framework\device_factory.cc中。DeviceFactory对象的定义是：
```cpp
class DeviceFactory {
 public:
  virtual ~DeviceFactory() {}
  static void Register(const std::string& device_type,
                       std::unique_ptr<DeviceFactory> factory, int priority,
                       bool is_pluggable_device);
  ABSL_DEPRECATED("Use the `Register` function above instead")
  static void Register(const std::string& device_type, DeviceFactory* factory,
                       int priority, bool is_pluggable_device) {
    Register(device_type, std::unique_ptr<DeviceFactory>(factory), priority,
             is_pluggable_device);
  }
  static DeviceFactory* GetFactory(const std::string& device_type);

  // Append to "*devices" CPU devices.
  static Status AddCpuDevices(const SessionOptions& options,
                              const std::string& name_prefix,
                              std::vector<std::unique_ptr<Device>>* devices);

  // Append to "*devices" all suitable devices, respecting
  // any device type specific properties/counts listed in "options".
  //
  // CPU devices are added first.
  static Status AddDevices(const SessionOptions& options,
                           const std::string& name_prefix,
                           std::vector<std::unique_ptr<Device>>* devices);

  // Helper for tests.  Create a single device of type "type".  The
  // returned device is always numbered zero, so if creating multiple
  // devices of the same type, supply distinct name_prefix arguments.
  static std::unique_ptr<Device> NewDevice(const string& type,
                                           const SessionOptions& options,
                                           const string& name_prefix);

  // Iterate through all device factories and build a list of all of the
  // possible physical devices.
  //
  // CPU is are added first.
  static Status ListAllPhysicalDevices(std::vector<string>* devices);

  // Iterate through all device factories and build a list of all of the
  // possible pluggable physical devices.
  static Status ListPluggablePhysicalDevices(std::vector<string>* devices);

  // Get details for a specific device among all device factories.
  // 'device_index' indexes into devices from ListAllPhysicalDevices.
  static Status GetAnyDeviceDetails(
      int device_index, std::unordered_map<string, string>* details);

  // For a specific device factory list all possible physical devices.
  virtual Status ListPhysicalDevices(std::vector<string>* devices) = 0;

  // Get details for a specific device for a specific factory. Subclasses
  // can store arbitrary device information in the map. 'device_index' indexes
  // into devices from ListPhysicalDevices.
  virtual Status GetDeviceDetails(int device_index,
                                  std::unordered_map<string, string>* details) {
    return OkStatus();
  }

  // Most clients should call AddDevices() instead.
  virtual Status CreateDevices(
      const SessionOptions& options, const std::string& name_prefix,
      std::vector<std::unique_ptr<Device>>* devices) = 0;

  // Return the device priority number for a "device_type" string.
  //
  // Higher number implies higher priority.
  //
  // In standard TensorFlow distributions, GPU device types are
  // preferred over CPU, and by default, custom devices that don't set
  // a custom priority during registration will be prioritized lower
  // than CPU.  Custom devices that want a higher priority can set the
  // 'priority' field when registering their device to something
  // higher than the packaged devices.  See calls to
  // REGISTER_LOCAL_DEVICE_FACTORY to see the existing priorities used
  // for built-in devices.
  static int32 DevicePriority(const std::string& device_type);

  // Returns true if 'device_type' is registered from plugin. Returns false if
  // 'device_type' is a first-party device.
  static bool IsPluggableDevice(const std::string& device_type);
};
```

在tensorflow\core\framework\device_factory.cc 中有一些基本的函数的实现，我们来看一下：
Register函数用于deviceFactory的注册。

```cpp
// static
void DeviceFactory::Register(const string& device_type,
                             std::unique_ptr<DeviceFactory> factory,
                             int priority, bool is_pluggable_device) {
  if (!IsDeviceFactoryEnabled(device_type)) {
    LOG(INFO) << "Device factory '" << device_type << "' disabled by "
              << "TF_ENABLED_DEVICE_TYPES environment variable.";
    return;
  }
  mutex_lock l(*get_device_factory_lock());
  std::unordered_map<string, FactoryItem>& factories = device_factories();
  auto iter = factories.find(device_type);
  if (iter == factories.end()) {
    factories[device_type] = {std::move(factory), priority,
                              is_pluggable_device};
  } else {
    if (iter->second.priority < priority) {
      iter->second = {std::move(factory), priority, is_pluggable_device};
    } else if (iter->second.priority == priority) {
      LOG(FATAL) << "Duplicate registration of device factory for type "
                 << device_type << " with the same priority " << priority;
    }
  }
}
```
deviceFactory的注册信息被放在一个静态变量map中，这一点和sesssionFactory的注册是一样的。这个存储注册信息的变量定义如下：

```cpp
std::unordered_map<string, FactoryItem>& device_factories() {
  static std::unordered_map<string, FactoryItem>* factories =
      new std::unordered_map<string, FactoryItem>;
  return *factories;
}
```

这个思路和sessionFactory的注册一模一样。FactoryItem是一个结构体，定义如下：

```cpp
struct FactoryItem {
  std::unique_ptr<DeviceFactory> factory;
  int priority;
  bool is_pluggable_device;
};
```

getFactory函数,即根据key从注册map中获取DeviceFactory

```cpp
DeviceFactory* DeviceFactory::GetFactory(const string& device_type) {
  tf_shared_lock l(*get_device_factory_lock());
  auto it = device_factories().find(device_type);
  if (it == device_factories().end()) {
    return nullptr;
  } else if (!IsDeviceFactoryEnabled(device_type)) {
    LOG(FATAL) << "Device type " << device_type  // Crash OK
               << " had factory registered but was explicitly disabled by "
               << "`TF_ENABLED_DEVICE_TYPES`. This environment variable needs "
               << "to be set at program startup.";
  }
  return it->second.factory.get();
}
```

NewDevice函数，用于从deviceFActory生成一个eDevices对象。入参是deviceFactory的key，根据key从注册map中找到对应的DeviceFactory，然后调用DeviceFactory的CreateDevices生成Device对象。CreateDevices是一个虚函数，具体的实现过程被写在各个

```cpp
std::unique_ptr<Device> DeviceFactory::NewDevice(const string& type,
                                                 const SessionOptions& options,
                                                 const string& name_prefix) {
  auto device_factory = GetFactory(type);
  if (!device_factory) {
    return nullptr;
  }
  SessionOptions opt = options;
  (*opt.config.mutable_device_count())[type] = 1;
  std::vector<std::unique_ptr<Device>> devices;
  TF_CHECK_OK(device_factory->CreateDevices(opt, name_prefix, &devices));
  int expected_num_devices = 1;
  auto iter = options.config.device_count().find(type);
  if (iter != options.config.device_count().end()) {
    expected_num_devices = iter->second;
  }
  DCHECK_EQ(devices.size(), static_cast<size_t>(expected_num_devices));
  return std::move(devices[0]);
}
```
