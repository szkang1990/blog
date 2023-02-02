

tensorflow中的device是一个非常重要的概念，在device中定义了

device相关的对象继承关系非常杂，下图所示：
![avatar](https://github.com/szkang1990/blog/blob/main/tensorflow%E6%BA%90%E7%A0%81%E7%B2%BE%E8%AF%BB/image/deviceTree.png?raw=true)

其中localDevice的各种子类由各种factory生成。syclDevice在最新版本的tensorflow中已经换成了plugindevice等设备。不过这些设备都不常用，所以我们把注意力放在localDevice中就好。


我们先来看deviceBase
## DeviceBase
&emsp;DeviceBase是device的基类。主要的属性如下：
>  Env* const env_;\
  CpuWorkerThreads* cpu_worker_threads_ = nullptr;\
  // Set by GPUs as well as by TPU devices. \
  AcceleratorDeviceInfo* accelerator_device_info_ = nullptr; \
  thread::ThreadPool* device_thread_pool_ = nullptr; \
  std::vector<Eigen::ThreadPoolDevice*> eigen_cpu_devices_;

其中Env是对操作系统相关功能的统一封装，包括了文件系统等功能

CpuWorkerThreads 是一个结构体，定义如下：

```cpp
  struct CpuWorkerThreads {
    int num_threads = 0;
    thread::ThreadPool* workers = nullptr;
  };
```
可以看到cpu_worker_threads_其实就是对线程池的简单封装


accelerator_device_info_,在很多博客中叫 gpu_device_info_, 是一个用于描述gpu或者其他设备的结构体，结构体的定义为
```cpp
  struct AcceleratorDeviceInfo {
    // Make sure all the defaults are NULL, so we can spot missing assignments.
    stream_executor::Stream* stream = nullptr;
    DeviceContext* default_context = nullptr;
    EventMgr* event_mgr = nullptr;
    int gpu_id = -1;
  };
```
这里面有一个属性DeviceContext，定义如下：

```cpp
class DeviceContext : public core::RefCounted {
  public:
    //...
    virtual void CopyCPUTensorToDevice(const Tensor* cpu_tensor, Device* device, Tensor* device_tensor, StatusCallback done) const;
    virtual void CopyDeviceTensorToCPU(const Tensor* device_tensor, StringPiece tensor_name, Device* device, Tensor* cpu_tensor, StatusCallback done);
};
```

可以看到DeviceContext的主要作用就是提供两个接口，用于tensor在cpu和其他设备之间的相互拷贝

device_thread_pool_ 也是一个线程池

eigen_cpu_devices_ Eigen库定义的ThreadPoolDevice类。

