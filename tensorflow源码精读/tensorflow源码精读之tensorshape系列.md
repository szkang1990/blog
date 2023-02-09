本届介绍一下tensorflow中关于形状的一些对象。
首先介绍一下Shape家族

# ShapeHandle
ShapeHandle的代码路径在tensorflow\core\framework\shape_inference.h，源码如下：

```cpp
class ShapeHandle {
 public:
  ShapeHandle() {}
  bool SameHandle(ShapeHandle s) const { return ptr_ == s.ptr_; }
  std::size_t Handle() const { return reinterpret_cast<std::size_t>(ptr_); }

 private:
  ShapeHandle(const Shape* shape) { ptr_ = shape; }
  const Shape* operator->() const { return ptr_; }
  bool IsSet() const { return ptr_ != nullptr; }

  const Shape* ptr_ = nullptr;

  friend class InferenceContext;
  friend class ShapeInferenceTest;
  friend class ShapeInferenceTestutil;
  friend class ::tensorflow::grappler::SymbolicShapeManager;

  // Intentionally copyable.
};
```
ShapeHandle 的属性非常简单就一个Shape类型的指针const Shape* ptr_ = nullptr;
构造函数也很简单，输入一个shapeShapeHandle(const Shape* shape)
其他函数同样简单，例如SameHandle(ShapeHandle s) 判断是否与入参是相同
值得一提的是，这里重载了符号 -> 会返回属性ptr_

所以说ShapeHandle本质上就是Shape指针的封装

# Shape

shape的源码也在tensorflow\core\framework\shape_inference.h，源码如下：
```cpp
class Shape {
 private:
  Shape();
  Shape(const std::vector<DimensionHandle>& dims);
  ~Shape() {}

  const int32 rank_;
  const std::vector<DimensionHandle> dims_;

  friend class InferenceContext;
  friend class ::tensorflow::grappler::SymbolicShapeManager;

  TF_DISALLOW_COPY_AND_ASSIGN(Shape);
};
```
他有两个属性：
```cpp
  const int32 rank_;
  const std::vector<DimensionHandle> dims_;
```
其中rank_ 表示shape的形状的长度，例如一个tensor是
<pre> [[[1]],[[2]]] </pre>
那么tensor的形状是[2,1,1]，[2,1,1]的长度是3，所以rank_就是3

dims_是一个DimensionHandle数组，长度是rank_，表示每个维度上的值

# DimensionHandle
DimensionHandle 的代码路径也在tensorflow\core\framework\shape_inference.h，源码如下：

```cpp

class DimensionHandle {
 public:
  DimensionHandle() {}
  bool SameHandle(DimensionHandle d) const { return ptr_ == d.ptr_; }
  std::size_t Handle() const { return reinterpret_cast<std::size_t>(ptr_); }

 private:
  DimensionHandle(const Dimension* dim) { ptr_ = dim; }

  const Dimension* operator->() const { return ptr_; }
  bool IsSet() const { return ptr_ != nullptr; }

  const Dimension* ptr_ = nullptr;

  friend struct DimensionOrConstant;
  friend class InferenceContext;
  friend class ShapeInferenceTest;
  friend class ShapeInferenceTestutil;
  friend class ::tensorflow::grappler::GraphProperties;
  friend class ::tensorflow::grappler::SymbolicShapeManager;

  // Intentionally copyable.
};
```

DimensionHandle和ShapeHandle非常类似，属性只有一个指针，构造函数和成员函数简单，所以就不赘述了

# Dimension

Dimension的代码路径也在tensorflow\core\framework\shape_inference.h，源码如下：

```cpp
class Dimension {
 private:
  Dimension();
  Dimension(int64_t value);
  ~Dimension() {}

  const int64_t value_;

  friend class InferenceContext;
  friend class ShapeManager;
  TF_DISALLOW_COPY_AND_ASSIGN(Dimension);
};
```
Dimension 本质上就是一个Int64，不赘述了


```mermaid
classDiagram 
ShapeHandle *-- Shape
ShapeHandle : +Shape*
Shape *-- DimensionHandle
Shape :+ int64
Shape :+ vector&#60DimensionHandle&#62
DimensionHandle *-- Dimension
DimensionHandle : +Dimension
Dimension : +int64

```


# TensorShape

TensorShape的定义在tensorflow\core\framework\tensor_shape.h，源码如下


# PartialTensorShape

TensorShape的定义在tensorflow\core\framework\tensor_shape.h，源码如下




