对象初始化的时候，有两种方法。例如，对于一个vector，我们有下面两种写法：
```cpp
vector<int> v1 = vector<int>();
vector<int> *v2 = new vector<int>();
```
这里就能看出来 有new 和没有new初始化的区别，没有new的时候会返回一个对象引用，而有new的时候则会返回一个对象指针。

然后这里面还有其他的区别

```cpp
vector<int> v1 = vector<int>();
vector<int> *v2 = new vector<int>();
v1.emplace_back(20);
std::cout<< (v1.size()) << std::endl;
v2->emplace_back(20);
std::cout<< (v2 -> size()) << std::endl;
```

上面的代码运行起来没有问题，都会打印出vector的长度

然而下面的代码确会报错：

```cpp
vector<int> v1 ;
vector<int> *v2 ;
v1.emplace_back(20);
std::cout<< (v1.size()) << std::endl;
v2->emplace_back(20);
std::cout<< (v2 -> size()) << std::endl;
```
上面的代码会报错：Segmentation fault: 11

原因是vector<int> *v2 中我们只是指定了指针v2的类型，但是确并没有给他指定内存空间，所以这是一个野指针，所以在对野指针进行操作的时候会报错。

而引用则没有这个问题，我们定义一个引用的时候，这个引用就必须指向一个内存空间。


假设扩充负样本之前的正样本概率为 $p_0$， 假设正样本数量是$y_0$, 负样本数量是$y_1$,则
$$p_0 = y_1/(y_0+y_1)
扩充zhen
