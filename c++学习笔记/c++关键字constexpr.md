&emsp;constexpr  用于修饰一个函数，表示在编译期间就能得到其返回值，而不是在运行期间得到。这样可以很大的提升代码的运行效率。

&emsp;宏是在预编译期间就做了置换，这一点需要和constexpr 表达式分清楚

基本使用方法如下
```cpp
constexpr int myFunc()
{
  return 1;
}
constexpr int i = myFunc() * 4;
```
constexpr  有一些要求，若违反这些要求，则在不同的c++版本中会有警告或者报错等不同的问题，所以使用时最好严格遵守这些要求

* constexpr  函数体必须有return 函数，且只能有return函数,下面的用法属于不规范用法
   
```cpp
constexpr int test(int k){
    std::cout << k <<  " in function" << std::endl;
    return k;
}
int main()
{   
    std::cout << test(13) << " in main" << std:: endl;
}
```
>运行结果： \
>13 in function \
>13 in main 

上面代码虽然不会报错，运行结果也没问题但是会报警告

> warning: use of this statement in a constexpr function is a C++14 extension [-Wc++14-extensions]

* constexpr 不允许出现变量，因为是在编译阶段就推断出结果。所以引入变量必然会导致编译阶段的值不确定，这里的变量包括：实参不允许是变量，函数体内部不允许定义变量,下面的用法属于不规范用法

```cpp
constexpr int test(int k){
    return k;
}
 
int main()
{   
    int ce = 13;  // 不允许传入变量
    std::cout << test(ce) << " in main" << std:: endl;
}
constexpr int test(int k){
    int ce;  // 不允许再constexpr中定义变量
    return k;
}
 
int main()
{   
    std::cout << test(13) << " in main" << std:: endl;
}
```
上面两个代码在c++11 会报错，应该避免
