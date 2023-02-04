## 内联函数
&emsp;&emsp;在大型项目中，函数的调用是非常频繁的。但是函数的调用是有时间和空间成本的，程序在执行一个函数之前需要做一些准备工作，要将实参、局部变量、返回地址以及若干寄存器都压入栈中，然后才能执行函数体中的代码；函数体中的代码执行完毕后还要清理现场，将之前压入栈中的数据都出栈，才能接着执行函数调用位置以后的代码。

&emsp;&emsp;对于一些运算复杂的函数，这种调用的准备工作的开支可以忽略，但是如果函数体本身比较简单而且调用非常频繁，那么这种调用的时间开支就非常大了。为了消除函数调用的时空开销，C++ 提供一种提高效率的方法，即在编译时将函数调用处用函数体替换，类似于C语言中的宏展开。

&emsp;&emsp;这种在函数调用的地方直接嵌入函数体的做法，叫做内联函数。指定内联函数的做法非常简单，只需要在函数定义处增加 inline 关键字，例如

```cpp
#include <iostream>
using namespace std;
//内联函数，交换两个数的值
inline void swap(int *a, int *b){
    int temp;
    temp = *a;
    *a = *b;
    *b = temp;
}
int main(){
    int m, n;
    cin>>m>>n;
    cout<<m<<", "<<n<<endl;
    swap(&m, &n);
    cout<<m<<", "<<n<<endl;
    return 0;
}
```

内联函数也是有一些缺点的，因为是把函数体直接嵌入，所以如果内联函数函数体比较大，那么编译以后的程序体积会很大。所以内联函数一般只用于函数体短小，调用频繁的函数。

最后需要说明的是，对函数作 inline 声明只是程序员对编译器提出的一个建议，而不是强制性的，并非一经指定为 inline 编译器就必须这样做。编译器有自己的判断能力，它会根据具体情况决定是否这样做。

如果不想让编译器决定是否使用内联函数 ,而是强制使用内联函数，可以用__forceinline来实现

## 显式构造函数

显式构造函数修饰词是explicit，这个修饰词只能用于构造函数。首先介绍一下显式和隐式的构造函数。
一个对象的的构造函数只有一个入参，而且所有参数都有默认值，那么可以隐式的创建一个类，例如：

```cpp
#include <iostream>
using namespace std;
class B{
public:
    int data;
    B(int _data):data(_data){}
    //explicit B(int _data):data(_data){}
};
 
int main(){
    B temp = 5;
    cout << temp.data << endl;
    return 0;
}
```

在main函数中，

```cpp
B temp = 5;
```
是隐式的生成了一个B类，对象的构造函数默认都是可以进行隐式转换的。在一些场景中，我们希望对象不能隐式的转换，以避免一些混淆。这时候用explicit就可以要求这个对象必须显式的. 具体用法是：

```cpp
#include <iostream>
using namespace std;
class B{
public:
    int data;
    explicit B(int _data):data(_data){}
    //explicit B(int _data):data(_data){}
};
 
int main(){
    B temp = 5;  // 报错
    cout << temp.data << endl;
    return 0;
}
```
