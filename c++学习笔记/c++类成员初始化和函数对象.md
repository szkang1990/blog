类成员可以在构造函数中进行初始化，具体写法是
```cpp
public className  :  property(value) {
    函数主体
}
```
这种写法对于基本数据类型，例如int，long相当于直接赋值。对于对象相当于调用对象的构造函数。
例如：
```cpp
class FeatureValue
{
public:
    int sss;
    FeatureValue(int value):sss(value){
    };
};
class Feature
{
public:
    FeatureValue kkk;
    Feature(int value):kkk(value) {}
};
 
int main()
{
    FeatureValue fv = FeatureValue(2);
    std::cout << fv.sss << std::endl;
    Feature f = Feature(1);
    std::cout << f.kkk.sss << std::endl;
}
```
上面的例子中，对于FeatureValue的类成员初始化比较好理解，把2 赋给sss，因为sss本就是int。

然后后面Feature 是比较反常的，kkk是一个类，但是却可以直接用这种方式进行初始化。

函数对象

如果对象中有对括号的符号重载，那么这个对象就是函数对象，其**实例化的类**可以当做函数使用。

例如：
```cpp
#include <iostream>
class Temp{

public:
    int a;
    Temp(int value): a(value){}
    //Temp(): a(7){}
    void operator()(){
        std::cout << "kkk" << std::endl;
        this->a = 8;
    }
 
 
};
int main() {
    std::cout << "hello world" << std::endl;
    Temp t2 = Temp(6);
    //Temp t2 = Temp();
    std::cout << t2.a << std::endl;
    t2();
    std::cout << t2.a << std::endl;
    return 0;
}
```
>输出\
6\
kkk\
8

在上面的例子中，对象Temp是一个函数对象，在main函数中，temp初始化时，类成员a初始化为6，实例t2以后，调用了t2一次，相当于调用了一次operator，所以输出了kkk，同时把类成员a改成了6。

这里需要强调的是，函数对象是对象实例能够当做函数使用，而不是对象本身。