&emsp;c++中static类型的变量的初始化和普通变量不同，由于static变量在整个程序中只创建一次，所以初始化也只会执行一次，例如如下代码：

```cpp
#include <iostream>
using namespace std;

#include <unordered_map>
std::unordered_map<string, int>* test(string a){
	static std::unordered_map<string, int>* sv = new std::unordered_map<string, int>();
	sv -> insert({a,1});
	static int b = 2;
	b +=1;
	return sv;
}

int main()
{
   cout<< test("k") -> size() <<endl;
   cout<< test("s") -> size() <<endl;
   cout << "Hello World";
   return 0;
}
```

在上面的代码中，test函数中有一个静态map变量sv, 然后在map中插入一个元素。main函数中，调用两次test函数。如果是普通变量，那么每次调用sv都会被指向一个新的内存，所以最终输出应该是两个1，然而这个代码的执行结果是：
>1\
2\
Hello World

这说明上面的代码在第二次调用test的时候，sv的空初始化并没有执行。