​
使用 STL 时，往往会大量用到函数对象，为此要编写很多函数对象类。有的函数对象类只用来定义了一个对象，而且这个对象也只使用了一次，编写这样的函数对象类就有点浪费。

而且，定义函数对象类的地方和使用函数对象的地方可能相隔较远，看到函数对象，想要查看其 operator() 成员函数到底是做什么的也会比较麻烦。

对于只使用一次的函数对象类，能否直接在使用它的地方定义呢？Lambda 表达式能够解决这个问题。使用 Lambda 表达式可以减少程序中函数对象类的数量，使得程序更加优雅。

Lambda 表达式的定义形式如下：




函数入参




1. 参数捕捉符号不可省略，

[]：默认不捕获任何变量；
[=]：默认以复制捕获所有变量；
[&]：默认以引用捕获所有变量；
[x]：仅以复制捕获x，其它变量不捕获；
[x...]：以包展开方式复制捕获参数包变量；
[&x]：仅以引用捕获x，其它变量不捕获；
[&x...]：以包展开方式引用捕获参数包变量；
[=, &x]：默认以复制捕获所有变量，但是x是例外，通过引用捕获；
[&, x]：默认以引用捕获所有变量，但是x是例外，通过复制捕获；
[this]：通过引用捕获当前对象（其实是复制指针）；
[*this]：通过复制方式捕获当前对象；
一般来说 用的最多的是 [] , [=], [&], [this]， 其他的类型用的较少

2. 如果函数没有返回值，且不是mutable类型，则括号可以省略，这里建议任何情况下都不省略，增加debug成本

    auto fun = []()->int{return 2;};
    std::cout << fun() << std::endl;
    auto fun = []->int{return 2;};  // 有返回值，不可省略，否则报错

3. mutable，修改lambda函数为mutable，mutable模式下可以修改入参的值，函数入参的括号不可省略。如果不是mutable， 则可以省略

4. 如果没有异常捕捉则可以省略

5. 如果没有返回值则可以省略

6. 不可省略

    std::tuple<int, std::string, double> t2{42, "Test", -3.14};
    auto fun1 = [=]()->int{return std::get<0>(t2);};
    auto fun2 = [&]()->int{return std::get<0>(t2);};
    auto fun2_ = []()->int{return std::get<0>(t2);}; // 报错
    std::cout << fun2() << std::endl;
    // 生成一个递增序列
    auto fun3 = [](int length) -> std::vector<int>{
        std::vector<int> res(length, 0);
        for(int i=0; i<length;i++){
            res[i] = i;
        }
        return res;
    };
    std::vector<int> lambdak = fun3(10);
    for (auto k: lambdak){
        std::cout << k << " ";
    }


​