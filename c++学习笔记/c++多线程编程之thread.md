&emsp;&emsp;C++11 新标准中引入了五个头文件来支持多线程编程，分别是atomic, thread, mutex, condition_variable和future。

* atomic：该头文主要声明了两个类, std::atomic 和 std::atomic_flag，另外还声明了一套 C 风格的原子类型和与 C 兼容的原子操作的函数。
* thread：该头文件主要声明了 std::thread 类，另外 std::this_thread 命名空间也在该头文件中。
* mutex：该头文件主要声明了与互斥量(mutex)相关的类，包括 std::mutex 系列类，std::lock_guard, std::unique_lock, 以及其他的类型和函数。
* condition_variable：该头文件主要声明了与条件变量相关的类，包括 std::condition_variable 和 std::condition_variable_any。
* future：该头文件主要声明了 std::promise, std::package_task 两个 Provider 类，以及 std::future 和 std::shared_future 两个 Future 类，另外还有一些与之相关的类型和函数，std::async() 函数就声明在此头文件中。

&emsp;&emsp;这一节介绍一下thread的用法。
thread的用法很简单，下面是一个典型的用法

```cpp
void output(int i)
{
	cout << i << endl;
}
int thread_output(){
    for (uint8_t i = 0; i < 4; i++)
	{
		thread t(output, i);
		t.detach();	
	}
		
	getchar();
	return 0;
}

int main()
{
    thread_output();
}
```

&emsp;&emsp;上面是一个典型用法，std::thread 是一个对象，每次创建一个thread实例，就会创建一个进程。一个极简的用法就是
```cpp
void test();
std::thread(test);
```
&emsp;&emsp;thread实例化的时候必须传入的参数是一个函数，可选的参数是这个函数的入参。在最上面的例子中，就是传入了output的入参i。
&emsp;&emsp;thread对象的入参是一个函数，所以除了传入显式定义的函数，也可以传入一个lambda函数或者一个函数对象。例如：

```cpp
/* lambda函数 */
for (int i = 0; i < 4; i++)
{
	thread t([i]{
		cout << i << endl;
	});
	t.detach();
}
```
```cpp
/*函数对象*/
class Task
{
public:
	void operator()(int i)
	{
		cout << i << endl;
	}
};

int main()
{
	
	for (uint8_t i = 0; i < 4; i++)
	{
		Task task;
		thread t(task, i);
		t.detach();	
	}
}
```
有一个需要注意的点是，如果传入的是一个函数对象,那么必须传入一个命名变量而不是临时变量。下面的用法是错误的：
```cpp
std::thread t(Task());
```
上面的用法会首先被认为是创建了一个名为t的返回thread的函数，而不是创建了一个新的线程。如果必须要用临时变量，可以用下面的写法(改成大括号）：
```cpp
std::thread t{Task()};
```

## detach 和 join
&emsp;&emsp;每段代码执行的时候，都会有一个主线程，即main函数。除此之外，每次调用thread都会新建一个线程。
有时候我们需要主线程等待其他线程运行完以后再执行，有时候则不需要，这就设计新建线程的两个策略join和detach。
join()函数是一个等待线程完成函数，主线程需要等待子线程运行结束了才可以结束.例如
```cpp

void output(int i)
{
	cout << i << endl;
}
int thread_output(){
    for (uint8_t i = 0; i < 4; i++)
	{
		thread t(output, i);
		t.join();	
	}
		
	// getchar();
	return 0;
}

int main()
{
    thread_output();
}
```
上面的例子的输出的结果是
> 0
> 1
> 2
> 3
之所以按照顺序输出，是因为每次新建一个新线程，join()函数都会堵塞住主线程，等新线程运行完了，然后才能继续运行，新建下一个线程。如果我们把代码写成下面这个样子，那么输出就不是按照顺序输出的了。
```cpp

void output(int i)
{
	cout << i << endl;
}
int thread_output(){
    for (uint8_t i = 0; i < 4; i++)
	{
		thread t(output, i);
		t.join();	
	}
		
	// getchar();
	return 0;
}

int main()
{
    thread t1(output, 1);
    thread t2(output, 2);
    thread t3(output, 3);
    t1.join();
    t2.join();
    t3.join();
}
```
可能的输出是：
> 132
> 
> 
> 

上面代码中，新建了3个线程，然后这三个线程的join函数都在最后，所以三个线程并行执行，因为三个线程都用到了终端资源来显示输出，所以终端资源在三个线程的抢夺下被轮流使用。

detach()函数则相反，主线程并不需要新线程执行完再执行。主线程会继续执行，如果主线程先运行结束，那么新线程的会被打断。在最一开始的例子中，之所以要加入一个getchar函数来获取输入，就是为了等新建的线程执行结束，否则会出现主线程先运行完，没有任何输出的情况。