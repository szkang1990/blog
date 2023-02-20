在thread那一节里面我们提到过，当多个线程抢夺一个终端资源的时候，会导致终端的输出出现问题，例如下面的代码

```cpp
void output(int i)
{
	cout << i << endl;
}
int thread_output2(){
    thread t1(output, 1);
    thread t2(output, 2);
    thread t3(output, 3);
    t1.join();
    t2.join();
    t3.join();
	return 0;
}

int main()
{
    thread_output2();
}
```
例如我们希望的输出是
1
2
3
或者
2
3
1
而因为多线程的抢夺，变成了
132


\
(连续两个换行)
同样的道理，如果多个线程在编辑同一个数据的时候，那么就有可能会出现错误。所以我们希望对同一份资源，完全使用完了以后再由别的线程使用。这就需要用到mutex了。mutex的英文意思是互斥，直观的意义就很明确了。我们只需要把代码稍作修改


```cpp
void output(int i)
{
    mt.lock();
	cout << i << endl;
    mt.unlock();
}
int thread_output2(){
    thread t1(output, 1);
    thread t2(output, 2);
    thread t3(output, 3);
    t1.join();
    t2.join();
    t3.join();
	return 0;
}

int main()
{
    thread_output2();
}
```
上面的代码输出结果是
1
2
3
或者
1
3
2
或者
2
1
3
总之是把数字和换行符打印完了以后再释放终端资源。