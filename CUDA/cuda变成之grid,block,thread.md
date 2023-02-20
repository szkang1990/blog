#cuda编程之thread, block, grid

在cuda编程的时候我们经常会看到函数后面跟着<pr><<<...>>></pr>。这是一个设置grid ，block的语法。
## thread
thread 是cuda运行的最小单元，每个thread都会把我们提交的kernel执行一遍，不同的thread运行不同的数据，就实现了并行的运算。当然，这些thread的数据也可以是一样的。

## block
block 多个thread合并在一起，可以组成一个block。这种组合形式可以是一维的：0号线程，1号线程...。也可以是二维的：(0,0)线程，(0,1)线程...。 也可以是三维的：(0,0,0)线程，(0,0,1)线程...。

block中的线程数量和各个维度的的数量都有上线，一个block最多包含1024个线程，同时x,y,z三个维度上最大的线程数是（1024，1024，64），在x*y*z < 1024 的前提下，还要x < 1024, y < 1024, z < 64

在gpu内部，线程以warp为单位进行并发执行，一个warp包含32个thread，这个数值无法修改，所以一个block包含的thead的数量最好是32的倍数。


每个 thread都有自己的local memory。同一个block的thread有一个共享内存，而不同block的线程是无法相互影响的。

## grid

 多个block组成一个grid，一个kernel只能有一个grid。与thread一样，block在grid中的排列形式也可以是一维二维和三维的。


## dim3

前面已经说了thread组成block和block组成grid的形式可以是一维二维三维的。那么怎么在kernel确定足thread和block的组织形式呢？这是通过dim来实现的。
dim3是一个结构体，有三个属性，xyz。分别代表三个维度的大小。dim3的初始化有以下形式
```c
%%cuda --name dim3.cu
#include <cuda_runtime.h>
#include <stdio.h>
int main(int argc, char **argv)
{
    dim3 block1(3);
    dim3 block2(3,2);
    dim3 block3(3,2,4);
    printf("block1.x %d block1.y %d block1.z %d\n",block1.x,block1.y,block1.z);
    printf("block2.x %d block2.y %d block2.z %d\n",block2.x,block2.y,block2.z);
    printf("block3.x %d block3.y %d block3.z %d\n",block3.x,block3.y,block3.z);
    return 0;
}
```
输出
> block1.x 3 block1.y 1 block1.z 1
block2.x 3 block2.y 2 block2.z 1
block3.x 3 block3.y 2 block3.z 4

可以看到，事实上thread和block的组织形式默认都是三维形式。如果我们定义其中某个维度，其他维度会被设置成1

一般情况下，会按照xyz的顺序把维度设置成不是1，也就是说：2,1,1 和  2,2,1 和2,2,2 都是比较常见的。而1,2,2、2,1,2、1,1,2这种方式是非常不建议使用的，会导致后面计算thread 索引很容易出错。
## <<<...>>>
现在我们再来看每个kernel后面跟随这个符号，这个符号中有两个必须传入的入参，格式是dim3，分别表示block的组织形式和thread的组织形式, 
下面是一个例子
```c
#include <cuda_runtime.h>
#include <stdio.h>
__global__ void check_dim(void)
{
  printf("threadIdx:(%d,%d,%d) blockIdx:(%d,%d,%d) blockDim:(%d,%d,%d)\
  gridDim(%d,%d,%d)\n",threadIdx.x,threadIdx.y,threadIdx.z,
  blockIdx.x,blockIdx.y,blockIdx.z,blockDim.x,blockDim.y,blockDim.z,
  gridDim.x,gridDim.y,gridDim.z);
}
int main(int argc, char **argv)
{
    int nElem = 6;
    dim3 block(3,2);
    dim3 grid(2,3);
    printf("grid.x %d grid.y %d grid.z %d\n",grid.x,grid.y,grid.z);
    printf("block.x %d block.y %d block.z %d\n",block.x,block.y,block.z);
    check_dim<<<grid,block>>>();
    cudaDeviceReset();
    return 0;
}
```
输出如下：
> grid.x 2 grid.y 3 grid.z 1
block.x 3 block.y 2 block.z 1
threadIdx:(0,0,0) blockIdx:(0,1,0) blockDim:(3,2,1)  gridDim(2,3,1)
threadIdx:(1,0,0) blockIdx:(0,1,0) blockDim:(3,2,1)  gridDim(2,3,1)
threadIdx:(2,0,0) blockIdx:(0,1,0) blockDim:(3,2,1)  gridDim(2,3,1)
threadIdx:(0,1,0) blockIdx:(0,1,0) blockDim:(3,2,1)  gridDim(2,3,1)
threadIdx:(1,1,0) blockIdx:(0,1,0) blockDim:(3,2,1)  gridDim(2,3,1)
threadIdx:(2,1,0) blockIdx:(0,1,0) blockDim:(3,2,1)  gridDim(2,3,1)
threadIdx:(0,0,0) blockIdx:(1,2,0) blockDim:(3,2,1)  gridDim(2,3,1)
threadIdx:(1,0,0) blockIdx:(1,2,0) blockDim:(3,2,1)  gridDim(2,3,1)
threadIdx:(2,0,0) blockIdx:(1,2,0) blockDim:(3,2,1)  gridDim(2,3,1)
threadIdx:(0,1,0) blockIdx:(1,2,0) blockDim:(3,2,1)  gridDim(2,3,1)
threadIdx:(1,1,0) blockIdx:(1,2,0) blockDim:(3,2,1)  gridDim(2,3,1)
threadIdx:(2,1,0) blockIdx:(1,2,0) blockDim:(3,2,1)  gridDim(2,3,1)
threadIdx:(0,0,0) blockIdx:(1,1,0) blockDim:(3,2,1)  gridDim(2,3,1)
threadIdx:(1,0,0) blockIdx:(1,1,0) blockDim:(3,2,1)  gridDim(2,3,1)
threadIdx:(2,0,0) blockIdx:(1,1,0) blockDim:(3,2,1)  gridDim(2,3,1)
threadIdx:(0,1,0) blockIdx:(1,1,0) blockDim:(3,2,1)  gridDim(2,3,1)
threadIdx:(1,1,0) blockIdx:(1,1,0) blockDim:(3,2,1)  gridDim(2,3,1)
threadIdx:(2,1,0) blockIdx:(1,1,0) blockDim:(3,2,1)  gridDim(2,3,1)
threadIdx:(0,0,0) blockIdx:(0,2,0) blockDim:(3,2,1)  gridDim(2,3,1)
threadIdx:(1,0,0) blockIdx:(0,2,0) blockDim:(3,2,1)  gridDim(2,3,1)
threadIdx:(2,0,0) blockIdx:(0,2,0) blockDim:(3,2,1)  gridDim(2,3,1)
threadIdx:(0,1,0) blockIdx:(0,2,0) blockDim:(3,2,1)  gridDim(2,3,1)
threadIdx:(1,1,0) blockIdx:(0,2,0) blockDim:(3,2,1)  gridDim(2,3,1)
threadIdx:(2,1,0) blockIdx:(0,2,0) blockDim:(3,2,1)  gridDim(2,3,1)
threadIdx:(0,0,0) blockIdx:(0,0,0) blockDim:(3,2,1)  gridDim(2,3,1)
threadIdx:(1,0,0) blockIdx:(0,0,0) blockDim:(3,2,1)  gridDim(2,3,1)
threadIdx:(2,0,0) blockIdx:(0,0,0) blockDim:(3,2,1)  gridDim(2,3,1)
threadIdx:(0,1,0) blockIdx:(0,0,0) blockDim:(3,2,1)  gridDim(2,3,1)
threadIdx:(1,1,0) blockIdx:(0,0,0) blockDim:(3,2,1)  gridDim(2,3,1)
threadIdx:(2,1,0) blockIdx:(0,0,0) blockDim:(3,2,1)  gridDim(2,3,1)
threadIdx:(0,0,0) blockIdx:(1,0,0) blockDim:(3,2,1)  gridDim(2,3,1)
threadIdx:(1,0,0) blockIdx:(1,0,0) blockDim:(3,2,1)  gridDim(2,3,1)
threadIdx:(2,0,0) blockIdx:(1,0,0) blockDim:(3,2,1)  gridDim(2,3,1)
threadIdx:(0,1,0) blockIdx:(1,0,0) blockDim:(3,2,1)  gridDim(2,3,1)
threadIdx:(1,1,0) blockIdx:(1,0,0) blockDim:(3,2,1)  gridDim(2,3,1)
threadIdx:(2,1,0) blockIdx:(1,0,0) blockDim:(3,2,1)  gridDim(2,3,1)

上面的例子中，<pr><<<grid,block>>></pr>定义了一个grid中有3 * 2 个block， 每个block中有2*3个thread。所以一共有 2\*3\*3\*2个线程，最后的输出也是这么多行。

在很多例子中，我们会看到这种用法：
```c
#include <cuda_runtime.h>
#include <stdio.h>
__global__ void check_dim(void)
{
  printf("threadIdx:(%d,%d,%d) blockIdx:(%d,%d,%d) blockDim:(%d,%d,%d)\
  gridDim(%d,%d,%d)\n",threadIdx.x,threadIdx.y,threadIdx.z,
  blockIdx.x,blockIdx.y,blockIdx.z,blockDim.x,blockDim.y,blockDim.z,
  gridDim.x,gridDim.y,gridDim.z);
}
int main(int argc, char **argv)
{

    check_dim<<<2,4>>>();
    cudaDeviceReset();
    return 0;
}
```
这种用法用到了dim3的隐式声明，本质上是这样的

```c
#include <cuda_runtime.h>
#include <stdio.h>
__global__ void check_dim(void)
{
  printf("threadIdx:(%d,%d,%d) blockIdx:(%d,%d,%d) blockDim:(%d,%d,%d)\
  gridDim(%d,%d,%d)\n",threadIdx.x,threadIdx.y,threadIdx.z,
  blockIdx.x,blockIdx.y,blockIdx.z,blockDim.x,blockDim.y,blockDim.z,
  gridDim.x,gridDim.y,gridDim.z);
}
int main(int argc, char **argv)
{
    check_dim<<<dim3{2},dim3{4}>>>(); // 本质上还是声明了一个dim3
    cudaDeviceReset();
    return 0;
}
```

## 相关的内置变量

上面用到了一些内置的变量blockIdx，threadIdx，blockDim，gridDim
这些变量是为了唯一确定一个thread。
前面已经介绍了，若干个thread组成block， 若干个block组成grid。这样我们在实际应用中，就可以把数据分成和thread一样的份数，每个thread计算一份，实验并行运算。
这样，就会有一个需求:我们要把thread和数据一一对应起来。一般数据就是简单的一维或者二维数组，我们可以直接通过下标就能取到。所以最好的方式就是给每个thread同样编上号，与数组的下标一一对应。
通过blockIdx，threadIdx，blockDim，gridDim这四个就可以唯一的确定一个thread。
blockIdx：表示当前线程所在的block在grid中的下标，是一个dim3类型的结构体
threadIdx：表示当前线程所在的block的下标，是一个dim3类型的结构体
blockDim： block的形状
gridDim：  grid的形状
下面就涉及thread的索引计算问题了。
thread的索引计算思想非常简单，block和thread都是以三维形式堆叠的结果，所以我们先找到blockid在grid中的位置，然后再找到thread在block中的位置，最后用如下公式：
$$pos_{thread} = id_{block} \times size_{block} + id_{thread} $$
因而，索引计算就被分成两部分：blockid的计算和threadid的计算。
本质上两者的计算是一样的。都是在一个二维或者三维的数组中计算线性序号。这与多维数组的存储非常类似
下面用一个例子说明, 如果一个thread 的 block 在grid中的位置是blockIdx， thread在block中的位置是threadIdx

那么这个block的位置是 

int blockId = blockIdx.x + blockIdx.y * gridDim.x
 + gridDim.x * gridDim.y * blockIdx.z;


int threadId =  threadIdx.z * (blockDim.x * blockDim.y)
 + (threadIdx.y * blockDim.x) + threadIdx.x;

最终的 thread索引是  blockId * (blockDim.x * blockDim.y * blockDim.z)
 + (threadIdx.z * (blockDim.x * blockDim.y))
 + (threadIdx.y * blockDim.x) + threadIdx.x;






## 并行运算
在深度学习中，矩阵






