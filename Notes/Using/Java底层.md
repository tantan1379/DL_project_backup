# 多线程

### 线程简介

进程（Process）：进程是程序执行的一次过程，是一个**动态**的概念。是系统资源分配的基本单位，

线程（Thread）：一个进程中可以包含若干个进程。线程是CPU运算调度的基本单位。

多线程：很多多线程是模拟出来的，真正的多线程是指有多个cpu（多核，如服务器）。模拟出来的多线程是在一个cpu的情况下，在同一时间点，cpu只能执行一个代码，只是切换的很快有同时执行的错觉。



**多线程和普通方法的区别：**

普通方法：只有主线程一条执行路径；

多线程：多条执行路径，主线程和子线程**并行交替**执行；



**核心概念：**

- 线程是独立的执行路径；
- 在程序运行时，即使没有自己创造线程，后台也会有多个线程，如主线程，gc线程；
- main()称为主线程，为系统的入口，用于执行整个程序；
- 在进程中，如果开辟了很多个线程，线程的运行由调度器安排调度，调度器是与操作系统紧密相关的，不能人为干预；
- 对同一份资源操作时，会存在资源抢夺问题，需要加入并发控制；
- 线程会带来额外的开销，如cpu调度时间，并发控制开销；
- **每个线程只能在自己的工作内存交互，内存控制不当会导致数据不一致**



##### **多线程三要素**

**原子性：**

指的是一个或者多个操作，要么全部执行并且在执行的过程中不被其他操作打断，要么就全部不执行。原子性是数据一致性的保障。

解决：synchronized关键字

**可见性：**

指多个线程操作一个共享变量时，其中一个线程对变量进行修改后，其他线程可以立即看到修改的结果。(线程间的通信实现)

解决：synchronized关键字、volatile关键字

**有序性：**

在执行程序时，为了提供性能，处理器和编译器常常会对指令进行重排序，但是不能随意重排序。

需要满足以下两个条件：（1）在单线程环境下不能改变程序运行的结果；（2）存在数据依赖关系的不允许重排序



### 线程创建

##### **lambda表达式**

Lambda 表达式，也可称为闭包，它是推动 Java 8 发布的最重要新特性。Lambda 允许把函数作为一个方法的参数。

**具体用法：**对于函数式接口，可以通过lambda表达式创建该接口的对象。（**函数式接口：**任何接口只包含**一个**抽象方法）

e.g.:`new Thread(()->System.out.println("学习")).start();`

**语法：**

（1）(params)->expression

（2）(params)->{statements;}



**说明：**

- lambda表达式用接口类型的对象进行接收，表示对接口中的唯一方法进行实现（如果需要参数需要添加参数）
- 可选类型声明：不需要声明参数类型（要去除就都去除），编译器可以统一识别参数值。
- 可选的参数圆括号：一个参数无需定义圆括号，但多个参数需要定义圆括号。
- 可选的大括号：如果主体只包含了一个语句，就不需要使用大括号。
- 可选的返回关键字：如果主体只有一个表达式返回值则编译器会自动返回值，大括号需要指定表达式返回了一个数值。



**优势：**

- 避免匿名内部类定义过多
- 简化代码，只留下核心逻辑



##### **三种方式**

（1）继承Thread类【声明为Thread的子类，并重写run类】（重点）

（2）实现Runnable接口【重写run类的T】（重点）

（3）实现Callable接口（了解）



**方式一、继承Thread类**

**过程：**

- 自定义线程继承Thread类；
- 重写run方法，编写线程执行体；
- 创建线程对象，调用start()方法启动线程。（注意：调用start方法会自动调用重写的run方法）



demo:

```
public class TestThread1 extends Thread{
    @Override
    public void run() {
        //run方法线程体
        for (int i = 0; i < 20; i++) {
            System.out.println("Thread线程"+i);
        }
    }

    public static void main(String[] args) {
        //主线程
        //线程开启不一定立即执行，由CPU调度执行；
        //多线程和主线程并行交替执行
        TestThread1 testThread1 = new TestThread1();//创建线程对象
        testThread1.start();//调用start方法开启线程

        for (int i = 0; i < 1000; i++) {
            System.out.println("主线程"+i);
        }

    }
}
```



**方式二、实现Runnable接口**

**过程：**

- 自定义线程实现Runnable接口；
- 重写run方法，编写线程执行体；
- 创建线程对象，创建Thread对象并作为参数传递，调用start()方法启动线程，可以传入同一个线程。（注意：调用start方法会自动调用重写的run方法）



demo:

```
public class TestThread4 implements Runnable{
    private int ticketNum = 10;

    public void run() {
        while(true){
            if(ticketNum<=0){
                break;
            }
            System.out.println(Thread.currentThread().getName()+"拿到了第"+ticketNum--+"张票");
        }
    }

    public static void main(String[] args) {
        TestThread4 tt = new TestThread4();
        new Thread(tt,"a").start();//Thread第二个参数可以为线程取名
        new Thread(tt,"b").start();
    }
}
```

**说明：**

- Thread第二个参数可以为线程取名
- `Thread.currentThread()`取得当前的线程，`Thread.currentThread().getName()`取得当前的线程名



**方式三、实现Callable接口**

**过程：**

- 实现Callable接口，需要返回值类型
- 需要重写call方法并抛出异常
- 创建目标对象：`TestCallable t1 = new TestCallable();`
- 创建执行服务：`ExecutorService ser = Executors.newFixedThreadPool(1);`
- 提交执行：`Future<Boolean> result1 = ser.submit(t1);`
- 获取结果：`boolean r1 = result1.get();`
- 关闭服务：`ser.shutdown();`



demo:

```
public class TestCallable implements Callable<Boolean> {

    //加载图片线程的执行体
    @Override
    public Boolean call() {
        System.out.println("下载完成");
        return true;
    }

    public static void main(String[] args) throws ExecutionException, InterruptedException {
        TestCallable t1 = new TestCallable();
        TestCallable t2 = new TestCallable();
        TestCallable t3 = new TestCallable();
        
        //创建执行服务
        ExecutorService ser = Executors.newFixedThreadPool(3);//创建线程池
        //提交执行
        Future<Boolean> result1 = ser.submit(t1);
        Future<Boolean> result2 = ser.submit(t2);
        Future<Boolean> result3 = ser.submit(t3);
        //获取结果(返回值）
        boolean r1 = result1.get();
        boolean r2 = result2.get();
        boolean r3 = result3.get();

        //关闭服务
        ser.shutdown();
    }
}
```



**三种方式比较**

**（1）继承Thread类**

- 子类继承Thread类就可以具备多线程能力
- 启动线程：`子类对象.start()`
- 不建议使用，因为OOP是单继承，同一个对象不能被多个线程使用



**（2）实现Runnable接口**

- 子类实现Runnable接口并重写run方法可以具备多线程能力
- 启动线程：`Thread对象(实现类对象).start()`
- 建议使用，灵活方便，同一个对象可以被多个线程使用（多代理）



**（3）实现Callable接口**

- 子类实现Runnable接口并重写call方法可以具备多线程能力
- 启动线程：创建执行服务+提交执行+获取结果
- Runnable 没有返回值，Callable 有返回值，Callable 可以看作是 Runnable 的补充



### 守护线程

线程分为用户线程和守护线程。

虚拟机必须确保用户线程执行完毕，不用等待守护线程执行完毕。

守护线程包括比如后台记录操作日志、监控内存、垃圾回收机制。

语法：`thread.setDaemon(true);`	设置某个线程为守护线程（默认为false，表示用户线程）



### 代理

##### **静态代理**

真实对象和代理对象实现同一个接口，代理对象要代理真实角色（设置真实角色的属性并传入）。

**优势：**之所以实现相同接口，是为了尽可能保证代理对象的内部结构和目标对象一致，这样我们对代理对象的操作最终都可以转移到目标对象身上，代理对象只需专注于增强代码的编写。

demo:

```
public class TestSP {
    public static void main(String[] args) {
        WeddingCompany weddingCompany = new WeddingCompany(new You());
        weddingCompany.HappyMarry();
    }
}

interface Marry{
    void HappyMarry();
}

//真实角色结婚
class You implements Marry{
    @Override
    public void HappyMarry() {
        System.out.println("我要结婚了，很开心！");
    }
}

//代理角色，帮助结婚
class WeddingCompany implements Marry{
    private Marry target;


    public WeddingCompany(Marry target) {
        this.target = target;
    }

    @Override
    public void HappyMarry() {
        before();
        this.target.HappyMarry();
        after();
    }

    private void after() {
        System.out.println("结婚之后收尾款");
    }

    public void before(){
        System.out.println("结婚前布置现场");
    }
}
```



##### **动态代理**

Jdk提供了**invocationHandler接口**和**Proxy类**。

动态代理和静态代理的区别在于静态代理我们需要手动的去实现目标对象的代理类，而动态代理可以在运行期间动态的生成代理类。

**实现流程：**

- 实现 InvocationHandler 接口，重写 invoke 方法，在invoke方法中完成代理的功能
- 创建 JDK 动态代理类，创建 JDK 动态代理类实例同样也是使用反射包中的 java.lang.reflect.Proxy 类进行创建。通过调用`Proxy.newProxyInstance`静态方法进行创建。

- 实现动态代理：通过向下转型，调用实现类的getProxy方法。



### 线程状态

java中用`Thread.state`表示线程当前的状态。

线程状态包括：创建状态(New)、就绪状态(Ready)、运行状态(Runnable)、阻塞状态(Blocked)、死亡状态(Terminated)



**过程：**

- 当线程对象被创建时，线程进入**创建状态**(New)
- 当调用start方法，线程立即进入**就绪状态**，等待调度执行(Ready)
- 当调用sleep、wait或同步锁时，线程进入**阻塞状态**(Blocked)；阻塞解除后，重新进入就绪状态等待cpu调度执行；
- cpu对线程进行调度执行，线程进入**运行状态**(Runnable)；
- 线程中断或结束，线程进入**死亡状态**(Terminated)，处于终止态的进程不再被调度执行



demo:

```
public class TestState {
    public static void main(String[] args) throws InterruptedException {
        Thread thread = new Thread(()->{//创建进程
            for (int i = 0; i < 5; i++) {
                try {
                    Thread.sleep(1000);//阻塞
                } catch (InterruptedException e) {
                    e.printStackTrace();
                }
            }
            System.out.println("//////");
        });
        Thread.State state = thread.getState();//(New)
        System.out.println(state);
        thread.start();//启动线程,进入预备状态
        state = thread.getState();//(Runnable)
        System.out.println(state);
        while(state != Thread.State.TERMINATED){//当阻塞状态结束，终止进程(TERMINATED)
            Thread.sleep(100);
            state = thread.getState();//(TIMED_WAITING)
            System.out.println(state);
        }
    }
}
```





### 线程方法

**停止线程**

废弃：stop()、destroy()

一般使用一个标志位作为终止变量：

```
public class TestStop implements Runnable{
    private boolean flag = true;

    @Override
    public void run() {
        int i = 0;
        while(flag){
            System.out.println("run Thread"+i++);
        }
    }

    public void stop(){
        flag = false;
    }

    public static void main(String[] args) throws InterruptedException {
        TestStop testStop = new TestStop();
        new Thread(testStop).start();
        Thread.sleep(10);//防止分线程未执行主线程就将flag置false
        for (int i = 0; i < 1000; i++) {
            System.out.println("main"+i);
            if(i==900){
                testStop.stop();
                System.out.println("该线程停止");
            }
        }
    }
}
```



**线程休眠**

**语法：**`Thread.sleep(time)`

**说明：**

- sleep(time)指定当前线程组成的毫秒数；
- sleep使用时需要抛出相关异常（InterruptedException)
- sleep时间到达后线程进入就绪状态；
- sleep可以模拟网络延时，倒计时等=>放大问题的发生性；
- **每一个对象都有一把锁，sleep不会释放锁**



**线程礼让**

**语法：**`Thread.yield()`

**作用：**

（1）礼让线程，让当前执行的线程暂停，但不阻塞；

（2）此时线程会从运行状态转为就绪状态；

（3）**让cpu重新调度，礼让不一定成功，看CPU分配情况。**



demo:

```
public class TestYield {
    public static void main(String[] args) {
        MyYield myYield = new MyYield();
        new Thread(myYield,"a").start();
        new Thread(myYield,"b").start();
    }
}

class MyYield implements Runnable{

    @Override
    public void run() {
        System.out.println(Thread.currentThread().getName()+"线程开始执行");
        Thread.yield();
        System.out.println(Thread.currentThread().getName()+"线程结束执行");
    }
}
```



**线程合并**

**语法：**`thread.join()`	注意，此处的thread是进程对象

**作用：**用于合并线程，当此线程执行完成后，再执行其他线程，其他线程阻塞(插队)

demo:

```
public class TestJoin implements Runnable{
    @Override
    public void run() {
        for (int i = 0; i < 1000; i++) {
            System.out.println("vip-"+i);
        }
    }

    public static void main(String[] args) throws InterruptedException {
        TestJoin testJoin = new TestJoin();
        Thread thread = new Thread(testJoin);
        thread.start();

        //主线程
        for (int i = 0; i < 500; i++) {
            if(i==200){
                thread.join();
            }
            System.out.println("main-"+i);
        }
    }
}
```



**设置优先级**

**语法：**

`Thread.currentThread().getPriority()`	显示当前进程的优先级

`thread.setPriority(PRIORITY)`	设置优先级







### 线程安全

方式：线程同步

场景：并发（同一个对象被多个线程同时操作）

线程同步是一种等待机制，多个需要同时访问此对象的线程进入**这个对象的等待池**形成队列，等先前的线程使用完毕下一个线程再使用。

**总结：**

- 方法一：使用安全类。
- 方法二：使用自动锁 synchronized。
- 方法三：使用手动锁 Lock。



**队列和锁**

每个对象都拥有一把锁。

由于同一进程的多个线程共享同一块存储空间，在带来方便的同时也会带来访问冲突问题。为了保证数据在方法中被访问时的正确性，在访问时需要加入锁机制 (synchronized)，当一个线程获得对象的排它锁，独占资源，其他线程必须等待，使用后释放锁即可。

**主要问题：**

- 一个线程持有锁会导致其他所有需要此锁的线程挂起，降低效率；
- 在多线程竞争下，加锁、释放锁会导致较多的上下文切换和调度延时，引起性能问题；
- 如果一个优先级高的线程等待一个优先级低的线程释放锁，会导致优先级倒置，引起性能问题；



##### synchronized同步

**同步方法**

**语法：**`public synchronized void method(args){}`

**说明：**synchronized方法控制对**”对象“**的访问，每个对象对应一把锁，每个synchronized方法都必须获得调用该方法的对象的锁才能执行，否则线程阻塞，方法一旦执行就独占该锁，直到该方法返回才释放锁，后面被阻塞的线程才能获得该锁继续执行。



**同步块**

**语法：**`synchronized(obj){}`

obj称为同步监视器。

**说明：** 

- Obj可以是任何对象，但是推荐使用**共享资源**作为同步监视器
- 同步方法中无需指定同步监视器，因为同步方法的同步监视器就是this，或者是class.

**同步监视器的执行过程：**

- 第一个线程访问，锁定同步监视器，执行代码；
- 第二个线程访问，发现同步监视器被锁定，无法访问；
- 第一个线程访问完毕，解锁同步监视器；
- 第二个线程访问，锁定，重复；



##### **Lock锁**

lock锁通过显示定义同步锁对象实现同步。

所属包：`java.util.concurrent.locks.Lock`

Lock接口是控制多个线程对共享资源访问的工具。锁提供了对共享资源的独占空间，每次只能有一个对象对Lock对象加锁，线程开始访问共享资源前应该先获得Lock对象。

**ReentrantLock类**实现了Lock接口，拥有和synchronized相同的并发性和内存语义。在实现线程安全控制中，比较常用的是ReentranLock，可以显式的加锁、释放锁。



synchronized和Lock对比：

- Lock是显式锁（需要手动开关），synchronized是隐式锁，退出作用域会自动释放；
- Lock只有代码块锁，synchronized有代码块锁和方法锁两种
- 使用Lock锁，JVM花费较少的时间调度线程，性能更好，有更好的扩展性



##### **死锁**

多个线程各自占有一些共享资源，并且互相等待其他线程占有的资源才能运行，而导致两个或以上的线程都在等待对方释放资源，都停止执行的情形。

同一个同步块同时拥有**两个以上对象的锁**，就可能发生死锁问题。



产生的四个必要条件：

- 互斥条件：进程对所分配到的资源不允许其他进程进行访问，若其他进程访问该资源，只能等待，直至占有该资源的进程使用完成后释放该资源
- 请求和保持条件：进程获得一定的资源之后，又对其他资源发出请求，但是该资源可能被其他进程占有，此时请求阻塞，但又对自己获得的资源保持不放
- 不可剥夺条件：是指进程已获得的资源，在未完成使用之前，不可被剥夺，只能在使用完后自己释放
- 环路等待条件：是指进程发生死锁后，若干进程之间形成一种头尾相接的循环等待资源关系



**怎么预防死锁**

预防死锁的方式就是打破四个必要条件中的任意一个即可。

1）打破互斥条件：在系统里取消互斥。若资源不被一个进程独占使用，那么死锁是肯定不会发生的。但一般来说在所列的四个条件中，“互斥”条件是无法破坏的。因此，在死锁预防里主要是破坏其他几个必要条件，而不去涉及破坏“互斥”条件。

2）打破请求和保持条件：采用资源预先分配策略，即进程运行前申请全部资源，满足则运行，不然就等待。 每个进程提出新的资源申请前，必须先释放它先前所占有的资源。

3）打破不可剥夺条件：当进程占有某些资源后又进一步申请其他资源而无法满足，则该进程必须释放它原来占有的资源。

4）打破环路等待条件：实现资源有序分配策略，将系统的所有资源统一编号，所有进程只能采用按序号递增的形式申请资源。



### ThreadLocal

ThreadLocal全称线程局部变量，主要解决多线程中数据因并发产生不一致问题。ThreadLocal为每一个线程都提供了变量threadLocals的副本，使得每个线程都能够独立地访问这个变量，隔离了多个线程对数据的数据共享。耗费了内存，但大大减少了线程同步所带来性能消耗，也减少了线程并发控制的复杂度。



**底层原理**

每个线程都要自己的一个map，map是一个数组的数据结构存储数据，每个元素是一个Entry，entry的key是threadlocal的引用，也就是当前变量的副本，value就是set的值。

1、Thread类中有变量threadLocals，类型为ThreadLocal.ThreadLocalMap，保存着每个线程的私有数据。

2、ThreadLocalMap是ThreadLocal的内部类，每个数据用Entry保存，其中的Entry继承于WeakReference，用一个键值对存储，键为ThreadLocal的引用。如果是强引用，即把ThreadLocal设置为null，GC也不会回收，因为ThreadLocalMap对它有强引用。

3、ThreadLocal中的**set方法**的实现逻辑，先获取当前线程，取出当前线程的ThreadLocalMap，如果不存在就会创建一个ThreadLocalMap，如果存在就会把当前的threadlocal的引用作为键，传入的参数作为值存入map中。

4、ThreadLocal中**get方法**的实现逻辑，获取当前线程，取出当前线程的ThreadLocalMap，用当前的threadlocak作为key在ThreadLocalMap查找，如果存在不为空的Entry，就返回Entry中的value，否则就会执行初始化并返回默认的值。

5、ThreadLocal中**remove方法**的实现逻辑，还是先获取当前线程的ThreadLocalMap变量，如果存在就调用ThreadLocalMap的remove方法。ThreadLocalMap的存储就是数组的实行，因此需要确定元素的位置，找到Entry，把entry的键值对都设为null，最后也Entry也设置为null。



### 线程通信

**生产者消费者模式**

**作用：**

（1）通过平衡生产者的生产能力和消费者的消费能力来提升整个系统的运行效率，这是生产者消费者模型最重要的作用

（2）解耦，这是生产者消费者模型附带的作用，解耦意味着生产者和消费者之间的联系少，联系越少越可以独自发展而不需要收到相互的制约

**通信方法：**

`wait()`		表示线程一直等待，会释放锁（`wait(timeout)`指定等待秒数）

`notify()`	唤醒处于等待的线程

`notifyAll()`	唤醒同一个对象上所有调用wait()方法的线程（优先级高的优先调度）



**并发协作模型（管理法）**

- 生产者：负责生产数据的模块
- 消费者：负责处理数据的模块
- 缓冲区：消费者不能直接使用生产者的数据，需要有缓冲区。生产者将生产好的数据放入缓冲区，消费者从缓冲区拿出数据。



**信号灯法**

- 设置一个标志位flag，如果为true，就让线程等待、如果为false，就让该线程去通知另一个线程、把两个线程衔接起来，就像咱们的信号灯红灯停，绿灯行，通过这样一个判断方式，只要来判断什么时候等待，什么时候将他唤醒。



### 线程池

**背景：**我们经常创建和销毁、使用量很大的资源，比如并发情况下的线程。

**思路：**提前创建好多个线程，放入线程池中，使用时直接获取，使用后放回池中。可以避免频繁创建销毁、实现重复利用。

**属性：**

- corePoolSize：线程池大小
- maximumPoolSize：最大线程数
- keepAliveTime：线程没有任务时最多保持多长时间后会终止



**创建线程池的优点：**

- 降低资源消耗。通过重复利用已创建的线程，降低线程创建和销毁造成的消耗。
- 提高响应速度。当任务到达时，任务可以不需要等到线程创建就能立即执行。
- 增加线程的可管理型。线程是稀缺资源，使用线程池可以进行统一分配，调优和监控。



**方法：**

`void executor(Runnable command)`	执行任务，没有返回值

`Future submit(Callable<T> task)`	执行任务，有返回值

`void shutdown()`	关闭线程池



# 反射

反射是指在运行状态中，对于任意一个类都能够知道这个类所有的属性和方法；并且对于任意一个对象，都能够调用它的任意一个方法；这种动态获取信息以及动态调用对象方法的功能称为反射机制。

### 反射机制

**需求：**通过外部文件配置，在不修改源码的情况下控制程序。满足开闭原则（不修改源码扩展功能）

**作用：**反射机制允许程序在执行期间借助于Reflection API取得任何类的内部信息（比如成员变量、构造器、成员方法、泛型、注解等），并能操作对象的属性及方法。反射在设计模式和框架底层都被广泛使用。具体而言，包括；

* 在运行时判断任意一个对象所属的类
* 在运行时构造任意一个类的对象
* 在运行时得到任意一个类所具有的成员变量和方法
* 在运行时调用任意一个对象的成员变量和方法
* 生成动态代理



**Java程序的三个阶段**

* 代码阶段/编译阶段：Java程序被javac编译为字节码文件（保存着一些变量、方法等）
* Class类阶段（类加载阶段）：通过类加载器将字节码文件加载（反射机制），在**堆**中生成Class类对象，用对象的方式保存成员变量、方法等
* Runtime运行阶段：生成对象后，该对象可以记录自己属于哪个Class类对象（互相关联）。通过Class对象可以调用对应的方法、属性。



**常用类：**java.lang.Class、java.lang.reflect.Method/Field/Constructor/Proxy



### 案例

通过配置文件使用反射机制的简单案例

demo:

```java
//使用IO文件流读取配置文件（使用Properties类）
Properties properties = new Properties();
properties.load(new FileInputStream("src\\main\\resources\\re.properties"));
String classfullpath = properties.get("classfullpath").toString();
Class cls = Class.forName(classfullpath);
Object o = cls.newInstance();//通过Class对象得到加载类的对象实例
Method method = cls.getMethod(methodName);//通过Class对象得到指定方法
method.invoke(o);//利用反射机制调用方法
Field nameField = cls.getField("age");//通过Class对象得到指定属性
System.out.println(nameField.get(o));//利用反射机制调用属性
Constructor constructor = cls.getConstructor();//通过Class对象得到无参构造器，括号中指定构造器参数类型
Constructor constructor2 = cls.getConstructor(String.class,int.class);//通过Class对象得到有参构造器，括号中指定构造器参数类型
```





### Class类分析

##### 主要特点

* Class类也是一个类，继承于Object
* Class类由系统自动创建，对于每个类内存中只有一份Class对象，因为类加载只进行一次。
* 每个对象知道自己属于哪个Class对象：`object.getClass()`
* 通过Class对象可以获得一个类的完整结构
* Class类对象存放在堆中
* 类的字节码二进制数据存放在方法区，也被称为类的元数据（包括方法代码、变量名、方法名、访问权限）



##### 常用方法

**基本信息**

`forName(String name)`	返回指定类名name的Class对象

`newInstance()`	调用无参构造函数，返回该Class对象的一个实例

`getName()`	返回该Class对象所表示的实体名称（类、接口、数组类、基本数据类型）

`getPackage()`	返回该Class对象所属的包信息

`getSuperClass()`	返回该Class对象的父类的全路径

`getClassLoader()`	返回该类的类加载器

**结构信息**

`getInterfaces()`	获取当前Class对象的所有接口（数组）

`getAnnotaions()`	获取当前Class对象的注解信息

`getConstructor(ClassObject...)`	返回Class对象所有本类以及父类的public构造器

`getDeclaredConstructor(ClassObject...)`	返回Class对象所有本类以及父类的所有构造器

`getConstructors()`	返回Class对象所有本类以及父类的public构造器（数组）

`getDeclaredConstructors()`	返回Class对象所有本类以及父类的所有构造器（数组）

`getField(String name)`	返回Class对象的名为name的Field对象

`getFields()`	返回Class对象所有本类以及父类的public属性（数组）

`getDeclaredFields()`	返回Class对象所有本类以及父类的所有属性（数组）

`getMethod(String name)`	返回Class对象的名为name的Method对象

`getMethods()`	返回Class对象所有本类以及父类的public方法（数组）

`getDeclaredMethods()`	返回Class对象所有本类以及父类的所有方法（数组）

**Field类方法**

`getModifiers()`	以int形式返回修饰符（默认为0，public为1，private为2，protected为4，static为16【通过加法得到结果】）

`getType()`	以Class形式返回类型

`getName()`	返回属性名

**Method类方法**

`getModifiers()`	以int形式返回修饰符（默认为0，public为1，private为2，protected为4，static为16【通过加法得到结果】）

`getReturnType()`	获得当前方法的返回值类型

**Constructor类方法**

`getModifiers()`	以int形式返回修饰符（默认为0，public为1，private为2，protected为4，static为16【通过加法得到结果】）

`getParameterTypes()`	返回参数列表的数组

`getName()`	返回构造器名



**获取Class类对象的方式**

编译阶段：`Class.forName(path)`	已知类的全类名（包名+类名），可以通过Class类的静态方法forName()获取（多用于配置文件的读取）

加载阶段：`类.class`	已知具体的类，通过类的class获取（多用于参数传递，比如通过反射得到对应构造器对象）

运行阶段：`对象.getClass()`	已知某个类的实例，可以调用该实例的getClass()方法获取Class对象（获取的是该对象的运行类型）

类加载器：

```java
ClassLoader classLoader = c3.getClass().getClassLoader();
Class<?> c4 = classLoader.loadClass(path);
```

特殊情况：

* 基本数据类型(int/char/float...)可以通过`.class`方式获取对应的Class对象
* 基本数据类型对应的包装类(Integer/Character/Float)，可以通过`.TYPE`获取对应的Class对象
* 实际上在java底层基本数据类型和其对应包装类持有同一个Class对象（hashcode相同）



**Class对象包含类型**

外部类（成员、静态、局部、匿名）、接口、数组、enum、annotation、基本数据类型、void



### 类加载

##### **动态加载**

反射机制是java实现动态语言的关键，也就是通过反射实现类动态加载，

静态加载：**编译时**加载相应的类，如果没有这个类则报错（必须存在该类编译才能通过）

动态加载：**运行时**加载需要的类，如果运行时不用该类则不报错，降低依赖性  （不存在该类编译也通过）



##### **类加载时机**

1、使用new创建对象时发生

2、当子类被加载，父类发生

3、调用类中的静态变量时发生（以上均为静态加载）

4、通过反射发生（动态加载）



##### **类加载过程**

**类加载的三个阶段：**加载、连接（验证、准备、解析）、初始化

* 加载：将类的class文件读入内存，并为之创建一个java.lang.Class对象（由JVM控制）

* 连接：将类的二进制数据合并到 JRE中（由JVM控制）
  * 验证：对文件的安全性进行验证
  * 准备：对静态变量分配内存并进行默认初始化
  * 解析：虚拟机将常量池中的符号引用替换为直接引用

* 初始化：执行静态代码块、静态变量的显式赋值（程序员控制）



**加载后内存分布情况：**

方法区：存放类的字节码二进制数据

堆区：存放类的Class对象（存在和方法区二进制数据的引用关系）



**加载阶段**

JVM会将将不同的数据源（包括class文件、jar包、甚至网络）转化为二进制字节流加载到内存中，并生成一个代表该类的Class对象。

**连接-验证阶段**

JVM需要确保Class文件的字节流中包含的信息符合当前虚拟机的要求，并且不会危害虚拟机本身的安全。

验证范围：文件格式验证（是否以魔术oxcafebabe开头）、元数据验证、字节码验证、符号引用验证

**连接-准备阶段**

JVM会在该阶段对静态变量分配内存并进行默认初始化，这些变量所使用的内存都将在方法区进行区分。

注意：实例属性不会分配内存；针对final static修饰的常量，由于一旦赋值就不会变化，因此jvm底层按照显式赋值。

**连接-解析阶段**

JVM将常量池中的符号引用替换为直接引用。

**初始化阶段**

操控者：程序员

初始化阶段真正执行类中定义的Java程序代码。此阶段是执行`<clinit>()`方法的过程。

注意：

* `<clinit>()`方法是由编译器按语句在源文件中出现的顺序，依次自动收集类中所有**静态变量**的赋值动作和**静态代码块**中的语句，并进行合并。

* 虚拟机会保证一个类的`<clinit>()`方法在多线程环境中被正确的加锁、同步，如果多个线程同时初始化一个类，那么只会有一个线程执行`<clinit>()`方法，其他线程都需要阻塞等待，直到活动线程执行`<clinit>()`方法完毕。



### 反射用途

##### 创建对象

通过反射创建对象有若干种方式：

* 调用类中public修饰的无参构造器

* 调用类中的指定构造器（public和private）

**相关方法：**

* 使用Class类中的相关方法：

  `newInstance()`	调用类中的无参构造器，获取对应类的对象

  `getConstructor(Class...clazz)`	根据参数列表，获取对应public构造器的对象

  `getDeclaredConstructor(Class...clazz)`	根据参数列表，获取对应所有构造器的对象

* 使用Constructor类中的相关方法：

  `setAccessible(true)`	爆破（通过这种方式可以访问私有的构造器方法）

  `newInstance(Object...obj)`	调用构造器

**步骤：**

```java
//先获取User类的Class对象
Class<?> userClass = Class.forName("com.twhupup.basic.User");
//1.通过public的无参构造器创建实例
Object o = userClass.newInstance();//newInstance()方法会直接调用无参构造器
//2.通过public的有参构造器创建实例
Constructor<?> constructor = userClass.getConstructor(int.class);//先得到对应的构造器对象
Object o1 = constructor.newInstance(20);//传入实参，创建实例
//3.通过非public的有参构造器创建实例
Constructor<?> constructor1 = userClass.getDeclaredConstructor(int.class, String.class);//先得到对应私有构造器
constructor1.setAccessible(true);//爆破，使用反射可以访问private构造器
Object o2 = constructor1.newInstance(30, "ming");//传入实参，创建实例
```



##### 访问类属性

**相关方法：**

根据属性名获取公有的Field对象：`Field f = clazz.getField(属性名)`

根据属性名获取Field对象：`Field f = clazz.getDeclaredField(属性名)`

设置属性：`f.set(o,value)`

暴力破解有权限属性：`f.setAccessible(true)`

注意点：如果属性被static修饰，在set()和get()中可以不指定对象

**步骤：**

```java
//获取Student的Class类
Class<?> stuClass = Class.forName("com.twhupup.Student");
//创建对象
Object o = stuClass.newInstance();
//使用反射操作公有属性
Field age = stuClass.getField("age");
age.set(o,88);//通过反射操作属性
System.out.println(age.get(o));//通过反射获取属性值
//使用反射操作私有属性
Field name = stuClass.getDeclaredField("name");
name.setAccessible(true);
name.set(o,"abc");//如果name是static属性，则set的一个传入参数可以为null
System.out.println(name.get(o));//如果name是static属性，则get的传入参数可以为null
```



##### 访问类方法

**相关方法：**

根据方法名和参数列表获取公有方法的Method对象：`Method m = clazz.getMethod(方法名, XX.class)`

根据方法名和参数列表获取Method对象：`Method m = clazz.getDeclaredMethod(方法名, XX.class)`

暴力破解有权限方法：`m.setAccessible(true)`

调用方法并得到返回值：`Object returnValue = m.invoke(o,实参列表)`

注意点：

* 如果方法被static修饰，在invoke方法中，可以传入null；
* 在方法中，如果方法有返回值，统一返回Object作为编译类型，但运行类型和方法定义保持一致

**步骤：**

```java
//获取Student的Class类
Class<?> bossClass = Class.forName("com.twhupup.Boss");
//创建对象
Object o = bossClass.newInstance();
//根据Class对象获取公有Method对象（需要在getMethod后添加实参的Class对象）
Method hi = bossClass.getMethod("hi",String.class);
//根据Class对象获取所有（私有）Method对象（需要在getDeclaredMethod后添加实参的Class对象）
Method say = bossClass.getDeclaredMethod("say",String.class);
hi.invoke(o,"twh");//使用反射调用方法
say.setAccessible(true);//爆破
say.invoke(o,"sjy");//使用反射调用方法
```













---



# 垃圾回收机制

java  语言中一个显著的特点就是引入了java回收机制。它使得java程序员在编写程序的时候不在考虑内存管理。由于有个垃圾回收机制，java中的对象不再有“作用域”的概念，只有对象的引用才有“作用域”。垃圾回收可以有效的防止内存泄露，有效的使用空闲的内存。



### 内存泄漏和内存溢出

**内存泄漏：**是指程序在申请内存后，无法释放已申请的内存空间，大量内存泄漏堆积后的后果就是内存溢出。

**内存溢出：**指程序申请内存时，没有足够的内存供申请者使用（或者说，给了你一块存储int类型数据的存储空间，但是你却存储long类型的数据，那么结果就是内存不够用）此时就会报错OOM，即所谓的内存溢出。 

**内存泄露量大到一定程度会导致内存溢出。但是内存溢出不一定是内存泄露引起的。**



**内存泄漏的分类**

* 常发性内存泄漏。发生内存泄漏的代码会被多次执行到，每次被执行的时候都会导致一块内存泄漏。
* 偶发性内存泄漏。发生内存泄漏的代码只有在某些特定环境或操作过程下才会发生。常发性和偶发性是相对的。对于特定的环境，偶发性的也许就变成了常发性的。所以测试环境和测试方法对检测内存泄漏至关重要。
* 一次性内存泄漏。发生内存泄漏的代码只会被执行一次，或者由于算法上的缺陷，导致总会有一块仅且一块内存发生泄漏。比如，在类的构造函数中分配内存，在析构函数中却没有释放该内存，所以内存泄漏只会发生一次。
* 隐式内存泄漏。程序在运行过程中不停的分配内存，但是直到结束的时候才释放内存。严格的说这里并没有发生内存泄漏，因为最终程序释放了所有申请的内存。但是对于一个服务器程序，需要运行几天，几周甚至几个月，不及时释放内存也可能导致最终耗尽系统的所有内存。所以，我们称这类内存泄漏为隐式内存泄漏



**内存溢出原因** 

* 内存中加载的数据量过于庞大，如一次从数据库取出过多数据； 
* 集合类中有对对象的引用，使用完后未清空，使得JVM不能回收； 
* 代码中存在死循环或循环产生过多重复的对象实体； 
* 使用的第三方软件中的BUG； 
* 启动参数内存值设定的过小
  



### 垃圾回收策略

使用**分代**的垃圾回收策略。

分代的原因：**不同的对象的生命周期是不一样的。不同生命周期的对象可以采取不同的回收算法，以便提高回收效率。**



**年轻代（Young Generation）**

* 所有新生成的对象首先都是放在年轻代的。年轻代的目标就是尽可能快速的收集掉那些生命周期短的对象。

* 新生代内存按照8:1:1的比例分为一个eden区和两个survivor(survivor0,survivor1)区。大部分对象在Eden区中生成。回收时先将eden区存活对象复制到一个survivor0区，然后清空eden区，当这个survivor0区也存放满了时，则将eden区和survivor0区存活对象复制到另一个survivor1区，然后清空eden和这个survivor0区，此时survivor0区是空的，然后将survivor0区和survivor1区交换，即保持survivor1区为空， 如此往复。

* 当survivor1区不足以存放 eden和survivor0的存活对象时，就将存活对象直接存放到老年代。若是老年代也满了就会触发一次Full GC，也就是新生代、老年代都进行回收

* 新生代发生的GC也叫做**Minor GC**，MinorGC发生频率比较高(不一定等Eden区满了才触发)



**年老代（Old Generation）**

* 在年轻代中经历了N次垃圾回收后仍然存活的对象，就会被放到年老代中。因此，可以认为年老代中存放的都是一些生命周期较长的对象。

* 内存比新生代也大很多(大概比例是1:2)，当老年代内存满时触发**Major GC**即**Full GC**，Full GC发生频率比较低，老年代对象存活时间比较长，存活率标记高。



**持久代（Permanent Generation）**

* 用于存放静态文件，如Java类、方法等。持久代对垃圾回收没有显著影响，但是有些应用可能动态生成或者调用一些class，例如Hibernate 等，在这种时候需要设置一个比较大的持久代空间来存放这些运行过程中新增的类。（在jdk新版本中，已经没有了永久代这个区域）
  



### 收集器

新生代收集器使用的收集器：Serial、PraNew、Parallel Scavenge

老年代收集器使用的收集器：Serial Old、Parallel Old、CMS



Serial收集器（复制算法)：新生代单线程收集器，标记和清理都是单线程，优点是简单高效。

Serial Old收集器(标记-整理算法)：老年代单线程收集器，Serial收集器的老年代版本。

ParNew收集器(停止-复制算法)　：新生代收集器，可以认为是Serial收集器的多线程版本，在多核CPU环境下有着比Serial更好的表现。

Parallel Scavenge收集器(停止-复制算法)：并行收集器，追求高吞吐量，高效利用CPU。吞吐量一般为99%， 吞吐量= 用户线程时间/(用户线程时间+GC线程时间)。适合后台应用等对交互相应要求不高的场景。

Parallel Old收集器(停止-复制算法)：Parallel Scavenge收集器的老年代版本，并行收集器，吞吐量优先。

CMS(Concurrent Mark Sweep)收集器（标记-清理算法）：高并发、低停顿，追求最短GC回收停顿时间，cpu占用比较高，响应时间快，停顿时间短，多核cpu 追求高响应时间的选择


### GC的执行机制

GC有两种类型：Scavenge GC和Full GC。

**Scavenge GC**

一般情况下，当新对象生成，并且在Eden申请空间失败时，就会触发Scavenge GC，对Eden区域进行GC，清除非存活对象，并且把尚且存活的对象移动到Survivor区。然后整理Survivor的两个区。这种方式的GC是对年轻代的Eden区进行，不会影响到年老代。因为大部分对象都是从Eden区开始的，同时Eden区不会分配的很大，所以Eden区的GC会频繁进行。因而，一般在这里需要使用速度快、效率高的算法，使Eden去能尽快空闲出来。

**Full GC**

对整个堆进行整理，包括Young、Tenured和Perm。Full GC因为需要对整个堆进行回收，所以比Scavenge GC要慢，因此应该尽可能减少Full GC的次数。在对JVM调优的过程中，很大一部分工作就是对于FullGC的调节。有如下原因可能导致Full GC：

1.年老代（Tenured）被写满

2.持久代（Perm）被写满

3.System.gc()被显示调用

4.上一次GC之后Heap的各域分配策略动态变化

















