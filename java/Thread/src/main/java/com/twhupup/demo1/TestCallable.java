package com.twhupup.demo2;

import org.apache.commons.io.FileUtils;

import java.io.File;
import java.io.IOException;
import java.net.URL;
import java.util.concurrent.*;

/**
 * @Project: Thread
 * @Package: com.twhupup.demo2
 * @Date: 2022/3/29 20:14
 * @Author: Wenhao Tan
 * @Version: 1.0
 * @License: (C)2022, MIPAV Lab(mipav.net), Soochow University. tanritian1@163.com All Rights Reserved.
 */
public class TestCallable implements Callable<Boolean> {
    private final String url;//网络图片地址
    private final String name;//保存文件名

    public TestCallable(String url, String name) {
        this.url = url;
        this.name = name;
    }

    //加载图片线程的执行体
    @Override
    public Boolean call() {
        WebDownloader webDownloader = new WebDownloader();
        webDownloader.downloader(url,name);
        System.out.println(name+"下载完成");
        return true;
    }

    public static void main(String[] args) throws ExecutionException, InterruptedException {
        TestCallable t1 = new TestCallable("https://gimg2.baidu.com/image_search/src=http%3A%2F%2Fnewbbs-fd.zol-img.com.cn%2Ft_s1200x5000%2Fg6%2FM00%2F07%2F0B%2FChMkKmE3ammIQ_CsAAwcB7PVr-IAATfpQPOV5MADBwf181.jpg&refer=http%3A%2F%2Fnewbbs-fd.zol-img.com.cn&app=2002&size=f9999,10000&q=a80&n=0&g=0n&fmt=auto?sec=1650795859&t=55e0ae3360d90c5f921dc98b99fc36ce","output1.jpg");
        TestCallable t2 = new TestCallable("https://gimg2.baidu.com/image_search/src=http%3A%2F%2Fnewbbs-fd.zol-img.com.cn%2Ft_s1200x5000%2Fg6%2FM00%2F07%2F0B%2FChMkKmE3ammIQ_CsAAwcB7PVr-IAATfpQPOV5MADBwf181.jpg&refer=http%3A%2F%2Fnewbbs-fd.zol-img.com.cn&app=2002&size=f9999,10000&q=a80&n=0&g=0n&fmt=auto?sec=1650795859&t=55e0ae3360d90c5f921dc98b99fc36ce","output2.jpg");
        TestCallable t3 = new TestCallable("https://gimg2.baidu.com/image_search/src=http%3A%2F%2Fnewbbs-fd.zol-img.com.cn%2Ft_s1200x5000%2Fg6%2FM00%2F07%2F0B%2FChMkKmE3ammIQ_CsAAwcB7PVr-IAATfpQPOV5MADBwf181.jpg&refer=http%3A%2F%2Fnewbbs-fd.zol-img.com.cn&app=2002&size=f9999,10000&q=a80&n=0&g=0n&fmt=auto?sec=1650795859&t=55e0ae3360d90c5f921dc98b99fc36ce","output3.jpg");
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

class WebDownloader{
    public void downloader(String url,String name){
        try {
            FileUtils.copyURLToFile(new URL(url),new File(name));
        } catch (IOException e) {
            e.printStackTrace();
            System.out.println("IO异常，downloader方法出现问题");
        }
    }
}

