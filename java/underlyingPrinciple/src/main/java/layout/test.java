import org.openjdk.jol.info.ClassLayout;

/**
 * @Project: underlyingPrinciple
 * @Package: classLayout
 * @Date: 2022/2/22 19:11
 * @Author: Wenhao Tan
 * @Version: 1.0
 * @License: (C)2022, MIPAV Lab(mipav.net), Soochow University. tanritian1@163.com All Rights Reserved.
 */
public class test {
    private static class T{

    }

    public static void main(String[] args) throws Exception {
        T t = new T();
        System.out.println(ClassLayout.parseInstance(t).toPrintable());
    }
}
