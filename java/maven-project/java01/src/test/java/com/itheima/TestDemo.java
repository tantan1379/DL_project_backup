import org.junit.Assert;
import org.junit.Test;

/**
 * @Project: maven-project
 * @Package: PACKAGE_NAME
 * @Date: 2022/2/19 17:31
 * @Author: Wenhao Tan
 * @Version: 1.0
 * @License: (C)2022, MIPAV Lab(mipav.net), Soochow University. tanritian1@163.com All Rights Reserved.
 */
public class TestDemo {
    @Test
    public void testSay(){
        Demo d = new Demo();
        String ret = d.say("twh");
        Assert.assertEquals("hello twh",ret);
    }
}
