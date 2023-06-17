package com.xhy.mapreduce.order;

import org.apache.hadoop.conf.Configuration;
import org.apache.hadoop.fs.FileSystem;
import org.apache.hadoop.fs.Path;
import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Job;
import org.apache.hadoop.mapreduce.lib.input.FileInputFormat;
import org.apache.hadoop.mapreduce.lib.output.FileOutputFormat;

import java.io.IOException;

/**
 * @author xhy12137
 * @create 2023-03-12 18:10
 */
public class EstateDriver {

    public static void main(String[] args) throws IOException, ClassNotFoundException, InterruptedException {
        // 1 获取job
        Configuration conf = new Configuration();
        Job job = Job.getInstance(conf);

        // 2 设置jar
        job.setJarByClass(EstateDriver.class);

        // 3 关联mapper 和Reducer
        job.setMapperClass(EstateMapper.class);
        job.setReducerClass(EstateReducer.class);

        // 4 设置mapper 输出的key和value类型
        job.setMapOutputKeyClass(EstateBean.class);
        job.setMapOutputValueClass(LongWritable.class);

        // 5 设置最终数据输出的key和value类型
        job.setOutputKeyClass(LongWritable.class);
        job.setOutputValueClass(EstateBean.class);

        //设置reducer数量为2，设置数据的分区方法estatepartitioner
        job.setNumReduceTasks(2);
        job.setPartitionerClass(EstatePartitioner.class);


        // 判断output文件夹是否存在，如果存在则删除
        Path path = new Path("D:\\mapreduce\\output\\");
        FileSystem fileSystem = path.getFileSystem(conf);
        if (fileSystem.exists(path)) {
            fileSystem.delete(path, true);
        }

        // 6 设置数据的输入路径和输出路径
        FileInputFormat.setInputPaths(job, new Path("D:\\mapreduce\\input"));
        FileOutputFormat.setOutputPath(job, new Path("D:\\mapreduce\\output"));

        // 7 提交job
        boolean result = job.waitForCompletion(true);
        System.exit(result ? 0 : 1);

    }
}
