package com.xhy.mapreduce.order;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.mapreduce.Partitioner;

import javax.xml.soap.Text;

/**
 * @author xhy12137
 * @create 2023-03-13 21:58
 */
public class EstatePartitioner extends Partitioner<EstateBean, LongWritable> {

    @Override
    public int getPartition(EstateBean estateBean, LongWritable longWritable, int i) {
        Long id = longWritable.get();
        int partition = 0;
        if (id<=2000){
            partition = 0;
        }
        else if (id>2000){
            partition = 1;
        }
        
        return partition;

    }
}
