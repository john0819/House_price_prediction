package com.xhy.mapreduce.order;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Reducer;

import javax.swing.text.StyledEditorKit;
import java.io.IOException;

/**
 * @author xhy12137
 * @create 2023-03-12 18:10
 */
public class EstateReducer extends Reducer<EstateBean, LongWritable,LongWritable, EstateBean> {
    @Override
    protected void reduce(EstateBean key, Iterable<LongWritable> values, Reducer<EstateBean, LongWritable, LongWritable, EstateBean>.Context context) throws IOException, InterruptedException {
        for (LongWritable value:values) {
            context.write(value,key);
        }
    }
}
