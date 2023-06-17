package com.xhy.mapreduce.order;

import org.apache.hadoop.io.LongWritable;
import org.apache.hadoop.io.Text;
import org.apache.hadoop.mapreduce.Mapper;

import java.io.IOException;

/**
 * @author xhy12137
 * @create 2023-03-12 18:09
 */
public class EstateMapper extends Mapper<LongWritable, Text,EstateBean, LongWritable> {

    private LongWritable outV = new LongWritable();
    private EstateBean outK = new EstateBean();

    @Override
    protected void map(LongWritable key, Text value, Mapper<LongWritable, Text, EstateBean, LongWritable>.Context context) throws IOException, InterruptedException {
        String line = value.toString();
        //不读取csv第一行（column name）
        if (line.contains("date")) {
            return;
        }

        String[] split = line.split(",");

        Long id = Long.parseLong(split[21]);
        String exchangeRate = split[17];
        String residenceSpace  = split[3];
        String buildingSpace = split[4];
        String unitPriceOfResidenceSpace  = split[18];
        String unitPriceOfBuildingSpace  = split[19];
//        String totalCost  = split[20];

        outV.set(id);
        outK.setId(id);
        outK.setExchangeRate(Double.valueOf(exchangeRate));
        outK.setResidenceSpace(Double.valueOf(residenceSpace));
        outK.setBuildingSpace(Double.valueOf(buildingSpace));
        outK.setUnitPriceOfResidenceSpace(Double.valueOf(unitPriceOfResidenceSpace));
        outK.setUnitPriceOfBuildingSpace(Double.valueOf(unitPriceOfBuildingSpace));
        outK.setTotalCost();

        context.write(outK, outV);


    }

}
