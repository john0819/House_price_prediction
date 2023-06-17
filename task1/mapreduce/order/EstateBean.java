package com.xhy.mapreduce.order;

import org.apache.hadoop.io.WritableComparable;

import java.io.DataInput;
import java.io.DataOutput;
import java.io.IOException;

/**
 * @author xhy12137
 * @create 2023-03-12 19:30
 */
public class EstateBean implements WritableComparable<EstateBean> {
    public long id;
    public Double exchangeRate;
    public Double residenceSpace;
    public Double buildingSpace;
    public Double unitPriceOfResidenceSpace;
    public Double unitPriceOfBuildingSpace;
    public Double totalCost;

    public EstateBean() {
    }

    public long getId() {
        return id;
    }

    public void setId(long id) {
        this.id = id;
    }

    public Double getExchangeRate() {
        return exchangeRate;
    }

    public void setExchangeRate(Double exchangeRate) {
        this.exchangeRate = exchangeRate;
    }

    public Double getResidenceSpace() {
        return residenceSpace;
    }

    public void setResidenceSpace(Double residenceSpace) {
        this.residenceSpace = residenceSpace;
    }

    public Double getBuildingSpace() {
        return buildingSpace;
    }

    public void setBuildingSpace(Double buildingSpace) {
        this.buildingSpace = buildingSpace;
    }

    public Double getUnitPriceOfResidenceSpace() {
        return unitPriceOfResidenceSpace;
    }

    public void setUnitPriceOfResidenceSpace(Double unitPriceOfResidenceSpace) {
        this.unitPriceOfResidenceSpace = unitPriceOfResidenceSpace;
    }

    public Double getUnitPriceOfBuildingSpace() {
        return unitPriceOfBuildingSpace;
    }

    public void setUnitPriceOfBuildingSpace(Double unitPriceOfBuildingSpace) {
        this.unitPriceOfBuildingSpace = unitPriceOfBuildingSpace;
    }

    public Double getTotalCost() {
        return totalCost;
    }

    public void setTotalCost(Double totalCost) {
        this.totalCost = totalCost;
    }

    public void setTotalCost() {
        this.totalCost = this.exchangeRate*(this.residenceSpace*this.unitPriceOfResidenceSpace+this.buildingSpace*this.unitPriceOfBuildingSpace);
    }

    @Override
    public void write(DataOutput dataOutput) throws IOException {
        dataOutput.writeLong(id);
        dataOutput.writeDouble(exchangeRate);
        dataOutput.writeDouble(buildingSpace);
        dataOutput.writeDouble(residenceSpace);
        dataOutput.writeDouble(unitPriceOfResidenceSpace);
        dataOutput.writeDouble(unitPriceOfBuildingSpace);
        dataOutput.writeDouble(totalCost);
    }

    @Override
    public void readFields(DataInput dataInput) throws IOException {
        this.id = dataInput.readLong();
        this.exchangeRate = dataInput.readDouble();
        this.buildingSpace = dataInput.readDouble();
        this.residenceSpace = dataInput.readDouble();
        this.unitPriceOfResidenceSpace = dataInput.readDouble();
        this.unitPriceOfBuildingSpace = dataInput.readDouble();
        this.totalCost = dataInput.readDouble();
    }

    @Override
    public String toString() {
        return totalCost.toString();
    }

    @Override
    public int compareTo(EstateBean o) {
        if (this.id>o.id){
            return  1;
        }
        else if(this.id<o.id){
            return -1;
        }
        else {
            return 0;
        }
    }
}
