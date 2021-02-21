using R = System.Double;
using System.Collections.Generic;
using System.Linq;
using System;

namespace Rein.Functions.Set{
    public class Sum: UnaryFunction{
        private int Axis;
        private bool KeepDim;
        public Sum(int axis, bool keepDim = true):base("Sum"){
            this.Axis = axis;
            this.KeepDim = keepDim;
        }

        protected override Tensor UnaryForward(Tensor inTensor){
            List<int> shape = new List<int>(inTensor.Shape);
            int biggerStep, step, outSize, vecSize;
            R[] data;


            vecSize = shape[this.Axis];
            outSize = this.In.Size / vecSize;
            step = this.In.Size / shape.GetRange(0, this.Axis + 1).Aggregate((now, next) => now * next);
            biggerStep = step * vecSize;
            data = new R[this.In.Size / shape[this.Axis]];

            for (int i = 0; i < outSize; i+=step){
                for (int j = 0; j < biggerStep; j+=step){
                    for (int k = 0; k < step; k++)
                        data[i + k] += this.In.Data[vecSize * i + j + k];
                }
            }
            
            if (this.KeepDim){
                shape[this.Axis] = 1;
            }else{
                shape.RemoveAt(this.Axis);
            }

            return new Tensor(data, shape);
        }


        protected override void UnaryBackward(){

            List<int> shape = new List<int>(this.In.Shape);

            shape[this.Axis] = 1;

            int step, repNum;
            R[] grad;

            step = this.Out.Size / shape.GetRange(0, this.Axis + 1).Aggregate((now, next) => now * next);
            repNum = this.In.Shape[this.Axis];
            grad = new R[this.In.Size];
            
            for (int i = 0; i < this.Out.Size; i+=step){
                for (int j = 0; j < repNum; j++){
                    Array.Copy(this.Out.Grad, i, grad, i * repNum + j * step, step);
                }
            }

            for (int i = 0; i < this.In.Size; i++){
                this.In.Grad[i] += grad[i];
            }
        }
    }

    public class SumAll: UnaryFunction{
        private bool KeepDim;
        public SumAll(bool keepDim = true): base("SumAll"){
            this.KeepDim = keepDim;
        }

        protected override Tensor UnaryForward(Tensor tensor)
        {
            R sumR = 0;
            for (int i = 0; i < tensor.Size; i++){
                sumR += tensor.Data[i];
            }

            if (this.KeepDim){
                return new Tensor(
                    new R[]{sumR},
                    Enumerable.Repeat(1, tensor.Shape.Count).ToList()
                );
            }else{
                return new Tensor(
                    new R[]{sumR},
                    new List<int>(){1}
                );
            }
        }

        protected override void UnaryBackward()
        {
            for (int i = 0; i < this.In.Size; i++){
                this.In.Grad[i] += this.Out.Grad[0];
            }
        }
    }
}