using R = System.Double;
using System.Collections.Generic;
using System.Linq;
using System;

namespace Rein.Functions.Set{
    public class Mean: UnaryFunction{
        private int Axis;
        private bool KeepDim;
        public Mean(int axis, bool keepDim = true):base("Mean"){
            this.Axis = axis;
            this.KeepDim = keepDim;
        }

        protected override Tensor UnaryForward(Tensor inTensor){
            List<int> shape = new List<int>(inTensor.Shape);
            int biggerStep, step, vecSize, outSize;
            R[] data;

            vecSize = shape[this.Axis];
            outSize = this.In.Size / vecSize;
            step = this.In.Size / shape.GetRange(0, this.Axis + 1).Aggregate((now, next) => now * next);
            biggerStep = step * vecSize;
            data = new R[this.In.Size / vecSize];

            for (int i = 0; i < outSize; i+=step){
                for (int j = 0; j < biggerStep; j+=step){
                    for (int k = 0; k < step; k++)
                        data[i + k] += this.In.Data[vecSize * i + j + k] / vecSize;
                }
            }
            
            if (this.KeepDim){
                shape[Axis] = 1;
            }else{
                shape.RemoveAt(Axis);
            }

            return new Tensor(data, shape);
        }


        protected override void UnaryBackward(){

            List<int> shape = new List<int>(this.In.Shape);
            int meanNum = 1;

            meanNum *= shape[this.Axis];
            shape[this.Axis] = 1;

            int step, repNum;
            R[] grad;

            step = this.Out.Size / shape.GetRange(0, this.Axis + 1).Aggregate((now, next) => now * next);
            repNum = this.In.Shape[this.Axis];
            grad = new R[this.In.Size];
            
            // 総和処理
            for (int i = 0; i < this.Out.Size; i+=step){
                for (int j = 0; j < repNum; j++){
                    Array.Copy(this.Out.Grad, i, grad, i * repNum + j * step, step);
                }
            }


            for (int i = 0; i < this.In.Size; i++){
                this.In.Grad[i] += grad[i] / meanNum;
            }
        }
    }

    // 出力が一つの数値だけになる時
    public class MeanAll: UnaryFunction{
        private bool KeepDim;
        public MeanAll(bool keepDim): base("MeanAll"){
            this.KeepDim = keepDim;
        }

        protected override Tensor UnaryForward(Tensor tensor)
        {
            R sumR = 0;
            for (int i = 0; i < tensor.Size; i++){
                sumR += tensor.Data[i] / tensor.Size;
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
                this.In.Grad[i] += this.Out.Grad[0] / this.In.Size;
            }
        }
    }
}