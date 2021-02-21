using R = System.Double;
using System.Collections.Generic;
using System.Linq;
using System;

namespace Rein.Functions.Set{
    public class Min: UnaryFunction{
        private int Axis;
        private int[] ExtractIndex;
        private bool KeepDim;
        public Min(int axis, bool keepDim = true):base("Min"){
            this.Axis = axis;
            this.KeepDim = keepDim;
        }

        protected override Tensor UnaryForward(Tensor inTensor){
            List<int> shape = new List<int>(inTensor.Shape);
            int biggerStep, step, vecSize, outSize;
            R[] data;
            int[] extractIndex;

            vecSize = shape[this.Axis];
            outSize = this.In.Size / vecSize;
            step = this.In.Size / shape.GetRange(0, this.Axis + 1).Aggregate((now, next) => now * next);
            biggerStep = step * vecSize;
            data = new R[this.In.Size / vecSize];
            extractIndex = new int[this.In.Size / vecSize];

            for (int i = 0; i < outSize; i+=step){
                for (int k = 0; k < step; k++){
                    R minR = this.In.Data[vecSize * i + k];
                    int index = 0;
                    for (int j = 0; j < biggerStep; j+=step){
                        if (minR > this.In.Data[vecSize * i + j + k]){
                            minR = this.In.Data[vecSize * i + j + k];
                            index = j;
                        }
                        data[i + k] += this.In.Data[vecSize * i + j + k] / vecSize;
                    }
                    data[i + k] = minR;
                    extractIndex[i + k] = index;
                }
            }
            
            if (this.KeepDim){
                shape[Axis] = 1;
            }else{
                shape.RemoveAt(Axis);
            }

            this.ExtractIndex = extractIndex;

            return new Tensor(data, shape);
        }


        protected override void UnaryBackward(){
            int step, repNum;
            R[] grad;
            grad = this.Out.Grad;

            step = this.Out.Size / this.Out.Shape.GetRange(0, this.Axis + 1).Aggregate((now, next) => now * next);
            repNum = this.In.Shape[this.Axis];
            grad = new R[this.In.Size];
            
            for (int i = 0; i < this.Out.Size; i+=step){
                for (int k = 0; k < step; k++){
                    this.In.Grad[repNum * i + this.ExtractIndex[i + k] + k] += this.Out.Grad[i + k];
                }
            }
        }
    }

    public class MinAll: UnaryFunction{
        private int Index;
        private bool KeepDim;
        public MinAll(bool keepDim): base("MinAll"){
            this.KeepDim = keepDim;
        }

        protected override Tensor UnaryForward(Tensor tensor)
        {
            R maxR = tensor.Data[0];

            for (int i = 0; i < tensor.Size; i++){
                if (maxR > tensor.Data[i]){
                    maxR = tensor.Data[i];
                    this.Index = i;
                }
            }

            if (this.KeepDim){
                return new Tensor(
                    new R[]{maxR},
                    Enumerable.Repeat(1, tensor.Shape.Count).ToList()
                );
            }else{
                return new Tensor(
                    new R[]{maxR},
                    new List<int>(){1}
                );
            }
        }

        protected override void UnaryBackward()
        {
            this.In.Grad[this.Index] = this.Out.Grad[0];
        }
    }
}