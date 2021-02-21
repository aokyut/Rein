using R = System.Double;
using System.Collections.Generic;
using System.Linq;
using System;

namespace Rein.Functions.Set{
    public class Sum: UnaryFunction{
        private List<int> Axes;
        private bool KeepDim;
        public Sum(List<int> axes, bool keepDim = true):base("Sum"){
            this.Axes = axes;
            this.Axes.Sort((a, b) => b - a);
            this.KeepDim = keepDim;
        }

        protected override Tensor UnaryForward(Tensor inTensor){
            List<int> shape = new List<int>(inTensor.Shape);
            int size, offset, step, repNum, sizeAfter;
            R[] data, nextData;
            data = inTensor.Data;
            foreach (int axis in this.Axes){
                repNum = shape[axis];
                size = shape.Aggregate((now, next) => now * next);
                step = size / shape.GetRange(0, axis + 1).Aggregate((now, next) => now * next);
                offset = step * shape[axis];
                nextData = new R[size / shape[axis]];
                
                // 総和処理
                for (int i = 0; i * offset < size; i++){
                    for (int j = 0; j < offset; j+=step){
                        for (int k = 0; k < step; k++){
                            nextData[step * i + k] += data[offset * i + j + k];
                        }
                    }
                }

                data = nextData;
                //
                
                if (this.KeepDim){
                    shape[axis] = 1;
                }else{
                    shape.RemoveAt(axis);
                }
            }

            return new Tensor(data, shape);
        }


        protected override void UnaryBackward(){

            List<int> shape = new List<int>(this.In.Shape);
            foreach (int axis in this.Axes){
                shape[axis] = 1;
            }

            int size, step, repNum;
            R[] grad, nextGrad;
            grad = this.Out.Grad;
            foreach (int axis in this.Axes){
                size = shape.Aggregate((now, next) => now * next);
                step = size / shape.GetRange(0, axis + 1).Aggregate((now, next) => now * next);
                repNum = this.In.Shape[axis];
                nextGrad = new R[size * repNum];
                
                // 総和処理
                for (int i = 0; i < size; i+=step){
                    for (int j = 0; j < repNum; j++){
                        Array.Copy(grad, i, nextGrad, i * repNum + j * step, step);
                    }
                }

                grad = nextGrad;
                shape[axis] = repNum;
            }

            for (int i = 0; i < this.In.Size; i++){
                this.In.Grad[i] += grad[i];
            }
        }
    }
}