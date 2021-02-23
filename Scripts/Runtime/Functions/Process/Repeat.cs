using Rein.Utils.Exceptions;
using System.Linq;
using System;
using R = System.Double;
using System.Collections.Generic;

namespace Rein.Functions.Process{
    public class Repeat: UnaryFunction{
        int Dim, RepeatNum, Step;
        public Repeat(int dim, int rep): base($"Expand-d:{dim}-r:{rep}"){
            this.Dim = dim;
            this.RepeatNum = rep;
        }

        protected override Tensor UnaryForward(Tensor tensor)
        {
            if (tensor.Shape.Count < this.Dim - 1) throw new InvalidShapeException();
            R[] data = new R[tensor.Size * this.RepeatNum];
            // 対象となるDim以下の要素数を取得
            this.Step = tensor.Shape[this.Dim] * tensor.Size / tensor.Shape.GetRange(0, this.Dim + 1).Aggregate((now, next) => now * next);

            for (int i = 0; i < tensor.Size; i += this.Step){
                for (int j = 0; j < this.RepeatNum; j++){
                    Array.Copy(tensor.Data, i, data, i * this.RepeatNum + j * this.Step, this.Step);
                }
            }

            List<int> shape = new List<int>(tensor.Shape);
            shape[this.Dim] *= this.RepeatNum;
            
            return new Tensor(data, shape);
        }

        protected override void UnaryBackward()
        {
            for (int i = 0; i < this.In.Size; i += this.Step){
                for (int j = 0; j < this.RepeatNum; j++){
                    for (int k = 0; k < this.Step; k++){
                        this.In.Grad[i + k] += this.Out.Grad[i * this.RepeatNum + j * this.Step + k];
                    }
                }
            }
        }
    }
}