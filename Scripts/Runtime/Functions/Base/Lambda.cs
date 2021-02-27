using R = System.Double;
using System;
using System.Collections.Generic;

namespace Rein.Functions{
    public class Lambda: UnaryFunction{
        protected Func<R, R> LambdaForward, LambdaBackward;
        public Lambda(string name, Func<R, R> forward, Func<R, R> backward):base(name){
            this.LambdaForward = forward;
            this.LambdaBackward = backward;
        }

        protected override Tensor UnaryForward(Tensor tensor)
        {
            R[] data = new R[tensor.Size];

            for (int i = 0; i < tensor.Size; i++){
                data[i] = this.LambdaForward(tensor.Data[i]);
            }

            return new Tensor(data, new List<int>(tensor.Shape));
        }

        protected override void UnaryBackward()
        {
            for (int i = 0; i < this.In.Size; i++){
                this.In.Grad[i] += this.Out.Grad[i] * this.LambdaBackward(this.In.Data[i]);
            }
        }
    }
}