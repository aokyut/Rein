using R = System.Double;
using System;

namespace Rein.Functions{
    public class Lambda: UnaryFunction{
        protected Func<R, R> LambdaForward, LambdaBackward;
        public Lambda(string name, Func<R, R> forward, Func<R, R> backward):base(name){
            this.LambdaForward = forward;
            this.LambdaBackward = backward;
        }

        protected override Tensor UnaryForward(Tensor tensor)
        {
            R[] data = new R[this.In.Size];

            for (int i = 0; i < this.In.Size; i++){
                data[i] = this.LambdaForward(this.In.Data[i]);
            }

            return new Tensor(data, this.In.Shape);
        }

        protected override void UnaryBackward()
        {
            for (int i = 0; i < this.In.Size; i++){
                this.In.Grad[i] += this.Out.Grad[i] * this.LambdaBackward(this.In.Data[i]);
            }
        }
    }
}