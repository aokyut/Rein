using R = System.Double;
using System;

namespace Rein.Functions.Loss{
    public class HuberLoss: BinaryFunction{
        R Delta;
        R DeltaSquare;
        public HuberLoss(R delta):base("HuberLoss"){
            this.Delta = delta;
            this.DeltaSquare = delta * delta / 2;
        }

        protected override Tensor BinaryForward(Tensor tensor1, Tensor tensor2)
        {
            Tensor lossTensor = new Lambda(
                "HuberLossFunction",
                new Func<R, R>((x) => 
                x < -this.Delta ? -this.Delta * x - this.DeltaSquare : 
                (x > this.Delta ? this.Delta * x - this.DeltaSquare : x * x / 2)),
                new Func<R, R>((x) => 
                x < -this.Delta ? -this.Delta :
                (x > this.Delta ? this.Delta : x))
                ).Forward(tensor1 - tensor2);
            
            return lossTensor.Sum();
        }

        protected override void BinaryBackward()
        {
            // 何も書かない
        }
    }
}