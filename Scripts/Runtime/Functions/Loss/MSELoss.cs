using System;
using R = System.Double;

namespace Rein.Functions.Loss{
    public class MSELoss: BinaryFunction{
        public MSELoss(): base("MSELoss"){

        }

        protected override Tensor BinaryForward(Tensor inTensor, Tensor target)
        {
            Tensor lossTensor = new Lambda(
                "MSELoss",
                new Func<R, R>((x) => x * x),
                new Func<R, R>((x) => 2 * x)
            ).Forward(inTensor - target);
            return lossTensor.Mean();
        }

        protected override void BinaryBackward()
        {
            // 何もしない
        }
    }
}