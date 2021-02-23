using Rein.Functions;
using System.Linq;

namespace Rein.Functions.Layer{
    public class Linear: UnaryFunction{
        public Tensor Weight, Bias;
        public IFunction Activation;

        public Linear(int inputSize, int outputSize, bool bias = true, IFunction activation = null): base("Linear"){
            this.Weight = new Tensor(new int[]{inputSize, outputSize});
            this.Activation = activation;

            if (bias){
                this.Bias = new Tensor(new int[]{outputSize});
            }

            this.Params = new Tensor[]{Weight, Bias};
        }

        protected override Tensor UnaryForward(Tensor tensor)
        {
            Tensor outTensor = F.Dot(tensor, this.Weight);

            if (this.Bias != null){
                if (outTensor.Shape.Count == 0){
                    outTensor += this.Bias;
                }else{
                    outTensor += this.Bias.Repeat(0, outTensor.Size / outTensor.Shape.Last());
                }
            }

            if (this.Activation != null){
                return this.Activation.Forward(outTensor);
            }

            return outTensor;
        }

        protected override void UnaryBackward()
        {
            // 何もしない
        }
    }
}