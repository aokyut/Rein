using Rein.Functions;
using System.Linq;
using System.Collections.Generic;
using Rein.Functions.Arithmetic;
using Rein.Functions.Process;

namespace Rein.Functions.Layer{
    public class Linear: BaseFunction{
        public Tensor Weight, Bias;

        public Linear(int inputSize, int outputSize, bool bias = true): base("Linear"){
            this.Weight = new Tensor(new int[]{inputSize, outputSize});

            if (bias){
                this.Bias = new Tensor(new int[]{outputSize});
            }

            this.Params = new List<Tensor>(){Weight, Bias};
            this.FunctionBackward = this.LinearBackward;
        }

        public override Tensor[] Forward(Tensor[] tensors){
            return this.LinearForward(tensors, requireGrad: true);
        }

        public override Tensor[] Predict(Tensor[] tensors){
            return this.LinearForward(tensors, requireGrad: false);
        }

        public Tensor[] LinearForward(Tensor[] tensors, bool requireGrad){
            Tensor outTensor = new Dot(){RequireGrad = requireGrad}.Forward(tensors[0], this.Weight);
            if (this.Bias != null){
                outTensor = new Add(){RequireGrad = requireGrad}.Forward(
                    outTensor,
                    new Repeat(0, outTensor.Size / outTensor.Shape.Last()){RequireGrad = requireGrad}.Forward(this.Bias)
                );
            }

            return new Tensor[]{outTensor};
        }

        public void LinearBackward(){

        }
    }
}