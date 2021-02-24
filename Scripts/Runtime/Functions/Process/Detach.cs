using Rein.Utils.Exceptions;

namespace Rein.Functions.Process{
    public class Detach: BaseFunction{
        public Detach():base("Detach"){

        }
        public override Tensor[] Forward(params Tensor[] inputs){
            Tensor tensor = inputs[0];
            tensor.BackFunction = null;
            return new Tensor[]{tensor};
        }

        public override void Backward(){
            // 何もしない
        }
    }
}