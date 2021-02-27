using System.Collections.Generic;

namespace Rein.Functions.Pipeline{
    public class ModuleList: BaseFunction{
        private IFunction[] Modules;
        public ModuleList(params IFunction[] modules):base("ModuleList"){
            this.Modules = modules;
            this.Params =  new List<Tensor>();
            foreach(IFunction module in modules){
                this.Params.AddRange(module.Parameters);
            }
        }

        public override Tensor[] Forward(params Tensor[] inputs){
            Tensor[] X = inputs;
            foreach(IFunction module in this.Modules){
                X = module.Forward(X);
            }
            return X;
        }

        public override Tensor[] Predict(params Tensor[] inputs){
            Tensor[] X = inputs;
            foreach(IFunction module in this.Modules){
                X = module.Predict(X);
            }
            return X;
        }

        public override void Backward()
        {
            // 内部にあるIFunctionで処理が行われる。
            return;
        }
    }
}