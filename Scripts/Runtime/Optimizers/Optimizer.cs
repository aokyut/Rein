using Rein.Functions;
using System.Collections.Generic;

namespace Rein.Optimizers{
    public abstract class Optimizer{
        protected List<Tensor> Parameters;
        public Optimizer(IFunction func){
            this.Parameters = func.Parameters;
        }
        public abstract void Update();
    }
}