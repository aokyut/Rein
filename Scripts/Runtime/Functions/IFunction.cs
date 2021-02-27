using Rein;
using System.Collections.Generic;

namespace Rein.Functions
{
    public interface IFunction
    {
        public Tensor[] Forward(params Tensor[] inputs);

        public Tensor[] Predict(params Tensor[] inputs);
        public void Backward();

        public List<Tensor> Parameters {get; }

        public void Train();
        public void Eval();
    }
}
