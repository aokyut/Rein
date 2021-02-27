using Rein;
using System.Collections.Generic;

namespace Rein.Functions
{
    public interface IFunction
    {
        Tensor[] Forward(params Tensor[] inputs);

        Tensor[] Predict(params Tensor[] inputs);
        void Backward();

        List<Tensor> Parameters {get; }

        void Train();
        void Eval();
    }
}
