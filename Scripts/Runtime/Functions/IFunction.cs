using Rein;

namespace Rein.Functions
{
    public interface IFunction
    {
        public Tensor[] Forward(params Tensor[] inputs);
        public void Backward();

        public Tensor[] GetParams {get;}
    }
}
