using Rein.Functions.Process;

namespace Rein
{
    public partial class Tensor
    {
        public Tensor Repeat(int dim, int rep){
            return new Repeat(dim, rep).Forward(this);
        }
    }
}