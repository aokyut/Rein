using Rein.Functions.Process;
using System.Collections.Generic;


namespace Rein
{
    public partial class Tensor
    {
        public Tensor Repeat(int dim, int rep){
            return new Repeat(dim, rep).Forward(this);
        }

        public Tensor Detach(){
            return new Detach().Forward(this);
        }

        public Tensor Squeeze(int dim){
            return new Squeeze(dim).Forward(this);
        }

        // 全部の要素についてSqueezeを試みる
        public Tensor Squeeze(){
            return new SqueezeAll().Forward(this);
        }
        
        public Tensor  Unsqueeze(int dim = 0){
            return new Unsqueeze(dim).Forward(this);
        }

        public Tensor Reshape(List<int> shape){
            return new Reshape(shape).Forward(this);
        }
    }
}