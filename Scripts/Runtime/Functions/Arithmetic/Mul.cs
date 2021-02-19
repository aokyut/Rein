using System.Linq;
using R = System.Double;
using System.Collections.Generic;

namespace Rein.Functions.Arithmetic{
    public class Mul: BinaryFunction{

        public Mul(): base("Mul"){

        }
        protected override Tensor BinaryForward(Tensor left, Tensor right){
            R[] data = new R[left.Size];
            
            for(int i=0; i < left.Size; i++){
                data[i] = left.Data[i] * right.Data[i];
            }

            return new Tensor(
                data,
                new List<int>(left.Shape)
            );
        }

        protected override void BinaryBackward(){
            for(int i = 0; i < this.Left.Size; i++){
                this.Left.Grad[i] += this.Out.Grad[i] * this.Right.Data[i];
                this.Right.Grad[i] += this.Out.Grad[i] * this.Left.Data[i];
            }
        }
    }
}