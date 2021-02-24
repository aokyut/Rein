using Rein.Functions.Arithmetic;
using Rein.Utils.Exceptions;
using Rein.Functions.Set;
using R = System.Double;

namespace Rein
{
    public partial class Tensor
    {
        // 演算子のオーバーロード
        public static Tensor operator +(Tensor tensor1, Tensor tensor2)
        {
            return new Add().Forward(tensor1, tensor2);
        }

        public static Tensor operator +(Tensor tensor, R r){
            return F.AddConst(tensor, r);
        }
        public static Tensor operator +(R r, Tensor tensor){
            return F.AddConst(tensor, r);
        }

        public static Tensor operator -(Tensor tensor1, Tensor tensor2)
        {
            return new Sub().Forward(tensor1, tensor2);
        }
        public static Tensor operator -(Tensor tensor, R r)
        {
            return (-tensor) + r;
        }
        public static Tensor operator -(R r, Tensor tensor)
        {
            return (-tensor) + r;
        }

        public static Tensor operator -(Tensor tensor)
        {
            return F.Minus(tensor);
        }

        public static Tensor operator *(Tensor tensor1, Tensor tensor2)
        {
            return new Mul().Forward(tensor1, tensor2);
        }
        public static Tensor operator *(Tensor tensor, R r)
        {
            return F.MulConst(tensor, r);
        }
        public static Tensor operator *(R r, Tensor tensor)
        {
            return F.MulConst(tensor, r);
        }

        public static Tensor operator /(Tensor tensor1, Tensor tensor2)
        {
            return new Div().Forward(tensor1, tensor2);
        }
        public static Tensor operator /(Tensor tensor, R r){
            return (1 / r) * tensor;
        }
        public static Tensor operator /(R r, Tensor tensor){
            return F.DivConst(tensor, r);
        }

        public static implicit operator Tensor(Tensor[] tensor1)
        {
            if(!(tensor1.Length == 1))throw new InvalidLengthException();
            return tensor1[0];
        }

        public Tensor Sum(int axis, bool keepDim = true){
            return new Sum(axis, keepDim).Forward(this);
        }
        public Tensor Sum(bool keepDim = true){
            return new SumAll(keepDim).Forward(this);
        }

        public Tensor Mean(int axis, bool keepDim = true){
            return new Mean(axis, keepDim).Forward(this);
        }
        public Tensor Mean(bool keepDim = true){
            return new MeanAll(keepDim).Forward(this);
        }

        public Tensor Max(int axis, bool keepDim = true){
            return new Max(axis, keepDim).Forward(this);
        }
        public Tensor Max(bool keepDim = true){
            return new MaxAll(keepDim).Forward(this);
        }

        public Tensor Min(int axis, bool keepDim = true){
            return new Min(axis, keepDim).Forward(this);
        }
        public Tensor Min(bool keepDim = true){
            return new MinAll(keepDim).Forward();
        }
    }
}