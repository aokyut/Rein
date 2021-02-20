﻿using Rein.Functions.Arithmetic;
using Rein.Utils.Exceptions;

namespace Rein
{
    public partial class Tensor
    {
        // 演算子のオーバーロード
        public static Tensor operator +(Tensor tensor1, Tensor tensor2)
        {
            return new Add().Forward(tensor1, tensor2);
        }

        public static Tensor operator -(Tensor tensor1, Tensor tensor2)
        {
            return new Sub().Forward(tensor1, tensor2);
        }

        public static Tensor operator -(Tensor tensor)
        {
            return F.Minus(tensor);
        }

        public static Tensor operator *(Tensor tensor1, Tensor tensor2)
        {
            return new Mul().Forward(tensor1, tensor2);
        }

        public static Tensor operator /(Tensor tensor1, Tensor tensor2)
        {
            return new Div().Forward(tensor1, tensor2);
        }

        public static implicit operator Tensor(Tensor[] tensor1)
        {
            if(!(tensor1.Length == 1))throw new InvalidLengthException();
            return tensor1[0];
        }
    }
}