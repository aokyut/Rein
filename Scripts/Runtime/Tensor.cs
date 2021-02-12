using R = System.Double;
using System.Linq;
using System.Collections.Generic;
using System;
using System.Runtime.Serialization;
using Rein.Utills.Exceptions;

namespace Rein
{
    [Serializable]
    public class Tensor
    {
        private R[] Param, Grad;
        private List<int> Shape;
        private int Size;

        public Tensor(int[] shape)
        {
            System.Random random = new System.Random();
            this.Shape = shape.ToList();
            this.Size = shape.Aggregate((now, next) => now * next);
            this.Param = Enumerable.Range(0, this.Size).Select(_ => (R)random.NextDouble()).ToArray();
            this.Grad = new R[this.Size];
        }

        public Tensor(R[] data, int[] shape)
        {
            this.Shape = shape.ToList();
            this.Size = shape.Aggregate((now, next) => now * next);
            if(data.Length != this.Size) throw new InvalidSizeException();
            this.Param = data;
            this.Grad = new R[this.Size];
        }

        // 演算子のオーバーロード
        public static Tensor operator +(Tensor tensor1, Tensor tensor2)
        {
            return null;
        }

        public static Tensor operator -(Tensor tensor1, Tensor tensor2)
        {
            return null;
        }

        public static Tensor operator -(Tensor tensor)
        {
            return null;
        }

        public static Tensor operator *(Tensor tensor1, Tensor tensor2)
        {
            return null;
        }

        public static Tensor operator /(Tensor tensor1, Tensor tensor2)
        {
            return null;
        }
    }
}
