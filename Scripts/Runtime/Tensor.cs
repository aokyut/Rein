using System.Linq;
using System.Collections.Generic;
using System;
using System.Runtime.Serialization;
using R = System.Double;
using Rein.Utils.Exceptions;
using Rein.Functions;

namespace Rein
{
    [Serializable]
    public partial class Tensor
    {
        // Dataは変数データ、Gradは勾配データを格納する
        public R[] Data, Grad;
        // Shapeはデータの形を保存している
        public List<int> Shape;
        public int Size;
        // UseCountは計算グラフで使用された回数を保存することで、勾配の計算漏れを防ぐ
        public int UseCount = 0;
        // Backward時に呼び出す。IFunctionはRein.Functionsのinterface
        public IFunction BackFunction;

        // データの形で初期化
        public Tensor(int[] shape)
        {
            System.Random random = new System.Random();
            this.Shape = shape.ToList();
            this.Size = shape.Aggregate((now, next) => now * next);
            // 乱数で初期化
            this.Data = Enumerable.Range(0, this.Size).Select(_ => (R)random.NextDouble()).ToArray();
            this.Grad = new R[this.Size];
        }

        // データを直接入力して初期化
        public Tensor(R[] data)
        {
            this.Shape = new List<int>(1){ data.Length };
            this.Size = data.Length;
            this.Data = data;
            this.Grad = new R[this.Size];
        }

        // データとshapeで初期化
        public Tensor(R[] data, int[] shape):this(data, shape.ToList())
        {
        }

        public Tensor(R[] data, List<int> shape){
            this.Shape = shape;
            this.Size = shape.Aggregate((now, next) => now * next);
            if(data.Length != this.Size) throw new InvalidSizeException();
            this.Data = data;
            this.Grad = new R[this.Size];
        }

        public void Backward()
        {
            // BackFunctionが存在しない時は終了
            if(this.BackFunction == null)return;
            this.UseCount--;
            // 他の関数に対しても出力している場合にはまだ勾配を計算しない
            if(this.UseCount != 0)return;
            this.BackFunction.Backward();
        }
    }
}
