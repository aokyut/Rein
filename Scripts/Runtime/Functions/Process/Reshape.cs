using System.Collections.Generic;
using System.Linq;
using Rein.Utils.Exceptions;

namespace Rein.Functions.Process{
    public class Reshape: UnaryFunction{
        private List<int> OutShape;
        private List<int> InShape;
        public Reshape(List<int> shape): base($"Reshape-({string.Join(",", shape)})"){
            this.OutShape = shape;
        }

        protected override Tensor UnaryForward(Tensor tensor)
        {
            // サイズ確認
            if (this.OutShape.Aggregate((now, next) => now * next) != tensor.Size)
                throw new InvalidShapeException($"[Reshape] Expected Output Shape : ({string.Join(",", this.OutShape)})  ,Input Shape :({string.Join(",", tensor.Shape)})");
            this.InShape = tensor.Shape;
            tensor.Shape = this.OutShape;

            return tensor;
        }

        protected override void UnaryBackward()
        {
            this.In.Shape = this.InShape;
        }
    }
}