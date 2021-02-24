using System.Collections.Generic;

namespace Rein.Functions.Process{
    public class Squeeze: UnaryFunction{
        private List<int> InShape;
        private int Dim;
        public Squeeze(int dim): base($"Squeeze-{dim}"){
            this.Dim = dim;
        }

        protected override Tensor UnaryForward(Tensor tensor)
        {
            this.InShape = new List<int>(tensor.Shape);
            if(tensor.Shape[this.Dim] == 1)tensor.Shape.RemoveAt(this.Dim);
            return tensor;
        }

        protected override void UnaryBackward()
        {
            this.In.Shape = this.InShape;
        }
    }

    public class SqueezeAll: UnaryFunction{
        private List<int> InShape;
        public SqueezeAll(): base("SqueezeAll"){

        }

        protected override Tensor UnaryForward(Tensor tensor)
        {
            this.InShape = new List<int>(tensor.Shape);
            for (int i = 0; i < tensor.Shape.Count; i++){
                if (tensor.Shape[i] == 1)tensor.Shape.RemoveAt(i);
            }

            return tensor;
        }

        protected override void UnaryBackward()
        {
            this.In.Shape = this.InShape;
        }
    }
}