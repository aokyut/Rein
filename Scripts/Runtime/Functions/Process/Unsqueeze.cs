using System.Collections.Generic;

namespace Rein.Functions.Process{
    public class Unsqueeze: UnaryFunction{
        private List<int> InShape;
        private int Dim;
        public Unsqueeze(int dim): base($"Unsqueeze-{dim}"){
            this.Dim = dim;
        }

        protected override Tensor UnaryForward(Tensor tensor)
        {
            this.InShape = new List<int>(tensor.Shape);
            tensor.Shape.Insert(this.Dim, 1);
            return tensor;
        }

        protected override void UnaryBackward()
        {
            this.In.Shape = this.InShape;
        }
    }
}