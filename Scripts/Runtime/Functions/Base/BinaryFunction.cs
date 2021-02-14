namespace Rein.Functions
{
    public abstract class BinaryFunction: BaseFunction
    {
        protected abstract Tensor BinaryForward(Tensor tensor1, Tensor tensor2);
        protected abstract void BinaryBackward();

        public BinaryFunction()
        {
            this.FunctionForward = (tensor) => new Tensor[1]{ this.BinaryForward(tensor[0], tensor[1]) };
            this.FunctionBackward = this.BinaryBackward;
        }
    }
}