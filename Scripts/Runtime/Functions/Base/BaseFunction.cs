using System;

namespace Rein.Functions
{
    public abstract class BaseFunction: IFunction
    {
        protected Tensor[] Inputs, Outputs;
        protected int UseCount = 0;
        
        protected Func<Tensor[], Tensor[]> FunctionForward;
        protected Action FunctionBackward;

        public Tensor[] Forward(params Tensor[] inputs)
        {
            this.Inputs = inputs;
            this.Outputs = this.FunctionForward(inputs);
            this.UseCount = this.Outputs.Length;

            return this.Outputs;
        }

        public void Backward()
        {
            this.UseCount--;
            if (this.UseCount != 0)return;
            this.FunctionBackward();
        }
    }
}