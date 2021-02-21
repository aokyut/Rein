using System;

namespace Rein.Functions
{
    public abstract class BaseFunction: IFunction
    {
        protected Tensor[] Inputs, Outputs;
        protected int UseCount = 0;
        
        protected Func<Tensor[], Tensor[]> FunctionForward;
        protected Action FunctionBackward;
        
        protected string Name;

        public BaseFunction(string name){
            this.Name = name;
        }

        public Tensor[] Forward(params Tensor[] inputs)
        {
            foreach (Tensor input in inputs){
                input.UseCount++;
            }
            this.Inputs = inputs;
            this.Outputs = this.FunctionForward(inputs);
            foreach (Tensor output in this.Outputs){
                output.BackFunction = this;
            }
            this.UseCount = this.Outputs.Length;

            return this.Outputs;
        }

        public void Backward()
        {
            this.UseCount--;
            if (this.UseCount != 0)return;
            this.FunctionBackward();
            foreach(Tensor input in this.Inputs){
                input.Backward();
            }
        }
    }
}