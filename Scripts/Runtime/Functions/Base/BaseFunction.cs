using System;
using System.Collections.Generic;

namespace Rein.Functions
{
    public abstract class BaseFunction: IFunction
    {
        protected Tensor[] Inputs, Outputs;
        public List<Tensor> Params = new List<Tensor>();
        protected int UseCount = 0;
        
        protected Func<Tensor[], Tensor[]> FunctionForward;
        protected Action FunctionBackward;
        
        protected string Name;
        public bool RequireGrad;

        public BaseFunction(string name, bool requireGrad = true){
            this.Name = name;
            this.RequireGrad = requireGrad;
        }

        public virtual Tensor[] Forward(params Tensor[] inputs)
        {
            if (!this.RequireGrad) return this.Predict(inputs);
            foreach (Tensor input in inputs){
                input.UseCount++;
            }
            this.Inputs = inputs;
            this.Outputs = this.FunctionForward(inputs);
            foreach (Tensor output in this.Outputs){
                output.BackFunction = output.BackFunction ?? this;
            }
            this.UseCount = this.Outputs.Length;

            return this.Outputs;
        }

        // 勾配を保存しない
        public virtual Tensor[] Predict(params Tensor[] inputs){
            return this.FunctionForward(inputs);
        }

        public virtual void Backward()
        {
            this.UseCount--;
            if (this.UseCount != 0)return;
            this.FunctionBackward();
            foreach(Tensor input in this.Inputs){
                input.BackwardChain();
            }
        }

        public List<Tensor> Parameters{
            get{
                return this.Params;
            }
        }

        public void Train(){
            this.RequireGrad = true;
        }
    
        public void Eval(){
            this.RequireGrad = false;
        }
    }
}