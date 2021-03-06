﻿namespace Rein.Functions
{
    public abstract class BinaryFunction: BaseFunction
    {
        protected abstract Tensor BinaryForward(Tensor tensor1, Tensor tensor2);
        protected abstract void BinaryBackward();

        protected Tensor Left{
            get{
                return this.Inputs[0];
            }
            set{
                this.Inputs[0] = value;
            }
        }
        protected Tensor Right{
            get{
                return this.Inputs[1];
            }
            set{
                this.Inputs[1] = value;
            }
        }
        protected Tensor Out{
            get{
                return this.Outputs[0];
            }
            set{
                this.Outputs[0] = value;
            }
        }

        public BinaryFunction(string name): base(name)
        {
            this.FunctionForward = (tensor) => new Tensor[1]{ this.BinaryForward(tensor[0], tensor[1]) };
            this.FunctionBackward = this.BinaryBackward;
        }
    }
}