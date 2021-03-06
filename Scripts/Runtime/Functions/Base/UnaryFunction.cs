namespace Rein.Functions{
    public abstract class UnaryFunction: BaseFunction{
        
        protected abstract Tensor UnaryForward(Tensor tensor);
        protected abstract void UnaryBackward();
        protected Tensor In{
            get{
                return this.Inputs[0];
            }
            set{
                this.Inputs[0] = value;
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
        
        public UnaryFunction(string name):base(name){
            this.FunctionForward = (tensors) => new Tensor[1]{ this.UnaryForward(tensors[0]) };
            this.FunctionBackward = this.UnaryBackward;
        }


    }
}