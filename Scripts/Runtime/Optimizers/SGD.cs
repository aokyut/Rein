using Rein.Functions;
using R = System.Double;

namespace Rein.Optimizers{
    public class SGD: Optimizer{
        private R LearningRate;
        public SGD(IFunction func, R learningRate): base(func){
            this.LearningRate = learningRate;
        }

        public override void Update()
        {
            foreach(Tensor parameter in this.Parameters){
                for (int i = 0; i < parameter.Size; i++){
                    parameter.Data[i] -= this.LearningRate * parameter.Grad[i];
                }
            }
        }
    }
    
}