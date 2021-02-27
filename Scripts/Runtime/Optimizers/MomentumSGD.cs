using R = System.Double;
using Rein.Functions;

namespace Rein.Optimizers{
    public class MomentumSGD: Optimizer{
        private R LearningRate, Alpha;
        private R[][] MomentumParameters;

        public MomentumSGD(IFunction func, R learningRate = 0.01, R alpha = 0.9):base(func){
            this.LearningRate = learningRate;
            this.Alpha = alpha;

            this.MomentumParameters = new R[this.Parameters.Count][];

            for (int i = 0; i < this.Parameters.Count; i++){
                this.MomentumParameters[i] = new R[this.Parameters[i].Size];
            }
        }

        // Updateを書き換える
        public override void Update(){
            for (int i = 0; i < this.Parameters.Count; i++){
                for (int j = 0; j < this.Parameters[i].Size; j++){
                    this.MomentumParameters[i][j] =  (1 - this.Alpha) * this.Parameters[i].Grad[j] + this.Alpha * this.MomentumParameters[i][j];;
                    this.Parameters[i].Data[j] -= this.LearningRate * this.MomentumParameters[i][j];
                }
            }
        }
    }
}