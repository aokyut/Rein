using Rein.Functions;
using R = System.Double;
using System;

namespace Rein.Optimizers{
    public class Adam: Optimizer{
        private R Alpha, Beta1, Beta2, Epsilon, BetaPower1, BetaPower2;
        private R[][] Momentum, Velocity;
        public Adam(IFunction func, R alpha = 0.001, R beta1 = 0.9, R beta2 = 0.999, R epsilon = 0.00000001):base(func){
            this.Alpha = alpha;
            this.Beta1 = beta1;
            this.Beta2 = beta2;
            this.Epsilon = epsilon;
            this.BetaPower1 = 1.0;
            this.BetaPower2 = 1.0;

            this.Momentum = new R[this.Parameters.Count][];
            this.Velocity = new R[this.Parameters.Count][];
            
            for (int i = 0; i < this.Parameters.Count; i++){
                this.Momentum[i] = new R[this.Parameters[i].Size];
                this.Velocity[i] = new R[this.Parameters[i].Size];
            }
        }

        public override void Update(){
            this.BetaPower1 *= this.Beta1;
            this.BetaPower2 *= this.Beta2;
            for (int i = 0; i < this.Parameters.Count; i++){
                for (int j = 0; j < this.Parameters[i].Size; j++){
                    this.Momentum[i][j] = this.Beta1 * this.Momentum[i][j] + (1 - this.Beta1) * this.Parameters[i].Grad[j];
                    this.Velocity[i][j] = this.Beta2 * this.Velocity[i][j] + (1 - this.Beta2) * this.Parameters[i].Grad[j] * this.Parameters[i].Grad[j];
                    
                    R mHat = this.Momentum[i][j] / (1 - this.BetaPower1);
                    R vHat = this.Velocity[i][j] / (1 - this.BetaPower2);
                    this.Parameters[i].Data[j] -= this.Alpha * mHat / (Math.Sqrt(vHat) + this.Epsilon);
                }
            }
        }
    }
}