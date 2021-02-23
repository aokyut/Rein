using R = System.Double;
using System.Collections.Generic;
using System;
using Rein.Functions;
using Rein.Functions.Layer;

namespace Rein.Tests{
    public class NetworkTest: Test{
        public NetworkTest(): base("NetworkTest"){
            this.Tests = new List<Action>(){
                this.TestLinear,
            };


            this.RunTest();
        }

        public void TestLinear(){
            Linear linear = new Linear(2, 3);
            Linear linear2 = new Linear(3, 3);
            linear.Weight.Data = new R[]{
                1, 2, 3,
                4, 5, 6
            };
            linear.Bias.Data = new R[]{
                0.0, 0.0, 0.0
            };

            Tensor inTensor = new Tensor(
                new R[]{
                    0.1, 0.3,
                    2.0, 1.0,
                },
                new List<int>(){2, 2}
            );

            R[] expectSecond = new R[]{
                1.3, 1.7, 2.1,
                6, 9, 12
            };

            R[] expectLinearGrad = new R[]{
                1.0, 1.0, 1.0,
                1.0, 1.0, 1.0
            };

            R[] expectBiasGrad = new R[]{
                2.0, 2.0, 2.0
            };

            R[] expectWeightGrad = new R[]{
                2.1, 2.1, 2.1,
                1.3, 1.3, 1.3
            };

            R[] expectInputGrad = new R[]{
                6, 15,
                6, 15
            };

            Tensor secondTensor = (linear as IFunction).Forward(inTensor);
            Tensor resultTensor = F.Sum(secondTensor);

            resultTensor.Grad = new R[]{ 1.0 };
            resultTensor.UseCount++;
            resultTensor.Backward();
            
            this._CheckArrayEqual(resultTensor.Data, new R[]{32.1}, "TestLinearOut");
            this._CheckArrayEqual(secondTensor.Data, expectSecond, "TestLinearSecondData");
            this._CheckArrayEqual(secondTensor.Grad, expectLinearGrad, "Test");
            this._CheckArrayEqual(linear.Weight.Grad, expectWeightGrad, "TestLinearWeightGrad");
            this._CheckArrayEqual(linear.Bias.Grad, expectBiasGrad, "TestLinearBiasGrad");
            this._CheckArrayEqual(inTensor.Grad, expectInputGrad, "TestLinearInputGrad");
        }
    }
}