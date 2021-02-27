using R = System.Double;
using System.Collections.Generic;
using System;
using Rein.Functions;
using Rein.Functions.Layer;
using Rein.Functions.Pipeline;
using Rein.Optimizers;


namespace Rein.Tests{
    public class OptimizerTest: Test{
        private Func<R, R, R> testFunction = (x, y) => Math.Sin(x * 2 * Math.PI) - Math.Cos(y * 2 * Math.PI);
        private Func<R, R, R> testFunction2 = (x, y) => x - y;
        public OptimizerTest():base("OptimizerTest"){
            this.Tests = new List<Action>(){

            };

            this.RunTest();
        }

        public void TestOptimizer(){
            IFunction network = new ModuleList(
                    N.Linear(2, 10),
                    N.ReLU(),
                    N.Linear(10, 1)
                );
            // Optimizer sgd = new SGD(network, 0.01);
            Optimizer sgd = new MomentumSGD(network);
            Random rand = new Random();
            // Tensor[] data = this._TestData(1000);
            for (int i = 0; i < 100; i++){
                Tensor[] data = this._TestData(100);

                // Tensor X = network.Forward(
                //     data[0]
                // );
                Tensor X = network.Forward(
                    data[0]
                );
                Tensor loss = F.MSELoss(X, data[1]);
                // Console.WriteLine($"{i}step:{loss.Data[0]}");
                Console.WriteLine($"{loss.Data[0]}");
                loss.Backward();
                sgd.Update();
            }
            
        }

        // TestFunction z = x^2 - y^2
        private Tensor[] _TestData(int size){
            Tensor In = new Tensor(new int[]{size, 2});
            Tensor Out = new Tensor(new int[]{size});
            for (int i = 0; i < size; i++){
                Out.Data[i] = this.testFunction(In.Data[2 * i], In.Data[2 * i + 1]);
            }
            return new Tensor[]{In, Out};
        }
    }
}