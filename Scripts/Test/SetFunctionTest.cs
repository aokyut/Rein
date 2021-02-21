using R = System.Double;
using System.Collections.Generic;
using System;

namespace Rein.Tests{
    public class SetFunctionTest: Test{
        public SetFunctionTest(): base("SetFunctionTest")
        {
            this.Tests = new List<Action>(){
                this.TestSum,
                this.TestSumGrad,
                this.TestMean,
                this.TestMeanGrad,
                this.TestMax,
                this.TestMaxGrad,
                this.TestMin,
                this.TestMinGrad,
            };


            this.RunTest();
        }

        public void TestSum(){
            Tensor input = new Tensor(
                new R[]{
                    -29, -28, -27, -26, -25,
                    -24, -23, -22, -21, -20,
                    -19, -18, -17, -16, -15,
                    -14, -13, -12, -11, -10,

                    -9, -8, -7, -6, -5,
                    -4, -3, -2, -1, 0,
                    1, 2, 3, 4, 5,
                    6, 7, 8, 9, 10,

                    11, 12, 13, 14, 15,
                    16, 17, 18, 19, 20,
                    21, 22, 23, 24, 25,
                    26, 27, 28, 29, 30
                }, new int[3]{3, 4, 5}
            );

            // axis = 2
            R[] out1 = new R[]{
                -135, -110, -85, -60,
                -35, -10, 15, 40,
                65, 90, 115, 140
            };

            // axis = 1
            R[] out2 = new R[]{
                -86, -82, -78, -74, -70,
                -6, -2, 2, 6, 10,
                74, 78, 82, 86, 90
            };

            // axis = 0 2
            R[] out3 = new R[]{
                -105, -30, 45, 120
            };

            Tensor Out1 = input.Sum(2);
            Tensor Out2 = input.Sum(1);
            Tensor Out3 = input.Sum(2).Sum(0);

            this._CheckArrayEqual(out1, Out1.Data, "TestSum");
            this._CheckArrayEqual(out2, Out2.Data, "TestSum");
            this._CheckArrayEqual(out3, Out3.Data, "TestSum");

            Console.WriteLine("TestSum");
        }
        public void TestSumGrad(){
            
            Tensor input = new Tensor(
                new R[]{
                    -29, -28, -27, -26, -25,
                    -24, -23, -22, -21, -20,
                    -19, -18, -17, -16, -15,
                    -14, -13, -12, -11, -10,

                    -9, -8, -7, -6, -5,
                    -4, -3, -2, -1, 0,
                    1, 2, 3, 4, 5,
                    6, 7, 8, 9, 10,

                    11, 12, 13, 14, 15,
                    16, 17, 18, 19, 20,
                    21, 22, 23, 24, 25,
                    26, 27, 28, 29, 30
                }, new int[3]{3, 4, 5}
            );

            // axis = 0 2
            R[] grad = new R[]{
                1 ,1, 1, 1, 1,
                2, 2, 2, 2, 2,
                3, 3, 3, 3, 3,
                4, 4, 4, 4, 4,

                1 ,1, 1, 1, 1,
                2, 2, 2, 2, 2,
                3, 3, 3, 3, 3,
                4, 4, 4, 4, 4,

                1 ,1, 1, 1, 1,
                2, 2, 2, 2, 2,
                3, 3, 3, 3, 3,
                4, 4, 4, 4, 4,
            };

            Tensor Out1 = input.Sum(2).Sum(0);

            Out1.Grad = new R[]{1, 2, 3, 4};
            Out1.UseCount++;

            Out1.Backward();


            this._CheckArrayEqual(grad, input.Grad, "TestSumGrad");

            Console.WriteLine("TestSumGrad");
        }

        public void TestMean(){
            
            Tensor input = new Tensor(
                new R[]{
                    -29, -28, -27, -26, -25,
                    -24, -23, -22, -21, -20,
                    -19, -18, -17, -16, -15,
                    -14, -13, -12, -11, -10,

                    -9, -8, -7, -6, -5,
                    -4, -3, -2, -1, 0,
                    1, 2, 3, 4, 5,
                    6, 7, 8, 9, 10,

                    11, 12, 13, 14, 15,
                    16, 17, 18, 19, 20,
                    21, 22, 23, 24, 25,
                    26, 27, 28, 29, 30
                }, new int[3]{3, 4, 5}
            );

            // axis = 2
            R[] out1 = new R[]{
                -27, -22, -17, -12,
                -7, -2, 3, 8,
                13, 18, 23, 28
            };

            // axis = 1
            R[] out2 = new R[]{
                -21.5, -20.5, -19.5, -18.5, -17.5,
                -1.5, -0.5, 0.5, 1.5, 2.5,
                18.5, 19.5, 20.5, 21.5, 22.5
            };

            // axis = 0 2
            R[] out3 = new R[]{
                -7, -2, 3, 8
            };

            Tensor Out1 = input.Mean(2);
            Tensor Out2 = input.Mean(1);
            Tensor Out3 = input.Mean(2).Mean(0);

            this._CheckArrayEqual(out1, Out1.Data, "TestMean");
            this._CheckArrayEqual(out2, Out2.Data, "TestMean");
            this._CheckArrayEqual(out3, Out3.Data, "TestMean");

            Console.WriteLine("TestMean");
        }
        public void TestMeanGrad(){
            
            Tensor input = new Tensor(
                new R[]{
                    -29, -28, -27, -26, -25,
                    -24, -23, -22, -21, -20,
                    -19, -18, -17, -16, -15,
                    -14, -13, -12, -11, -10,

                    -9, -8, -7, -6, -5,
                    -4, -3, -2, -1, 0,
                    1, 2, 3, 4, 5,
                    6, 7, 8, 9, 10,

                    11, 12, 13, 14, 15,
                    16, 17, 18, 19, 20,
                    21, 22, 23, 24, 25,
                    26, 27, 28, 29, 30
                }, new int[3]{3, 4, 5}
            );

            // axis = 0 2
            R[] grad = new R[]{
                1.0 / 15.0 , 1.0 / 15.0 , 1.0 / 15.0 , 1.0 / 15.0 , 1.0 / 15.0 , 
                2.0 / 15.0 , 2.0 / 15.0 , 2.0 / 15.0 , 2.0 / 15.0 , 2.0 / 15.0 , 
                3.0 / 15.0 , 3.0 / 15.0 , 3.0 / 15.0 , 3.0 / 15.0 , 3.0 / 15.0 , 
                4.0 / 15.0 , 4.0 / 15.0 , 4.0 / 15.0 , 4.0 / 15.0 , 4.0 / 15.0 , 
                
                1.0 / 15.0 , 1.0 / 15.0 , 1.0 / 15.0 , 1.0 / 15.0 , 1.0 / 15.0 , 
                2.0 / 15.0 , 2.0 / 15.0 , 2.0 / 15.0 , 2.0 / 15.0 , 2.0 / 15.0 , 
                3.0 / 15.0 , 3.0 / 15.0 , 3.0 / 15.0 , 3.0 / 15.0 , 3.0 / 15.0 , 
                4.0 / 15.0 , 4.0 / 15.0 , 4.0 / 15.0 , 4.0 / 15.0 , 4.0 / 15.0 , 
                
                1.0 / 15.0 , 1.0 / 15.0 , 1.0 / 15.0 , 1.0 / 15.0 , 1.0 / 15.0 , 
                2.0 / 15.0 , 2.0 / 15.0 , 2.0 / 15.0 , 2.0 / 15.0 , 2.0 / 15.0 , 
                3.0 / 15.0 , 3.0 / 15.0 , 3.0 / 15.0 , 3.0 / 15.0 , 3.0 / 15.0 , 
                4.0 / 15.0 , 4.0 / 15.0 , 4.0 / 15.0 , 4.0 / 15.0 , 4.0 / 15.0 , 
            };

            Tensor Out1 = input.Mean(2).Mean(0);

            Out1.Grad = new R[]{1, 2, 3, 4};
            Out1.UseCount++;

            Out1.Backward();


            this._CheckArrayEqual(grad, input.Grad, "TestMeanGrad");

            Console.WriteLine("TestMeanGrad");
        }


        public void TestMax(){
            
            Tensor input = new Tensor(
                new R[]{
                    -29, -28, -27, -26, -25,
                    -24, -23, -22, -21, -20,
                    -19, -18, -17, -16, -15,
                    -14, -13, -12, -11, -10,

                    -9, -8, -7, -6, -5,
                    -4, -3, -2, -1, 0,
                    1, 2, 3, 4, 5,
                    6, 7, 8, 9, 10,

                    11, 12, 13, 14, 15,
                    16, 17, 18, 19, 20,
                    21, 22, 23, 24, 25,
                    26, 27, 28, 29, 30
                }, new int[3]{3, 4, 5}
            );

            // axis = 2
            R[] out1 = new R[]{
                -25, -20, -15, -10,
                -5, 0, 5, 10,
                15, 20, 25, 30
            };

            // axis = 1
            R[] out2 = new R[]{
                -14, -13, -12, -11, -10,
                6, 7, 8, 9, 10,
                26, 27, 28, 29, 30
            };

            // axis = 0 2
            R[] out3 = new R[]{
                15, 20, 25, 30
            };

            Tensor Out1 = input.Max(2);
            Tensor Out2 = input.Max(1);
            Tensor Out3 = input.Max(2).Max(0);

            this._CheckArrayEqual(out1, Out1.Data, "TestMax1");
            this._CheckArrayEqual(out2, Out2.Data, "TestMax2");
            this._CheckArrayEqual(out3, Out3.Data, "TestMax3");

            Console.WriteLine("TestMax");
        }
        public void TestMaxGrad(){
            
            Tensor input = new Tensor(
                new R[]{
                    -29, -28, -27, -26, -25,
                    -24, -23, -22, -21, -20,
                    -19, -18, -17, -16, -15,
                    -14, -13, -12, -11, -10,

                    -9, -8, -7, -6, -5,
                    -4, -3, -2, -1, 0,
                    1, 2, 3, 4, 5,
                    6, 7, 8, 9, 10,

                    11, 12, 13, 14, 15,
                    16, 17, 18, 19, 20,
                    21, 22, 23, 24, 25,
                    26, 27, 28, 29, 30
                }, new int[3]{3, 4, 5}
            );

            // axis = 0 2
            R[] grad = new R[]{
                0, 0, 0, 0, 0,
                0, 0, 0, 0, 0,
                0, 0, 0, 0, 0,
                0, 0, 0, 0, 0,
                
                0, 0, 0, 0, 0,
                0, 0, 0, 0, 0,
                0, 0, 0, 0, 0,
                0, 0, 0, 0, 0,

                0, 0, 0, 0, 1,
                0, 0, 0, 0, 2,
                0, 0, 0, 0, 3,
                0, 0, 0, 0, 4,
            };

            Tensor Out1 = input.Max(2).Max(0);

            Out1.Grad = new R[]{1, 2, 3, 4};
            Out1.UseCount++;

            Out1.Backward();


            this._CheckArrayEqual(grad, input.Grad, "TestMaxGrad");

            Console.WriteLine("TestMaxGrad");
        }

        public void TestMin(){
            
            Tensor input = new Tensor(
                new R[]{
                    -29, -28, -27, -26, -25,
                    -24, -23, -22, -21, -20,
                    -19, -18, -17, -16, -15,
                    -14, -13, -12, -11, -10,

                    -9, -8, -7, -6, -5,
                    -4, -3, -2, -1, 0,
                    1, 2, 3, 4, 5,
                    6, 7, 8, 9, 10,

                    11, 12, 13, 14, 15,
                    16, 17, 18, 19, 20,
                    21, 22, 23, 24, 25,
                    26, 27, 28, 29, 30
                }, new int[3]{3, 4, 5}
            );

            // axis = 2
            R[] out1 = new R[]{
                -29, -24, -19, -14,
                -9, -4, 1, 6,
                11, 16, 21, 26
            };

            // axis = 1
            R[] out2 = new R[]{
                -29, -28, -27, -26, -25,
                -9, -8, -7, -6, -5,
                11, 12, 13, 14, 15
            };

            // axis = 0 2
            R[] out3 = new R[]{
                -29, -24, -19, -14
            };

            Tensor Out1 = input.Min(2);
            Tensor Out2 = input.Min(1);
            Tensor Out3 = input.Min(2).Min(0);

            this._CheckArrayEqual(out1, Out1.Data, "TestMin1");
            this._CheckArrayEqual(out2, Out2.Data, "TestMin2");
            this._CheckArrayEqual(out3, Out3.Data, "TestMin3");

            Console.WriteLine("TestMin");
        }
        public void TestMinGrad(){
            
            Tensor input = new Tensor(
                new R[]{
                    -29, -28, -27, -26, -25,
                    -24, -23, -22, -21, -20,
                    -19, -18, -17, -16, -15,
                    -14, -13, -12, -11, -10,

                    -9, -8, -7, -6, -5,
                    -4, -3, -2, -1, 0,
                    1, 2, 3, 4, 5,
                    6, 7, 8, 9, 10,

                    11, 12, 13, 14, 15,
                    16, 17, 18, 19, 20,
                    21, 22, 23, 24, 25,
                    26, 27, 28, 29, 30
                }, new int[3]{3, 4, 5}
            );

            // axis = 0 2
            R[] grad = new R[]{
                1, 0, 0, 0, 0,
                2, 0, 0, 0, 0,
                3, 0, 0, 0, 0,
                4, 0, 0, 0, 0,
                
                0, 0, 0, 0, 0,
                0, 0, 0, 0, 0,
                0, 0, 0, 0, 0,
                0, 0, 0, 0, 0,

                0, 0, 0, 0, 0,
                0, 0, 0, 0, 0,
                0, 0, 0, 0, 0,
                0, 0, 0, 0, 0,
            };

            Tensor Out1 = input.Min(2).Min(0);

            Out1.Grad = new R[]{1, 2, 3, 4};
            Out1.UseCount++;

            Out1.Backward();


            this._CheckArrayEqual(grad, input.Grad, "TestMinGrad");

            Console.WriteLine("TestMinGrad");
        }

    }
}