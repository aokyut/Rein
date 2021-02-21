using System;
using System.Collections.Generic;
using System.Collections;
using R = System.Double;
using System.Diagnostics;
using Rein.Functions.Arithmetic;

namespace Rein.Tests
{
    public class TensorTest: Test
    {
        public TensorTest(): base("TensorTest")
        {
            this.Tests = new List<Action>(){
                this.TestAddition,
                this.TestSubstruction,
                this.TestMultiplication,
                this.TestDivision,
                this.TestDot,
                this.TestSum,
                this.TestSumGrad,
                this.TestMean,
                this.TestMeanGrad,
            };


            this.RunTest();
        }

        public void TestAddition()
        {
            var test = new (Tensor, Tensor, Tensor)[] {
                (
                    new Tensor(
                        new R[] {1, 2, 3, 4},
                        new int[] {2, 2}
                    ),
                    new Tensor(
                        new R[] {2, 4, 8, 16},
                        new int[] {2, 2}
                    ),
                    new Tensor(
                        new R[] {3, 6, 11, 20},
                        new int[] {2, 2}
                    )
                ),
                (
                    new Tensor(
                        new R[] {3.2, 4.5, 5.1, 2.0/3.0},
                        new int[] {2, 2}
                    ),
                    new Tensor(
                        new R[] {16, -8, 4, 2},
                        new int[] {2, 2}
                    ),
                    new Tensor(
                        new R[] {19.2, -3.5, 9.1, 8.0/3.0},
                        new int[] {2, 2}
                    )
                )
            };

            foreach (var (ten1, ten2, ten3) in test)
            {
                Tensor resultTensor = ten1 + ten2;
                
                _CheckTensorEqual(ten3, resultTensor, "TestAddition");
            }
            Console.WriteLine("TestAddition");
        }

        public void TestSubstruction(){
            var test = new (Tensor, Tensor, Tensor)[] {
                (
                    new Tensor(
                        new R[] {5, 6, 7, 8},
                        new int[] {2, 2}
                    ),
                    new Tensor(
                        new R[] {2, 4, 8, 16},
                        new int[] {2, 2}
                    ),
                    new Tensor(
                        new R[] {3, 2, -1, -8},
                        new int[] {2, 2}
                    )
                ),
                (
                    new Tensor(
                        new R[] {3.2, 4.5, 5.1, 2.0/3.0},
                        new int[] {2, 2}
                    ),
                    new Tensor(
                        new R[] {16, -8, 4, 2.0},
                        new int[] {2, 2}
                    ),
                    new Tensor(
                        new R[] {-12.8, 12.5, 1.1, -4.0/3.0},
                        new int[] {2, 2}
                    )
                )
            };

            foreach (var (ten1, ten2, ten3) in test)
            {
                Tensor resultTensor = ten1 - ten2;
                
                _CheckTensorEqual(ten3, resultTensor, "TestSubstruction");
            }
            Console.WriteLine("TestSubstruction");
        }

        public void TestMultiplication(){
            var test = new (Tensor, Tensor, Tensor)[] {
                (
                    new Tensor(
                        new R[] {5, 6, 7, 8},
                        new int[] {2, 2}
                    ),
                    new Tensor(
                        new R[] {2, -4, 8, -16},
                        new int[] {2, 2}
                    ),
                    new Tensor(
                        new R[] {10, -24, 56, -128},
                        new int[] {2, 2}
                    )
                ),
                (
                    new Tensor(
                        new R[] {3.2, 4.5, 5.1, 2.0/3.0},
                        new int[] {2, 2}
                    ),
                    new Tensor(
                        new R[] {7, -8, 4, 2},
                        new int[] {2, 2}
                    ),
                    new Tensor(
                        new R[] {22.4, -36, 20.4, 4.0/3.0},
                        new int[] {2, 2}
                    )
                )
            };

            foreach (var (ten1, ten2, ten3) in test)
            {
                Tensor resultTensor = ten1 * ten2;
                
                _CheckTensorEqual(ten3, resultTensor, "TestMultiplication");
            }
            Console.WriteLine("TestMultiplication");
        }

        public void TestDivision(){
            var test = new (Tensor, Tensor, Tensor)[] {
                (
                    new Tensor(
                        new R[] {5, 6, 7, 8},
                        new int[] {2, 2}
                    ),
                    new Tensor(
                        new R[] {2, 4, 8, 16},
                        new int[] {2, 2}
                    ),
                    new Tensor(
                        new R[] {2.5, 1.5, 0.875, 0.5},
                        new int[] {2, 2}
                    )
                ),
                (
                    new Tensor(
                        new R[] {3.2, 4.5, 5.1, 2.0/3.0},
                        new int[] {2, 2}
                    ),
                    new Tensor(
                        new R[] {16, -8, 4, 2},
                        new int[] {2, 2}
                    ),
                    new Tensor(
                        new R[] {3.2/16.0, -4.5/8.0, 5.1/4.0, 1.0/3.0},
                        new int[] {2, 2}
                    )
                )
            };

            foreach (var (ten1, ten2, ten3) in test)
            {
                Tensor resultTensor = ten1 / ten2;
                
                _CheckTensorEqual(ten3, resultTensor, "TestDivision");
            }
            Console.WriteLine("TestDivision");
        }

        public void TestDot(){
            var tests = new (Tensor, Tensor, Tensor, R[], R[])[] {
                (
                    new Tensor(
                        new R[]{
                            2, 5,
                            1, 4,
                            -1, 3
                        },
                        new int[]{3, 2}
                    ),
                    new Tensor(
                        new R[]{
                            1, 3, 
                            8, -2
                        },
                        new int[]{2, 2}
                    ),
                    new Tensor(
                        new R[]{
                            42, -4,
                            33, -5,
                            23, -9
                        },
                        new int[]{3, 2}
                    ),
                    new R[]{
                        4, 6,
                        4, 6,
                        4, 6
                    },
                    new R[]{
                        2, 2,
                        12, 12
                    }
                ),
            };
            
            foreach (var (ten1, ten2, ten3, r1, r2) in tests)
            {
                Tensor resultTensor = new Dot().Forward(ten1, ten2);
                resultTensor.Grad = new R[]{
                    1, 1, 1, 1, 1, 1
                };

                resultTensor.UseCount = 1;
                resultTensor.Backward();
                
                _CheckTensorEqual(ten3, resultTensor, "TestDot");
                _CheckArrayEqual(ten1.Grad, r1, "TestDot-Grad1");
                _CheckArrayEqual(ten2.Grad, r2, "TestDot-Grad2");
            }
            Console.WriteLine("TestDot");
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

            Tensor Out1 = new Rein.Functions.Set.Sum(new List<int>(){2}).Forward(input);
            Tensor Out2 = new Rein.Functions.Set.Sum(new List<int>(){1}).Forward(input);
            Tensor Out3 = new Rein.Functions.Set.Sum(new List<int>(){0, 2}).Forward(input);

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

            Tensor Out1 = new Rein.Functions.Set.Sum(new List<int>(){0, 2}).Forward(input);

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

            Tensor Out1 = new Rein.Functions.Set.Mean(new List<int>(){2}).Forward(input);
            Tensor Out2 = new Rein.Functions.Set.Mean(new List<int>(){1}).Forward(input);
            Tensor Out3 = new Rein.Functions.Set.Mean(new List<int>(){0, 2}).Forward(input);

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

            Tensor Out1 = new Rein.Functions.Set.Sum(new List<int>(){0, 2}).Forward(input);

            Out1.Grad = new R[]{1, 2, 3, 4};
            Out1.UseCount++;

            Out1.Backward();


            this._CheckArrayEqual(grad, input.Grad, "TestMeanGrad");

            Console.WriteLine("TestMeanGrad");
        }

        public void ProfileDot(){
            Stopwatch sw1 = new Stopwatch();
            Stopwatch sw2 = new Stopwatch();
            // int[] shape = new int[2]{ , 300 };
            int loopNum = 10;
            int[] sizes = new int[]{ 50, 100, 150, 200, 250, 300, 350, 400, 450, 500, 600, 700, 800, 900, 1000 };

            foreach (int size in sizes){
                // Console.WriteLine(size);
                sw1.Reset();
                sw2.Reset();
                for (int i=0; i < loopNum; i++){
                    Tensor left = new Tensor(
                        new int[2]{size, size + 10}
                    );
                    Tensor right = new Tensor(
                        new int[2]{size + 10, size}
                    );

                    sw1.Start();
                    Tensor out1 = new Dot().Forward(left, right);
                    sw1.Stop();

                    sw2.Start();
                    Tensor out2 = new DotParallel().Forward(left, right);
                    sw2.Stop();

                    System.Threading.Thread.Sleep(100);
                }

                // Console.WriteLine($"交換+アンロー：{sw1.ElapsedMilliseconds}/{loopNum}ms");
                // Console.WriteLine($"交換：{sw2.ElapsedMilliseconds}/{loopNum}ms");
                Console.WriteLine($"|{size}|{sw1.ElapsedMilliseconds}ms|{sw2.ElapsedMilliseconds}ms|");
            }
        }

        private void _CheckTensorEqual(Tensor expected, Tensor actual, string testName="")
        {
            this._CheckArrayEqual(expected.Data, actual.Data);
        }

        private void _CheckArrayEqual(R[] expected, R[] actual, string testName=""){
            Debug.Assert(
                this._IsArrayEqual(expected, actual),
                $"[{testName}]Array Equal Check failed expected:{this.DebugArray(expected)}, actual:{this.DebugArray(actual)}"
            );
        }

        private bool _IsArrayEqual(R[] expected, R[] actual){
            if (expected.Length != actual.Length) return false;
            for (int i = 0; i < expected.Length; i++){
                double dis = Math.Abs(expected[i] - actual[i]);
                double ex = Math.Abs(expected[i]);
                if (dis <= (0.0001 * ex + 0.0001)){
                    continue;
                }else{
                    return false;
                }
            }
            return true;
        }
    }
}