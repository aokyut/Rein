using System;
using System.Collections.Generic;
using System.Collections;
using R = System.Double;
using System.Diagnostics;
using Rein;

namespace Rein.Tests
{
    public class TensorTest: Test
    {
        public TensorTest(): base("TensorTest")
        {
            this.Tests = new List<Action>(){
                // this.TestAddition,
                // this.TestSubstruction,
                // this.TestMultiplication,
                // this.TestDivision,
                this.TestDot,
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
                        new R[] {12.8, 12.5, 1.1, -4.0/3.0},
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
                Tensor resultTensor = new Rein.Functions.Dot().Forward(ten1, ten2);
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
                    Tensor out1 = new Rein.Functions.Dot().Forward(left, right);
                    sw1.Stop();

                    sw2.Start();
                    Tensor out2 = new Rein.Functions.DotParallel().Forward(left, right);
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
                double ex = Math.Abs(expected[i]);
                double ac = Math.Abs(actual[i]);
                if (ex * 0.9999 <= ac && ac <= ex * 1.0001){
                    continue;
                }else{
                    return false;
                }
            }
            return true;
        }
    }
}