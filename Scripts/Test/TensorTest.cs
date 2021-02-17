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
                this.TestAddition,
                this.TestSubstruction,
                this.TestMultiplication,
                this.TestDivision
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
                        new R[] {3.2, 4.5, 5.1, 2/3},
                        new int[] {2, 2}
                    ),
                    new Tensor(
                        new R[] {16, -8, 4, 2},
                        new int[] {2, 2}
                    ),
                    new Tensor(
                        new R[] {19.2, -3.5, 9.1, 8/3},
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
                        new R[] {3.2, 4.5, 5.1, 2/3},
                        new int[] {2, 2}
                    ),
                    new Tensor(
                        new R[] {16, -8, 4, 2},
                        new int[] {2, 2}
                    ),
                    new Tensor(
                        new R[] {12.8, 12.5, 1.1, -4/3},
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
                        new R[] {3.2, 4.5, 5.1, 2/3},
                        new int[] {2, 2}
                    ),
                    new Tensor(
                        new R[] {7, -8, 4, 2},
                        new int[] {2, 2}
                    ),
                    new Tensor(
                        new R[] {22.4, -36, 20.4, 4/3},
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
                        new R[] {3.2, 4.5, 5.1, 2/3},
                        new int[] {2, 2}
                    ),
                    new Tensor(
                        new R[] {16, -8, 4, 2},
                        new int[] {2, 2}
                    ),
                    new Tensor(
                        new R[] {3.2/16, -4.5/8, 5.1/4, 1/3},
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

        private void _CheckTensorEqual(Tensor expected, Tensor actual, string testName="")
        {
            Debug.Assert(
                !StructuralComparisons.Equals(expected.Data, actual.Data),
                $"[{testName}]Tensor Equal Check failed expected:{this.DebugArray(expected.Data)}, actual:{this.DebugArray(actual.Data)}"
            );
        }
    }
}