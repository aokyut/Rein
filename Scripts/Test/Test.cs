using System;
using System.Collections.Generic;
using R = System.Double;
using System.Diagnostics;

namespace Rein.Tests
{
    public abstract class Test{
        protected List<Action> Tests;
        protected string Name;

        public Test(string name){
            this.Name = name;
        }

        protected void RunTest(){
            foreach ( var test in this.Tests){
                test();
            }

            Console.WriteLine($"{Name} Succeed!");
        }

        protected string DebugArray<T>(T[] array) where T: IComparable{
            string str = "";
            foreach(var item in array){
                str += item.ToString();
                str += ", ";
            }

            return str;
        }

        protected void _CheckTensorEqual(Tensor expected, Tensor actual, string testName="")
        {
            this._CheckArrayEqual(expected.Data, actual.Data);
        }

        protected void _CheckArrayEqual(R[] expected, R[] actual, string testName=""){
            Debug.Assert(
                this._IsArrayEqual(expected, actual),
                $"[{testName}]Array Equal Check failed expected:{this.DebugArray(expected)}, actual:{this.DebugArray(actual)}"
            );
        }

        protected bool _IsArrayEqual(R[] expected, R[] actual){
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