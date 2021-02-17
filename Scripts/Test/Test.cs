using System;
using System.Collections.Generic;

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
    }
}