using System;

namespace Rein.Functions{
    public static class Math{
        
        // -X
        public static Tensor Minus(Tensor In){
            return new Lambda(
                "Minus",
                (x) => -x,
                (x) => -1
            ).Forward(In);
        }

        // ReLU
        public static Tensor ReLU(Tensor In){
            return new Lambda(
                "ReLU",
                (x) => x > 0 ? x : 0,
                (x) => x > 0 ? 1 : 0
            ).Forward(In);
        }

        // Exponential
        public static Tensor Exp(Tensor In){
            return new Lambda(
                "Exp",
                (x) => System.Math.Exp(x),
                (x) => System.Math.Exp(x)
            ).Forward(In);
        }

        // Log
        public static Tensor Log(Tensor In){
            return new Lambda(
                "Log_e",
                (x) => System.Math.Log(x),
                (x) => 1 / x
            ).Forward(In);
        }
    }
}