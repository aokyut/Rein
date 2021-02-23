using Rein.Functions;
using Rein.Functions.Arithmetic;
using Rein.Functions.Layer;
using R = System.Double;

namespace Rein{
    public static class F{
        // -X
        public static Tensor Minus(Tensor In){
            return new Lambda(
                "Minus",
                (x) => -x,
                (x) => -1
            ).Forward(In);
        }

        public static Tensor AddConst(Tensor In, R r){
            return new Lambda(
                "AddConst",
                (x) => x + r,
                (x) => 1
            ).Forward(In);
        }
        public static Tensor MulConst(Tensor In, R r){
            return new Lambda(
                "MulConst",
                (x) => x * r,
                (x) => r
            ).Forward(In);
        }

        public static Tensor DivConst(Tensor In, R r){
            return new Lambda(
                "DivCosnt",
                (x) => r / x,
                (x) => - r / (x * x)
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

        // Dot
        public static Tensor Dot(Tensor left, Tensor right){
            return new Dot().Forward(left, right);
        }

        // SetFunction
        public static Tensor Sum(Tensor tensor){
            return tensor.Sum();
        }
        public static Tensor Mean(Tensor tensor){
            return tensor.Mean();
        }
        public static Tensor Max(Tensor tensor){
            return tensor.Max();
        }
        public static Tensor Min(Tensor tensor){
            return tensor.Min();
        }

        // Layer
        public static Tensor Linear(Tensor tensor, int inputSize, int outputSize, bool bias = true){
            return new Linear(inputSize, outputSize, bias).Forward(tensor);
        }
    }
}