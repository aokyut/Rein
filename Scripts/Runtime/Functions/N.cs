using Rein.Functions.Layer;
using Rein.Functions;

namespace Rein{
    public static class N{
        public static IFunction Linear(int inputSize, int outputSize, bool bias = true){
            return new Linear(inputSize, outputSize, bias);
        }

        public static IFunction ReLU(){
            return new Lambda(
                "ReLU",
                (x) => x > 0 ? x : 0,
                (x) => x > 0 ? 1 : 0
            );
        }

        
    }
}