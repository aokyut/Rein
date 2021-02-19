using System.Linq;
using R = System.Double;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace Rein.Functions{
    public class Dot: BinaryFunction{
        public Dot():base("Dot"){

        }

        protected override Tensor BinaryForward(Tensor left, Tensor right)
        {
            int N = left.Size / left.Shape.Last();
            int M= right.Shape[1];
            int L = right.Shape[0];
            R[] data = new R[N * M];

            for (int i=0; i < N; i++){
                int k;
                for (k=0; k < L - 3; k+=4){
                    int k1 = k + 1;
                    int k2 = k + 1;
                    int k3 = k + 1;
                    int j;
                    for (j=0; j < M - 3; j+=4){
                        data[i*L + j] += left.Data[i*L + k] * right.Data[k*L + j];
                        data[i*L + j+1] += left.Data[i*L + k] * right.Data[k*L + j+1];
                        data[i*L + j+2] += left.Data[i*L + k] * right.Data[k*L + j+2];
                        data[i*L + j+3] += left.Data[i*L + k] * right.Data[k*L + j+3];

                        data[i*L + j] += left.Data[i*L + k1] * right.Data[k1*L + j];
                        data[i*L + j+1] += left.Data[i*L + k1] * right.Data[k1*L + j+1];
                        data[i*L + j+2] += left.Data[i*L + k1] * right.Data[k1*L + j+2];
                        data[i*L + j+3] += left.Data[i*L + k1] * right.Data[k1*L + j+3];

                        data[i*L + j] += left.Data[i*L + k2] * right.Data[k2*L + j];
                        data[i*L + j+1] += left.Data[i*L + k2] * right.Data[k2*L + j+1];
                        data[i*L + j+2] += left.Data[i*L + k2] * right.Data[k2*L + j+2];
                        data[i*L + j+3] += left.Data[i*L + k2] * right.Data[k2*L + j+3];

                        data[i*L + j] += left.Data[i*L + k3] * right.Data[k3*L + j];
                        data[i*L + j+1] += left.Data[i*L + k3] * right.Data[k3*L + j+1];
                        data[i*L + j+2] += left.Data[i*L + k3] * right.Data[k3*L + j+2];
                        data[i*L + j+3] += left.Data[i*L + k3] * right.Data[k3*L + j+3];
                    }
                    
                    for(; j < M; j++){
                        data[i*L + j] += left.Data[i*L + k] * right.Data[k*L + j];
                    }
                }

                for (; k < L; k++){
                    int k1 = k + 1;
                    int k2 = k + 1;
                    int k3 = k + 1;
                    int j;
                    for (j=0; j < M - 3; j+=4){
                        data[i*L + j] += left.Data[i*L + k] * right.Data[k*L + j];
                        data[i*L + j+1] += left.Data[i*L + k] * right.Data[k*L + j+1];
                        data[i*L + j+2] += left.Data[i*L + k] * right.Data[k*L + j+2];
                        data[i*L + j+3] += left.Data[i*L + k] * right.Data[k*L + j+3];
                    }
                    
                    for(; j < M; j++){
                        data[i*L + j] += left.Data[i*L + k] * right.Data[k*L + j];
                    }
                }
            }

            List<int> shape = new List<int>(left.Shape);
            shape[shape.Count - 1] = right.Shape[1];
            return new Tensor(data, shape);
        }

        protected override void BinaryBackward()
        {
            throw new System.NotImplementedException();
        }
    }
    public class DotParallel: BinaryFunction{
        public DotParallel():base("DotParallel"){

        }

        protected override Tensor BinaryForward(Tensor left, Tensor right)
        {
            int N = left.Size / left.Shape.Last();
            int M= right.Shape[1];
            int L = right.Shape[0];
            R[] data = new R[N * M];

            Parallel.For(0, N, i => {
                int k;
                for (k=0; k < L - 3; k+=4){
                    int k1 = k + 1;
                    int k2 = k + 1;
                    int k3 = k + 1;
                    int j;
                    for (j=0; j < M - 3; j+=4){
                        data[i*L + j] += left.Data[i*L + k] * right.Data[k*L + j];
                        data[i*L + j+1] += left.Data[i*L + k] * right.Data[k*L + j+1];
                        data[i*L + j+2] += left.Data[i*L + k] * right.Data[k*L + j+2];
                        data[i*L + j+3] += left.Data[i*L + k] * right.Data[k*L + j+3];

                        data[i*L + j] += left.Data[i*L + k1] * right.Data[k1*L + j];
                        data[i*L + j+1] += left.Data[i*L + k1] * right.Data[k1*L + j+1];
                        data[i*L + j+2] += left.Data[i*L + k1] * right.Data[k1*L + j+2];
                        data[i*L + j+3] += left.Data[i*L + k1] * right.Data[k1*L + j+3];

                        data[i*L + j] += left.Data[i*L + k2] * right.Data[k2*L + j];
                        data[i*L + j+1] += left.Data[i*L + k2] * right.Data[k2*L + j+1];
                        data[i*L + j+2] += left.Data[i*L + k2] * right.Data[k2*L + j+2];
                        data[i*L + j+3] += left.Data[i*L + k2] * right.Data[k2*L + j+3];

                        data[i*L + j] += left.Data[i*L + k3] * right.Data[k3*L + j];
                        data[i*L + j+1] += left.Data[i*L + k3] * right.Data[k3*L + j+1];
                        data[i*L + j+2] += left.Data[i*L + k3] * right.Data[k3*L + j+2];
                        data[i*L + j+3] += left.Data[i*L + k3] * right.Data[k3*L + j+3];
                    }
                    
                    for(; j < M; j++){
                        data[i*L + j] += left.Data[i*L + k] * right.Data[k*L + j];
                    }
                }

                for (; k < L; k++){
                    int k1 = k + 1;
                    int k2 = k + 1;
                    int k3 = k + 1;
                    int j;
                    for (j=0; j < M - 3; j+=4){
                        data[i*L + j] += left.Data[i*L + k] * right.Data[k*L + j];
                        data[i*L + j+1] += left.Data[i*L + k] * right.Data[k*L + j+1];
                        data[i*L + j+2] += left.Data[i*L + k] * right.Data[k*L + j+2];
                        data[i*L + j+3] += left.Data[i*L + k] * right.Data[k*L + j+3];
                    }
                    
                    for(; j < M; j++){
                        data[i*L + j] += left.Data[i*L + k] * right.Data[k*L + j];
                    }
                }
            });

            List<int> shape = new List<int>(left.Shape);
            shape[shape.Count - 1] = right.Shape[1];
            return new Tensor(data, shape);
        }

        protected override void BinaryBackward()
        {
            throw new System.NotImplementedException();
        }
    }
}