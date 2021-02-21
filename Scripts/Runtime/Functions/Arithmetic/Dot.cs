using System.Linq;
using R = System.Double;
using System.Collections.Generic;
using System.Threading.Tasks;

namespace Rein.Functions.Arithmetic{
    public class Dot: BinaryFunction{
        public Dot():base("Dot"){

        }

        protected override Tensor BinaryForward(Tensor left, Tensor right)
        {
            int N = left.Size / left.Shape.Last();
            int M= right.Shape[1];
            int L = right.Shape[0];
            R[] data = new R[N * M];
            int k, k1, k2, k3, j;

            for (int i=0; i < N; i++){
                for (k=0; k < L - 3; k+=4){
                    k1 = k + 1;
                    k2 = k + 2;
                    k3 = k + 3;
                    for (j=0; j < M - 3; j+=4){
                        data[i*M + j] += left.Data[i*L + k] * right.Data[k*M + j]
                            + left.Data[i*L + k1] * right.Data[k1*M + j]
                            + left.Data[i*L + k2] * right.Data[k2*M + j]
                            + left.Data[i*L + k3] * right.Data[k3*M + j];
                        
                        data[i*M + j+1] += left.Data[i*L + k] * right.Data[k*M + j+1]
                            + left.Data[i*L + k1] * right.Data[k1*M + j+1]
                            + left.Data[i*L + k2] * right.Data[k2*M + j+1]
                            + left.Data[i*L + k3] * right.Data[k3*M + j+1];

                        data[i*M + j+2] += left.Data[i*L + k] * right.Data[k*M + j+2]
                            + left.Data[i*L + k1] * right.Data[k1*M + j+2]
                            + left.Data[i*L + k2] * right.Data[k2*M + j+2]
                            + left.Data[i*L + k3] * right.Data[k3*M + j+2];

                        data[i*M + j+3] += left.Data[i*L + k] * right.Data[k*M + j+3]
                            + left.Data[i*L + k1] * right.Data[k1*M + j+3]
                            + left.Data[i*L + k2] * right.Data[k2*M + j+3]
                            + left.Data[i*L + k3] * right.Data[k3*M + j+3];
                    }
                    
                    for(; j < M; j++){
                        data[i*M + j] += left.Data[i*L + k] * right.Data[k*M + j];
                    }
                }

                for (; k < L; k++){
                    k1 = k + 1;
                    k2 = k + 2;
                    k3 = k + 3;
                    for (j=0; j < M - 3; j+=4){
                        data[i*M + j] += left.Data[i*L + k] * right.Data[k*M + j];
                        data[i*M + j+1] += left.Data[i*L + k] * right.Data[k*M + j+1];
                        data[i*M + j+2] += left.Data[i*L + k] * right.Data[k*M + j+2];
                        data[i*M + j+3] += left.Data[i*L + k] * right.Data[k*M + j+3];
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
            int N = this.Left.Size / this.Left.Shape.Last();
            int M= this.Right.Shape[1];
            int L = this.Right.Shape[0];

            int i, j, k, k1, k2, k3;

            for (i=0; i < N; i++){

                for(j=0; j < L; j++){
                    R sum = 0;
                    for(k=0; k < M - 3; k+=4){
                        sum += this.Out.Grad[i * M + k] * this.Right.Data[j * M + k]
                            + this.Out.Grad[i * M + k + 1] * this.Right.Data[j * M + k + 1]
                            + this.Out.Grad[i * M + k + 2] * this.Right.Data[j * M + k + 2]
                            + this.Out.Grad[i * M + k + 3] * this.Right.Data[j * M + k + 3];
                    }

                    for(; k < M; k++){
                        sum += this.Out.Grad[i * M + k] * this.Right.Data[j * M + k];
                    }
                    this.Left.Grad[i * L + j] += sum;
                }
            }

            for (k=0; k < N - 3; k+=4){
                k1 = k + 1;
                k2 = k + 2;
                k3 = k + 3;
                for (i=0; i < L; i++){
                    for (j=0; j < M; j++){
                        this.Right.Grad[i * M + j] += this.Out.Grad[k * M + j] * this.Left.Data[k * L + i]
                        + this.Out.Grad[k1 * M + j] * this.Left.Data[k1 * L + i]
                        + this.Out.Grad[k2 * M + j] * this.Left.Data[k2 * L + i]
                        + this.Out.Grad[k3 * M + j] * this.Left.Data[k3 * L + i];
                    }
                }
            }

            for (; k < N; k++){
                for (i=0; i < L; i++){
                    for (j=0; j < M; j++){
                        this.Right.Grad[i * M + j] += this.Out.Grad[k * M + j] * this.Left.Data[k * L + i];
                    }
                }
            }
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
                        data[i*M + j] += left.Data[i*L + k] * right.Data[k*M + j]
                            + left.Data[i*L + k1] * right.Data[k1*M + j]
                            + left.Data[i*L + k2] * right.Data[k2*M + j]
                            + left.Data[i*L + k3] * right.Data[k3*M + j];
                        
                        data[i*M + j+1] += left.Data[i*L + k] * right.Data[k*M + j+1]
                            + left.Data[i*L + k1] * right.Data[k1*M + j+1]
                            + left.Data[i*L + k2] * right.Data[k2*M + j+1]
                            + left.Data[i*L + k3] * right.Data[k3*M + j+1];

                        data[i*M + j+2] += left.Data[i*L + k] * right.Data[k*M + j+2]
                            + left.Data[i*L + k1] * right.Data[k1*M + j+2]
                            + left.Data[i*L + k2] * right.Data[k2*M + j+2]
                            + left.Data[i*L + k3] * right.Data[k3*M + j+2];

                        data[i*M + j+3] += left.Data[i*L + k] * right.Data[k*M + j+3]
                            + left.Data[i*L + k1] * right.Data[k1*M + j+3]
                            + left.Data[i*L + k2] * right.Data[k2*M + j+3]
                            + left.Data[i*L + k3] * right.Data[k3*M + j+3];
                    }
                    
                    for(; j < M; j++){
                        data[i*M + j] += left.Data[i*L + k] * right.Data[k*M + j];
                    }
                }

                for (; k < L; k++){
                    int k1 = k + 1;
                    int k2 = k + 1;
                    int k3 = k + 1;
                    int j;
                    for (j=0; j < M - 3; j+=4){
                        data[i*M + j] += left.Data[i*L + k] * right.Data[k*M + j];
                        data[i*M + j+1] += left.Data[i*L + k] * right.Data[k*M + j+1];
                        data[i*M + j+2] += left.Data[i*L + k] * right.Data[k*M + j+2];
                        data[i*M + j+3] += left.Data[i*L + k] * right.Data[k*M + j+3];
                    }
                    
                    for(; j < M; j++){
                        data[i*M + j] += left.Data[i*L + k] * right.Data[k*M + j];
                    }
                }
            });

            List<int> shape = new List<int>(left.Shape);
            shape[shape.Count - 1] = right.Shape[1];
            return new Tensor(data, shape);
        }

        // TODO:並列への変更
        protected override void BinaryBackward()
        {
            int N = this.Left.Size / this.Left.Shape.Last();
            int M= this.Right.Shape[1];
            int L = this.Right.Shape[0];

            int i, j, k, k1, k2, k3;

            for (i=0; i < N; i++){
                for(j=0; j < L - 3; j++){
                    R sum = 0;
                    for(k=0; k < M; k+=4){
                        sum += this.Out.Grad[i * M + k] * this.Right.Data[j * M + k]
                            + this.Out.Grad[i * M + k + 1] * this.Right.Data[j * M + k + 1]
                            + this.Out.Grad[i * M + k + 2] * this.Right.Data[j * M + k + 2]
                            + this.Out.Grad[i * M + k + 3] * this.Right.Data[j * M + k + 3];
                    }

                    for(; k < M; k++){
                        sum += this.Out.Grad[i * M + k] * this.Right.Data[j * M + k];
                    }
                    this.Left.Grad[i * L + j] += sum;
                }
            }

            for (k=0; k < N - 3; k+=4){
                k1 = k + 1;
                k2 = k + 2;
                k3 = k + 3;
                for (i=0; i < L; i++){
                    for (j=0; j < M; j++){
                        this.Right.Grad[i * M + j] += this.Out.Grad[k * M + j] * this.Left.Data[k * L + i]
                        + this.Out.Grad[k1 * M + j] * this.Left.Data[k1 * L + i]
                        + this.Out.Grad[k2 * M + j] * this.Left.Data[k2 * L + i]
                        + this.Out.Grad[k3 * M + j] * this.Left.Data[k3 * L + i];
                    }
                }
            }

            for (; k < N; k++){
                for (i=0; i < L; i++){
                    for (j=0; j < M; j++){
                        this.Right.Grad[i * M + j] += this.Out.Grad[k * M + j] * this.Left.Data[k * L + i];
                    }
                }
            }
        }
    }
}