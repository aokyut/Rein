﻿using Rein;

namespace Rein.Functions
{
    public interface IFunction
    {
        public Tensor[] Forward(Tensor[] inputs);
        public void Backward();
    }
}