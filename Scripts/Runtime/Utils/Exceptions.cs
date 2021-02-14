using System;
using System.Runtime.Serialization;

namespace Rein.Utils.Exceptions
{
    [Serializable]
    public class InvalidSizeException: Exception
    {
        public InvalidSizeException()
        : base()
        {
        }

        public InvalidSizeException(string message)
            : base(message)
        {
        }

        public InvalidSizeException(string message, Exception innerException)
            : base(message, innerException)
        {
        }

        //逆シリアル化コンストラクタ。このクラスの逆シリアル化のために必須。
        //アクセス修飾子をpublicにしないこと！（詳細は後述）
        protected InvalidSizeException(SerializationInfo info, StreamingContext context)
            : base(info, context)
        {
        }
    }
}
