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

    public class InvalidLengthException: Exception
    {
        public InvalidLengthException()
        : base()
        {
        }
        public InvalidLengthException(string message)
            : base(message)
        {
        }
        public InvalidLengthException(string message, Exception innerException)
            : base(message, innerException)
        {
        }
        //逆シリアル化コンストラクタ。このクラスの逆シリアル化のために必須。
        //アクセス修飾子をpublicにしないこと！（詳細は後述）
        protected InvalidLengthException(SerializationInfo info, StreamingContext context)
            : base(info, context)
        {
        }
    }
    public class InvalidShapeException: Exception
    {
        public InvalidShapeException()
        : base()
        {
        }
        public InvalidShapeException(string message)
            : base(message)
        {
        }
        public InvalidShapeException(string message, Exception innerException)
            : base(message, innerException)
        {
        }
        //逆シリアル化コンストラクタ。このクラスの逆シリアル化のために必須。
        //アクセス修飾子をpublicにしないこと！（詳細は後述）
        protected InvalidShapeException(SerializationInfo info, StreamingContext context)
            : base(info, context)
        {
        }
    }
}
